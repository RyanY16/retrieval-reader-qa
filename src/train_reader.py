"""Train BERT reader variants on a SQuAD v2 subset."""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from reader import BertDrQAQuestionAttentionForQA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "bert-base-uncased"
SEED = 42
MAX_LENGTH = 384
DOC_STRIDE = 128


def add_reader_passages(batch):
    return {"passage": batch["context"]}


def prepare_datasets(subset_fraction: float):
    raw_squad = load_dataset("rajpurkar/squad_v2")
    train_count = int(len(raw_squad["train"]) * subset_fraction)
    validation_count = int(len(raw_squad["validation"]) * subset_fraction)

    train_data = raw_squad["train"].shuffle(seed=SEED).select(range(train_count))
    validation_data = raw_squad["validation"].shuffle(seed=SEED).select(range(validation_count))
    data = DatasetDict({"train": train_data, "validation": validation_data})
    return data.map(add_reader_passages, batched=True)


def build_feature_preparers(tokenizer):
    def prepare_train_features(examples):
        tokenized = tokenizer(
            [q.strip() for q in examples["question"]],
            examples["passage"],
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        special_tokens_mask = tokenized.pop("special_tokens_mask")

        start_positions, end_positions, example_ids = [], [], []
        question_token_mask, context_token_mask = [], []

        for feature_index, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][feature_index]
            sequence_ids = tokenized.sequence_ids(feature_index)
            sample_index = sample_mapping[feature_index]
            answers = examples["answers"][sample_index]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            example_ids.append(examples["id"][sample_index])

            question_token_mask.append([
                int(sequence_ids[i] == 0 and special_tokens_mask[feature_index][i] == 0)
                for i in range(len(input_ids))
            ])
            context_token_mask.append([
                int(sequence_ids[i] == 1 and special_tokens_mask[feature_index][i] == 0)
                for i in range(len(input_ids))
            ])

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            answer_start = answers["answer_start"][0]
            answer_end = answer_start + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= answer_start and offsets[token_end_index][1] >= answer_end):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start:
                    token_start_index += 1
                while offsets[token_end_index][1] >= answer_end:
                    token_end_index -= 1
                start_positions.append(token_start_index - 1)
                end_positions.append(token_end_index + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized["example_id"] = example_ids
        tokenized["question_token_mask"] = question_token_mask
        tokenized["context_token_mask"] = context_token_mask
        return tokenized

    def prepare_validation_features(examples):
        tokenized = tokenizer(
            [q.strip() for q in examples["question"]],
            examples["passage"],
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        special_tokens_mask = tokenized.pop("special_tokens_mask")
        example_ids, question_token_mask, context_token_mask = [], [], []

        for feature_index in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(feature_index)
            sample_index = sample_mapping[feature_index]
            example_ids.append(examples["id"][sample_index])
            question_token_mask.append([
                int(sequence_ids[i] == 0 and special_tokens_mask[feature_index][i] == 0)
                for i in range(len(tokenized["input_ids"][feature_index]))
            ])
            context_token_mask.append([
                int(sequence_ids[i] == 1 and special_tokens_mask[feature_index][i] == 0)
                for i in range(len(tokenized["input_ids"][feature_index]))
            ])
            tokenized["offset_mapping"][feature_index] = [
                offset if sequence_ids[i] == 1 else None
                for i, offset in enumerate(tokenized["offset_mapping"][feature_index])
            ]

        tokenized["example_id"] = example_ids
        tokenized["question_token_mask"] = question_token_mask
        tokenized["context_token_mask"] = context_token_mask
        return tokenized

    return prepare_train_features, prepare_validation_features


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    for example_index, example in enumerate(examples):
        min_null_score = None
        valid_answers = []
        context = example["passage"]

        for feature_index in features_per_example[example_index]:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_score = start_logits[0] + end_logits[0]
            min_null_score = cls_score if min_null_score is None else min(min_null_score, cls_score)

            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    start_char, _ = offset_mapping[start_index]
                    _, end_char = offset_mapping[end_index]
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char:end_char],
                    })

        best_answer = max(valid_answers, key=lambda x: x["score"]) if valid_answers else {"text": "", "score": 0.0}
        predictions[example["id"]] = "" if min_null_score is not None and min_null_score > best_answer["score"] else best_answer["text"]

    formatted_predictions = [
        {"id": example_id, "prediction_text": text, "no_answer_probability": float(text == "")}
        for example_id, text in predictions.items()
    ]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return formatted_predictions, references


def compute_squad_v2_metrics(predictions, references):
    squad_v2_metric = evaluate.load("squad_v2")
    overall = squad_v2_metric.compute(predictions=predictions, references=references)
    refs_by_id = {ref["id"]: ref for ref in references}
    true_positive = false_positive = true_negative = false_negative = 0

    for pred in predictions:
        ref = refs_by_id[pred["id"]]
        gold_answerable = len(ref["answers"]["text"]) > 0
        predicted_answerable = pred["prediction_text"] != ""

        if gold_answerable and predicted_answerable:
            true_positive += 1
        elif not gold_answerable and predicted_answerable:
            false_positive += 1
        elif not gold_answerable and not predicted_answerable:
            true_negative += 1
        else:
            false_negative += 1

    total = true_positive + false_positive + true_negative + false_negative
    answerability_accuracy = (true_positive + true_negative) / total * 100 if total else 0.0
    answerability_precision = true_positive / (true_positive + false_positive) * 100 if true_positive + false_positive else 0.0
    answerability_recall = true_positive / (true_positive + false_negative) * 100 if true_positive + false_negative else 0.0
    answerability_f1 = (
        2 * answerability_precision * answerability_recall / (answerability_precision + answerability_recall)
        if answerability_precision + answerability_recall else 0.0
    )

    return {
        "overall_em": overall.get("exact", 0.0),
        "overall_f1": overall.get("f1", 0.0),
        "answerability_accuracy": answerability_accuracy,
        "answerability_precision": answerability_precision,
        "answerability_recall": answerability_recall,
        "answerability_f1": answerability_f1,
    }


def build_model(model_kind: str):
    if model_kind == "bert_baseline":
        return AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    if model_kind == "bert_drqa_attention":
        return BertDrQAQuestionAttentionForQA.from_pretrained(MODEL_NAME)
    raise ValueError(f"Unknown model kind: {model_kind}")


def train_one_model(model_kind: str, data, train_features, validation_features, output_dir: Path, args) -> dict:
    model = build_model(model_kind)
    train_dataset = train_features.remove_columns(["example_id"])
    eval_dataset = validation_features.remove_columns(["example_id", "offset_mapping"])
    model_output_dir = output_dir / model_kind

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        seed=SEED,
        data_seed=SEED,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    start = time.perf_counter()
    trainer.train()
    train_time_sec = time.perf_counter() - start
    trainer.save_model(str(model_output_dir / "final"))

    start = time.perf_counter()
    raw_predictions = trainer.predict(eval_dataset).predictions
    inference_time_sec = time.perf_counter() - start

    predictions, references = postprocess_qa_predictions(data["validation"], validation_features, raw_predictions)
    metrics = compute_squad_v2_metrics(predictions, references)

    result = {
        "model": model_kind,
        "subset_fraction": args.subset_fraction,
        "train_examples": len(data["train"]),
        "validation_examples": len(data["validation"]),
        "train_time_sec": round(train_time_sec, 2),
        "inference_time_sec": round(inference_time_sec, 2),
        **{key: round(value, 2) for key, value in metrics.items()},
    }

    with (model_output_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT reader variants on SQuAD v2.")
    parser.add_argument("--subset-fraction", type=float, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-kind", choices=["bert_baseline", "bert_drqa_attention", "all"], default="all")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--logging-steps", type=int, default=100)
    args = parser.parse_args()

    set_seed(SEED)
    pct = int(args.subset_fraction * 100)
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / f"reader_{pct}pct_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    data = prepare_datasets(args.subset_fraction)
    prepare_train_features, prepare_validation_features = build_feature_preparers(tokenizer)
    train_features = data["train"].map(
        prepare_train_features,
        batched=True,
        remove_columns=data["train"].column_names,
    )
    validation_features = data["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=data["validation"].column_names,
    )

    model_kinds = ["bert_baseline", "bert_drqa_attention"] if args.model_kind == "all" else [args.model_kind]
    results = [
        train_one_model(model_kind, data, train_features, validation_features, output_dir, args)
        for model_kind in model_kinds
    ]

    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
