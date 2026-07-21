"""Fair retrieval and end-to-end evaluation on a matching SQuAD index."""

from __future__ import annotations

import argparse
import collections
import re
import string
from pathlib import Path

from build_squad_index import DEFAULT_OUTPUT_DIR
from qa_pipeline import RetrievalReaderQA


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return float(prediction == "")
    return float(any(normalize_answer(prediction) == normalize_answer(gold) for gold in gold_answers))


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return float(prediction == "")

    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    scores = []
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            scores.append(0.0)
            continue
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(gold_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores)


def answer_in_passage(gold_answers: list[str], passage: str) -> bool:
    normalized_passage = normalize_answer(passage)
    return any(normalize_answer(gold) in normalized_passage for gold in gold_answers)


def evaluate(
    data_dir: Path,
    dataset_name: str,
    split: str,
    sample_size: int,
    top_k: int,
    model_kind: str,
    reader_checkpoint: Path | None,
    no_answer_threshold: float,
) -> None:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=f"{split}[:{sample_size}]")
    qa = RetrievalReaderQA.from_artifacts(
        data_dir=data_dir,
        reader_checkpoint=reader_checkpoint,
        model_kind=model_kind,
    )

    retrieval_hits = 0
    answerable_total = 0
    exact_total = 0.0
    f1_total = 0.0
    no_answer_correct = 0
    no_answer_total = 0
    examples_total = 0

    for example in dataset:
        question = example["question"]
        gold_answers = example["answers"]["text"]
        result = qa.answer(question, top_k=top_k, no_answer_threshold=no_answer_threshold)
        prediction = result.answer.text

        examples_total += 1
        exact_total += exact_match(prediction, gold_answers)
        f1_total += token_f1(prediction, gold_answers)

        if gold_answers:
            answerable_total += 1
            if any(answer_in_passage(gold_answers, item["passage"]) for item in result.retrieved_passages):
                retrieval_hits += 1
        else:
            no_answer_total += 1
            no_answer_correct += int(prediction == "")

    retrieval_recall = retrieval_hits / answerable_total if answerable_total else 0.0
    exact = exact_total / examples_total if examples_total else 0.0
    f1 = f1_total / examples_total if examples_total else 0.0
    no_answer_accuracy = no_answer_correct / no_answer_total if no_answer_total else 0.0

    print(f"Dataset: {dataset_name} {split}[:{sample_size}]")
    print(f"Index: {data_dir}")
    print(f"Reader: {model_kind}")
    print(f"Retrieval recall@{top_k}: {retrieval_recall:.4f} ({retrieval_hits}/{answerable_total})")
    print(f"End-to-end exact match: {exact:.4f}")
    print(f"End-to-end F1: {f1:.4f}")
    if no_answer_total:
        print(f"No-answer accuracy: {no_answer_accuracy:.4f} ({no_answer_correct}/{no_answer_total})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and reader on a matching SQuAD index.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="rajpurkar/squad_v2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-kind", choices=["bert_baseline", "bert_drqa_attention"], default="bert_baseline")
    parser.add_argument("--reader-checkpoint", type=Path, default=None)
    parser.add_argument("--no-answer-threshold", type=float, default=0.0)
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=args.sample_size,
        top_k=args.top_k,
        model_kind=args.model_kind,
        reader_checkpoint=args.reader_checkpoint,
        no_answer_threshold=args.no_answer_threshold,
    )


if __name__ == "__main__":
    main()
