"""Extractive BERT readers used after passage retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CHECKPOINT = PROJECT_ROOT / "outputs" / "reader_squad_v2_1pct_saved_outputs" / "bert_baseline" / "final"
DEFAULT_DRQA_CHECKPOINT = PROJECT_ROOT / "outputs" / "reader_squad_v2_1pct_saved_outputs" / "bert_drqa_attention" / "final"
DEFAULT_TOKENIZER_NAME = "bert-base-uncased"


class BertDrQAQuestionAttentionForQA(BertPreTrainedModel):
    """BERT QA model with DrQA-style question-to-context attention."""

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.similarity = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        question_token_mask=None,
        context_token_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if question_token_mask is None:
            question_token_mask = ((token_type_ids == 0) & (attention_mask == 1)).long()
        if context_token_mask is None:
            context_token_mask = ((token_type_ids == 1) & (attention_mask == 1)).long()

        q_mask = question_token_mask.bool()
        c_mask = context_token_mask.bool()

        similarity_scores = torch.matmul(self.similarity(sequence_output), sequence_output.transpose(1, 2))
        similarity_scores = similarity_scores.masked_fill(~q_mask[:, None, :], torch.finfo(similarity_scores.dtype).min)
        attention_weights = torch.softmax(similarity_scores, dim=-1)
        aligned_question = torch.matmul(attention_weights, sequence_output)

        combined = torch.cat([sequence_output, aligned_question, sequence_output * aligned_question], dim=-1)
        enhanced_output = self.projection(combined)

        valid_answer_positions = c_mask.clone()
        valid_answer_positions[:, 0] = True
        enhanced_output = torch.where(valid_answer_positions[:, :, None], enhanced_output, sequence_output)

        logits = self.qa_outputs(enhanced_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        invalid_positions = ~valid_answer_positions
        start_logits = start_logits.masked_fill(invalid_positions, torch.finfo(start_logits.dtype).min)
        end_logits = end_logits.masked_fill(invalid_positions, torch.finfo(end_logits.dtype).min)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            total_loss = (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class ReaderAnswer:
    text: str
    score: float
    no_answer_score: float
    passage_index: int
    start_char: int | None = None
    end_char: int | None = None


class ExtractiveReader:
    """Runs a trained extractive QA model over retrieved passages."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str | None = None,
        max_length: int = 384,
        doc_stride: int = 128,
        max_answer_length: int = 30,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | None = None,
        model_kind: str = "bert_baseline",
        tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
        **kwargs,
    ) -> "ExtractiveReader":
        checkpoint_path = _resolve_checkpoint(checkpoint, model_kind)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        if model_kind == "bert_drqa_attention":
            model = BertDrQAQuestionAttentionForQA.from_pretrained(checkpoint_path)
        elif model_kind == "bert_baseline":
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_path)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        return cls(model=model, tokenizer=tokenizer, **kwargs)

    @torch.no_grad()
    def answer(self, question: str, passages: list[str], n_best_size: int = 20) -> ReaderAnswer:
        if not passages:
            return ReaderAnswer(text="", score=float("-inf"), no_answer_score=0.0, passage_index=-1)

        encoded = self.tokenizer(
            [question] * len(passages),
            passages,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )

        offset_mapping = encoded.pop("offset_mapping")
        sample_mapping = encoded.pop("overflow_to_sample_mapping")
        sequence_ids = [encoded.sequence_ids(i) for i in range(len(offset_mapping))]
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**model_inputs)

        best = ReaderAnswer(text="", score=float("-inf"), no_answer_score=float("-inf"), passage_index=0)
        for feature_index, passage_index_tensor in enumerate(sample_mapping):
            passage_index = int(passage_index_tensor)
            start_logits = outputs.start_logits[feature_index].detach().cpu()
            end_logits = outputs.end_logits[feature_index].detach().cpu()
            no_answer_score = float(start_logits[0] + end_logits[0])
            if no_answer_score > best.no_answer_score:
                best.no_answer_score = no_answer_score

            context_token_indexes = [
                i for i, sequence_id in enumerate(sequence_ids[feature_index])
                if sequence_id == 1 and offset_mapping[feature_index][i] is not None
            ]
            start_indexes = torch.argsort(start_logits, descending=True)[:n_best_size].tolist()
            end_indexes = torch.argsort(end_logits, descending=True)[:n_best_size].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index not in context_token_indexes or end_index not in context_token_indexes:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                        continue

                    start_char, _ = offset_mapping[feature_index][start_index].tolist()
                    _, end_char = offset_mapping[feature_index][end_index].tolist()
                    score = float(start_logits[start_index] + end_logits[end_index])
                    if score > best.score:
                        best = ReaderAnswer(
                            text=passages[passage_index][start_char:end_char],
                            score=score,
                            no_answer_score=best.no_answer_score,
                            passage_index=passage_index,
                            start_char=start_char,
                            end_char=end_char,
                        )

        return best


def _resolve_checkpoint(checkpoint: str | Path | None, model_kind: str) -> str:
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if _has_model_weights(checkpoint_path):
            return str(checkpoint_path)
        if checkpoint_path.exists():
            print(f"Checkpoint at {checkpoint_path} has no local weights; falling back to {DEFAULT_TOKENIZER_NAME}.")
        else:
            return str(checkpoint)

    default_checkpoint = DEFAULT_DRQA_CHECKPOINT if model_kind == "bert_drqa_attention" else DEFAULT_BASELINE_CHECKPOINT
    if _has_model_weights(default_checkpoint):
        return str(default_checkpoint)

    return DEFAULT_TOKENIZER_NAME


def _has_model_weights(checkpoint: Path) -> bool:
    return (checkpoint / "model.safetensors").exists() or (checkpoint / "pytorch_model.bin").exists()
