"""End-to-end retrieval plus reader question answering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_retriever import DEFAULT_DATA_DIR, DocumentRetriever
from reader import ExtractiveReader, ReaderAnswer


@dataclass
class QAResult:
    question: str
    answer: ReaderAnswer
    retrieved_passages: list[dict[str, Any]]

    @property
    def answer_passage(self) -> dict[str, Any] | None:
        if self.answer.passage_index < 0:
            return None
        return self.retrieved_passages[self.answer.passage_index]


class RetrievalReaderQA:
    """Searches for relevant passages, then extracts an answer from them."""

    def __init__(self, retriever: DocumentRetriever, reader: ExtractiveReader):
        self.retriever = retriever
        self.reader = reader

    @classmethod
    def from_artifacts(
        cls,
        data_dir: Path = DEFAULT_DATA_DIR,
        reader_checkpoint: str | Path | None = None,
        model_kind: str = "bert_baseline",
    ) -> "RetrievalReaderQA":
        retriever = DocumentRetriever.from_artifacts(data_dir=data_dir)
        reader = ExtractiveReader.from_checkpoint(checkpoint=reader_checkpoint, model_kind=model_kind)
        return cls(retriever=retriever, reader=reader)

    def answer(self, question: str, top_k: int = 5, no_answer_threshold: float = 0.0) -> QAResult:
        retrieved = self.retriever.retrieve(question, k=top_k)
        passages = [item["passage"] for item in retrieved]
        answer = self.reader.answer(question, passages)

        if answer.no_answer_score > answer.score + no_answer_threshold:
            answer = ReaderAnswer(
                text="",
                score=answer.score,
                no_answer_score=answer.no_answer_score,
                passage_index=answer.passage_index,
                start_char=answer.start_char,
                end_char=answer.end_char,
            )

        return QAResult(question=question, answer=answer, retrieved_passages=retrieved)


def evaluate_end_to_end(qa: RetrievalReaderQA, sample_size: int = 100, top_k: int = 5) -> None:
    from datasets import load_dataset

    squad = load_dataset("rajpurkar/squad", split=f"validation[:{sample_size}]")
    correct_contains = 0
    answerable = 0

    for example in squad:
        gold_answers = example["answers"]["text"]
        if not gold_answers:
            continue

        answerable += 1
        result = qa.answer(example["question"], top_k=top_k)
        prediction = result.answer.text.lower()
        if prediction and any(prediction in gold.lower() or gold.lower() in prediction for gold in gold_answers):
            correct_contains += 1

    accuracy = correct_contains / answerable if answerable else 0
    print(f"End-to-end contains-match accuracy@{top_k}: {accuracy:.4f} ({correct_contains}/{answerable})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval followed by extractive QA.")
    parser.add_argument("question", nargs="?", help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved passages to read.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reader-checkpoint", type=Path, default=None)
    parser.add_argument("--model-kind", choices=["bert_baseline", "bert_drqa_attention"], default="bert_baseline")
    parser.add_argument("--no-answer-threshold", type=float, default=0.0)
    parser.add_argument("--evaluate", action="store_true", help="Run a small SQuAD end-to-end check.")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    qa = RetrievalReaderQA.from_artifacts(
        data_dir=args.data_dir,
        reader_checkpoint=args.reader_checkpoint,
        model_kind=args.model_kind,
    )

    if args.evaluate:
        evaluate_end_to_end(qa, sample_size=args.sample_size, top_k=args.top_k)
        return

    question = args.question or "What is Atlantic City known for?"
    result = qa.answer(question, top_k=args.top_k, no_answer_threshold=args.no_answer_threshold)
    passage = result.answer_passage

    print(f"Question: {result.question}")
    print(f"Answer: {result.answer.text or '[no answer]'}")
    print(f"Reader score: {result.answer.score:.3f}")
    print(f"No-answer score: {result.answer.no_answer_score:.3f}")
    if passage:
        print(f"Source: {passage['title']} (retrieval score: {passage['score']:.3f})")
        print(f"Passage: {passage['passage'][:500].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
