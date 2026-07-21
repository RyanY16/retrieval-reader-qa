"""FAISS-backed passage retrieval for SQuAD-style question answering."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class DocumentRetriever:
    """Encodes questions and searches a prebuilt FAISS passage index."""

    encoder: Any
    index: Any
    passages: list[str]
    passage_titles: list[str]
    normalize_embeddings: bool = True

    @classmethod
    def from_artifacts(
        cls,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> "DocumentRetriever":
        index_path = data_dir / "passage_index.faiss"
        metadata_path = data_dir / "passages_metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing passage metadata: {metadata_path}")

        import faiss
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name)
        index = faiss.read_index(str(index_path))
        with metadata_path.open("rb") as f:
            metadata = pickle.load(f)

        return cls(
            encoder=encoder,
            index=index,
            passages=metadata["passages"],
            passage_titles=metadata["passage_titles"],
        )

    def retrieve(self, question: str, k: int = 5) -> list[dict[str, Any]]:
        q_vector = self.encoder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        scores, indices = self.index.search(q_vector, k)

        return [
            {
                "title": self.passage_titles[idx],
                "passage": self.passages[idx],
                "score": float(score),
            }
            for idx, score in zip(indices[0], scores[0])
        ]


def evaluate_retrieval(
    dataset,
    retriever: DocumentRetriever,
    k: int = 5,
    sample_size: int | None = None,
):
    """Measure whether any retrieved passage contains a gold answer string."""
    questions = dataset["question"]
    answers = dataset["answers"]

    if sample_size:
        questions = questions[:sample_size]
        answers = answers[:sample_size]

    correct = 0
    total = 0
    results_log = []

    for question, answer_dict in zip(questions, answers):
        gold_answers = answer_dict["text"]
        if not gold_answers:
            continue

        results = retriever.retrieve(question, k=k)
        found = any(
            gold.lower() in result["passage"].lower()
            for gold in gold_answers
            for result in results
        )

        correct += int(found)
        total += 1
        results_log.append(
            {
                "question": question,
                "gold_answers": gold_answers,
                "found": found,
                "retrieved_titles": [result["title"] for result in results],
            }
        )

    accuracy = correct / total if total else 0
    print(f"Retrieval Accuracy@{k}: {accuracy:.4f} ({correct}/{total})")
    return accuracy, results_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local passage FAISS index.")
    parser.add_argument("question", nargs="?", help="Question to retrieve passages for.")
    parser.add_argument("--k", type=int, default=5, help="Number of passages to return.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--evaluate", action="store_true", help="Run a small SQuAD validation evaluation.")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    retriever = DocumentRetriever.from_artifacts(args.data_dir)
    print(f"Loaded index with {retriever.index.ntotal} vectors and {len(retriever.passages)} passages")

    if args.evaluate:
        from datasets import load_dataset

        squad = load_dataset("rajpurkar/squad")
        evaluate_retrieval(squad["validation"], retriever, k=args.k, sample_size=args.sample_size)
        return

    question = args.question or "What is Atlantic City known for?"
    for result in retriever.retrieve(question, k=args.k):
        snippet = result["passage"][:150].replace("\n", " ")
        print(f"[{result['score']:.3f}] {result['title']}: {snippet}...")


if __name__ == "__main__":
    main()
