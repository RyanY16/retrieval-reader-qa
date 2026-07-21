"""Build a FAISS passage index from SQuAD contexts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from document_retriever import DEFAULT_MODEL_NAME, PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "squad_validation_500"


def collect_unique_contexts(dataset, max_documents: int | None = None) -> tuple[list[str], list[str]]:
    passages: list[str] = []
    passage_titles: list[str] = []
    seen: set[str] = set()

    for example in dataset:
        context = example["context"].strip()
        if context in seen:
            continue

        seen.add(context)
        passages.append(context)
        passage_titles.append(example.get("title") or f"squad_context_{len(passages)}")

        if max_documents is not None and len(passages) >= max_documents:
            break

    return passages, passage_titles


def build_index(
    dataset_name: str,
    split: str,
    output_dir: Path,
    max_documents: int | None,
    model_name: str,
    batch_size: int,
) -> None:
    import faiss
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split)
    passages, passage_titles = collect_unique_contexts(dataset, max_documents=max_documents)
    if not passages:
        raise ValueError("No SQuAD contexts were collected.")

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "passage_index.faiss"))
    np.save(output_dir / "passage_embeddings.npy", embeddings)
    with (output_dir / "passages_metadata.pkl").open("wb") as f:
        pickle.dump(
            {
                "passages": passages,
                "passage_titles": passage_titles,
                "dataset_name": dataset_name,
                "split": split,
                "model_name": model_name,
            },
            f,
        )

    print(f"Built index with {len(passages)} SQuAD contexts at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SQuAD context FAISS index.")
    parser.add_argument("--dataset-name", default="rajpurkar/squad_v2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_index(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        max_documents=args.max_documents,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
