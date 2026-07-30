"""Build a FAISS passage index from Wikipedia articles referenced by SQuAD."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests

from document_retriever import DEFAULT_MODEL_NAME, PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "squad_validation_500"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "NLP-QA-project/1.0 (https://example.local; educational use)"


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


def collect_unique_titles(dataset, max_documents: int | None = None) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()

    for example in dataset:
        title = (example.get("title") or "").strip()
        if not title or title in seen:
            continue

        seen.add(title)
        titles.append(title)

        if max_documents is not None and len(titles) >= max_documents:
            break

    return titles


def fetch_wikipedia_extract(title: str, sleep_seconds: float = 0.05) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    url = f"{WIKIPEDIA_API_URL}?{urlencode(params)}"

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Skipping {title!r}: Wikipedia request failed ({exc})")
        return None
    finally:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            print(f"Skipping {title!r}: page not found")
            return None
        extract = (page.get("extract") or "").strip()
        if extract:
            return extract

    print(f"Skipping {title!r}: empty page extract")
    return None


def chunk_article(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= overlap_words:
        raise ValueError("--chunk-words must be greater than --overlap-words")

    chunks: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break

    return chunks


def collect_wikipedia_passages(
    dataset,
    max_documents: int | None,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
    sleep_seconds: float,
) -> tuple[list[str], list[str], list[dict[str, int | str]]]:
    passages: list[str] = []
    passage_titles: list[str] = []
    passage_metadata: list[dict[str, int | str]] = []

    titles = collect_unique_titles(dataset, max_documents=max_documents)
    if not titles:
        raise ValueError("No SQuAD article titles were collected.")

    for title_index, title in enumerate(titles, start=1):
        extract = fetch_wikipedia_extract(title, sleep_seconds=sleep_seconds)
        if not extract:
            continue

        chunks = [
            chunk
            for chunk in chunk_article(extract, chunk_words=chunk_words, overlap_words=overlap_words)
            if len(chunk.split()) >= min_words
        ]
        for chunk_index, chunk in enumerate(chunks):
            passages.append(chunk)
            passage_titles.append(title)
            passage_metadata.append(
                {
                    "title": title,
                    "chunk_index": chunk_index,
                    "source_article_number": title_index,
                    "source": "wikipedia_api",
                }
            )

        print(f"[{title_index}/{len(titles)}] {title}: {len(chunks)} chunks")

    return passages, passage_titles, passage_metadata


def build_index(
    dataset_name: str,
    split: str,
    output_dir: Path,
    max_documents: int | None,
    model_name: str,
    batch_size: int,
    use_squad_contexts: bool,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
    sleep_seconds: float,
) -> None:
    import faiss
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split)
    if use_squad_contexts:
        passages, passage_titles = collect_unique_contexts(dataset, max_documents=max_documents)
        passage_metadata = [
            {"title": title, "chunk_index": 0, "source": "squad_context"}
            for title in passage_titles
        ]
        corpus_source = "squad_contexts"
    else:
        passages, passage_titles, passage_metadata = collect_wikipedia_passages(
            dataset,
            max_documents=max_documents,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            min_words=min_words,
            sleep_seconds=sleep_seconds,
        )
        corpus_source = "wikipedia_articles"

    if not passages:
        raise ValueError("No passages were collected.")

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
                "passage_metadata": passage_metadata,
                "dataset_name": dataset_name,
                "split": split,
                "model_name": model_name,
                "corpus_source": corpus_source,
                "max_documents": max_documents,
                "chunk_words": chunk_words if not use_squad_contexts else None,
                "overlap_words": overlap_words if not use_squad_contexts else None,
            },
            f,
        )

    print(f"Built index with {len(passages)} passages from {corpus_source} at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from SQuAD-referenced Wikipedia articles.")
    parser.add_argument("--dataset-name", default="rajpurkar/squad_v2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-documents", type=int, default=500, help="Maximum unique SQuAD article titles to index.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Delay between Wikipedia API requests.")
    parser.add_argument("--use-squad-contexts", action="store_true", help="Index SQuAD contexts instead of full Wikipedia articles.")
    args = parser.parse_args()

    build_index(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        max_documents=args.max_documents,
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_squad_contexts=args.use_squad_contexts,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
