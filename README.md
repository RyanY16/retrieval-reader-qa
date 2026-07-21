# Retrieval Reader QA

This repository contains a small question-answering pipeline built around:

- passage retrieval with SentenceTransformers embeddings and a FAISS index
- BERT reader experiments on SQuAD v2
- saved experiment metadata and result artifacts

## Project Structure

```text
.
├── data/
│   └── processed/              # Prebuilt retrieval artifacts
├── notebooks/                  # Reader smoke test and experiment notebooks
├── outputs/                    # Trainer outputs and checkpoint metadata
├── reports/
│   └── figures/                # Saved plots and result images
└── src/                        # Reusable Python code
```

## File Map

- `src/document_retriever.py` loads the local FAISS passage index and passage metadata, embeds questions with `sentence-transformers/all-MiniLM-L6-v2`, and returns the top-k matching passages.
- `data/processed/passage_index.faiss` is the FAISS nearest-neighbor index for passage search.
- `data/processed/passage_embeddings.npy` stores the passage embedding matrix used to build or inspect the index.
- `data/processed/passages_metadata.pkl` stores passage texts and titles aligned with FAISS index ids.
- `notebooks/reader_smoke_test.ipynb` runs a small SQuAD v2 reader smoke test to verify preprocessing, training, no-answer handling, and evaluation.
- `notebooks/reader_1pct_experiment.ipynb` runs the first larger reader experiment. Its current config uses `SUBSET_FRACTION = 0.01`.
- `outputs/reader_1pct_outputs/` contains Hugging Face Trainer metadata for BERT baseline and DrQA-attention reader runs. Large weight and optimizer files are intentionally ignored by git.
- `reports/figures/reader_results.png` is the saved experiment result figure.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Retrieve passages:

```bash
python3 src/document_retriever.py "What is Atlantic City known for?"
```

Run a small retrieval evaluation:

```bash
python3 src/document_retriever.py --evaluate --k 5 --sample-size 100
```

## Large Local Files

The trained checkpoint weights and optimizer states are kept locally under `outputs/reader_1pct_outputs/`, but are excluded from GitHub because they are larger than GitHub's regular file limits.
