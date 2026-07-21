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
- `src/reader.py` contains the BERT extractive reader code, including the baseline reader loader and the DrQA-style attention reader class used in the notebooks.
- `src/qa_pipeline.py` connects retrieval and reading: it searches for relevant passages, sends those passages to the reader, and returns the best extracted answer.
- `data/processed/passage_index.faiss` is the FAISS nearest-neighbor index for passage search.
- `data/processed/passage_embeddings.npy` stores the passage embedding matrix used to build or inspect the index.
- `data/processed/passages_metadata.pkl` stores passage texts and titles aligned with FAISS index ids.
- `notebooks/reader_smoke_test.ipynb` runs a small SQuAD v2 reader smoke test to verify preprocessing, training, no-answer handling, and evaluation.
- `notebooks/reader_1pct_experiment.ipynb` runs the first larger reader experiment. Its current config uses `SUBSET_FRACTION = 0.01`.
- `outputs/reader_1pct_outputs/` contains Hugging Face Trainer metadata for BERT baseline and DrQA-attention reader runs. Large weight and optimizer files are intentionally ignored by git.
- `reports/figures/reader_results.png` is the saved experiment result figure.

## Experiment Methodology

The project separates question answering into two stages:

1. Retrieval: passages are embedded with SentenceTransformers and indexed with FAISS. At query time, `src/document_retriever.py` embeds the question and searches `data/processed/passage_index.faiss`, using `data/processed/passages_metadata.pkl` to map retrieved ids back to passage titles and text.
2. Reading: the reader experiments use SQuAD v2 examples where the model receives a question and passage/context, then predicts an answer span or no-answer. `notebooks/reader_smoke_test.ipynb` verifies the full training/evaluation path on a tiny subset before running the larger setup in `notebooks/reader_1pct_experiment.ipynb`.

The 1% experiment compares a BERT baseline reader with a BERT reader variant that adds DrQA-style attention. Training metadata and checkpoint configuration files are saved under `outputs/reader_1pct_outputs/`, while the resulting plot is stored at `reports/figures/reader_results.png`.

The end-to-end implementation is in `src/qa_pipeline.py`: it takes a question, retrieves the most relevant passages from the FAISS index, runs the BERT reader over those passages, and returns either the best answer span or no answer.

For the fair project evaluation, the retrieval index should be built from the same SQuAD split being tested. `src/build_squad_index.py` creates a 500-context SQuAD v2 validation index, and `src/evaluate_pipeline.py` tests retrieval recall plus end-to-end answer quality on questions from that same split.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Retrieve passages:

```bash
python3 src/document_retriever.py "What is Atlantic City known for?"
```

Run the full retrieval + reader pipeline:

```bash
python3 src/qa_pipeline.py "What is Atlantic City known for?" --top-k 5
```

Run a small retrieval evaluation:

```bash
python3 src/document_retriever.py --evaluate --k 5 --sample-size 100
```

Run a small end-to-end evaluation:

```bash
python3 src/qa_pipeline.py --evaluate --top-k 5 --sample-size 100
```

Build the fair SQuAD evaluation index:

```bash
python3 src/build_squad_index.py --max-documents 500
```

Evaluate retrieval and answer extraction on that matching SQuAD index:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/evaluate_pipeline.py --sample-size 100 --top-k 5
```

## Current Results

These results use `rajpurkar/squad_v2`, the first 500 unique validation contexts as the retrieval collection, and the first 100 validation questions for evaluation.

| Reader | Retrieval Recall@5 | End-to-End EM | End-to-End F1 | No-Answer Accuracy |
| --- | ---: | ---: | ---: | ---: |
| BERT baseline | 0.8667 | 0.5100 | 0.5100 | 0.8909 |
| BERT + DrQA attention | 0.8667 | 0.5000 | 0.5050 | 0.7818 |

The baseline is slightly stronger in this small run, mainly because it predicts no-answer cases more accurately.

## Large Local Files

The trained checkpoint weights and optimizer states are kept locally under `outputs/reader_1pct_outputs/`, but are excluded from GitHub because they are larger than GitHub's regular file limits.
