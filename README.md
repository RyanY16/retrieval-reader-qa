# Retrieval Reader QA

This repository contains a small question-answering pipeline built around:

- passage retrieval with SentenceTransformers embeddings and a FAISS index
- BERT reader experiments on SQuAD 1.1 and SQuAD v2
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

1. Retrieval: `src/build_squad_index.py` collects Wikipedia article titles from SQuAD, downloads the corresponding full article text through the Wikipedia API, splits each article into overlapping 200-word passages with a 20-word overlap, embeds those passages with SentenceTransformers, and indexes them with FAISS. At query time, `src/document_retriever.py` embeds the question and searches `data/processed/passage_index.faiss`, using `data/processed/passages_metadata.pkl` to map retrieved ids back to passage titles and text.
2. Reading: the reader experiments use SQuAD v2 examples where the model receives a question and passage/context, then predicts an answer span or no-answer. `notebooks/reader_smoke_test.ipynb` verifies the full training/evaluation path on a tiny subset before running the larger setup in `notebooks/reader_1pct_experiment.ipynb`.

The 1% experiment compares a BERT baseline reader with a BERT reader variant that adds DrQA-style attention. Training metadata and checkpoint configuration files are saved under `outputs/reader_1pct_outputs/`, while the resulting plot is stored at `reports/figures/reader_results.png`.

The end-to-end implementation is in `src/qa_pipeline.py`: it takes a question, retrieves the most relevant passages from the FAISS index, runs the BERT reader over those passages, and returns either the best answer span or no answer.

For the fair project evaluation, the retrieval index should be built from Wikipedia articles referenced by the same SQuAD split being tested. `src/build_squad_index.py` creates a chunked article index from up to 500 SQuAD-referenced Wikipedia pages, and `src/evaluate_pipeline.py` tests retrieval recall plus end-to-end answer quality on questions from that same split.

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

Train reader models on a larger SQuAD v2 subset:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/train_reader.py --subset-fraction 0.05 --model-kind all
```

The 5% run writes to `outputs/reader_5pct_outputs/`. On a local Mac, these are long training jobs: the 5% baseline run took about 58 minutes, and the 5% BERT + DrQA attention run took about 80 minutes.

Train reader models on SQuAD 1.1 without saving large model weights:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/train_reader.py --dataset squad_v1 --subset-fraction 0.01 --model-kind all --skip-save-model
```

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/train_reader.py --dataset squad_v1 --subset-fraction 0.05 --model-kind all --skip-save-model
```

Build the fair SQuAD evaluation index from Wikipedia articles:

```bash
python3 src/build_squad_index.py --max-documents 500
```

For a faster debugging run, build a tiny article index in a separate directory:

```bash
python3 src/build_squad_index.py --max-documents 5 --output-dir data/wiki_smoke
```

To reproduce the older SQuAD-context-only setup, add `--use-squad-contexts`.

Evaluate retrieval and answer extraction on that matching SQuAD index:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/evaluate_pipeline.py --sample-size 100 --top-k 5
```

## Current Results

All reader-only SQuAD results are saved in `reports/all_reader_results.md`.

The end-to-end results below were produced before the retrieval corpus was changed to full Wikipedia articles. They use `rajpurkar/squad_v2`, the first 500 unique validation contexts as the retrieval collection, and the first 100 validation questions for evaluation. Rebuild the article index and rerun `src/evaluate_pipeline.py` before reporting final end-to-end retrieval numbers.

| Reader | Retrieval Recall@5 | End-to-End EM | End-to-End F1 | No-Answer Accuracy |
| --- | ---: | ---: | ---: | ---: |
| BERT baseline | 0.8667 | 0.5100 | 0.5100 | 0.8909 |
| BERT + DrQA attention | 0.8667 | 0.5000 | 0.5050 | 0.7818 |

The baseline is slightly stronger in this small run, mainly because it predicts no-answer cases more accurately.

### 5% Reader Training Results

These results train directly on a 5% SQuAD v2 training subset and evaluate on the configured SQuAD v2 validation subset.

| Reader | Train Examples | Validation Examples | EM | F1 | Answerability Accuracy | Answerability F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT baseline | 6515 | 593 | 52.95 | 55.16 | 61.38 | 59.18 |
| BERT + DrQA attention | 6515 | 593 | 52.95 | 56.03 | 62.39 | 62.52 |

The attention model is slightly better on F1 and answerability recall/F1 in this 5% run, while exact match is tied.

### SQuAD 1.1 Reader Training Results

SQuAD 1.1 has answerable questions only, so these runs report EM and F1 but no no-answer metrics.

| Training Size | Reader | Train Examples | Validation Examples | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1% | BERT baseline | 875 | 105 | 40.00 | 49.70 |
| 1% | BERT + DrQA attention | 875 | 105 | 32.38 | 48.15 |
| 5% | BERT baseline | 4379 | 528 | 60.80 | 72.98 |
| 5% | BERT + DrQA attention | 4379 | 528 | 64.02 | 75.40 |

The 1% SQuAD 1.1 run favors the baseline on exact match, while the 5% run favors the attention reader on both EM and F1.

## Large Local Files

The trained checkpoint weights are kept locally under `outputs/reader_*_outputs/`, but are excluded from GitHub because they are larger than GitHub's regular file limits. The committed output files are the lightweight configs and result JSON files.
