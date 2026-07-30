# Retrieval Reader QA

This repository contains an extractive question-answering pipeline with:

- dense passage retrieval using `sentence-transformers/all-MiniLM-L6-v2` and FAISS
- BERT reader experiments on SQuAD 1.1 and SQuAD v2
- a baseline BERT reader and a BERT reader with an added question-passage attention layer
- saved evaluation outputs and report figures

The project evaluates two settings:

1. **Oracle passage setting**: the reader receives the correct SQuAD passage directly.
2. **Full-corpus setting**: the system retrieves passages from a Wikipedia article index, then the reader extracts an answer from the retrieved passages.

## Project Structure

```text
.
├── data/
│   └── processed/                  # Final 481-article full-corpus retrieval index
├── outputs/                        # Result JSON files and lightweight checkpoint metadata
├── reports/
│   └── figures/                    # Final result plots used in the report
├── requirements.txt
└── src/
    ├── build_squad_index.py        # Build Wikipedia/SQuAD FAISS indexes
    ├── document_retriever.py       # Retrieve top-k passages from a FAISS index
    ├── evaluate_pipeline.py        # Evaluate retrieval + reader end to end
    ├── qa_pipeline.py              # Run retrieval followed by answer extraction
    ├── reader.py                   # Reader model and inference helpers
    └── train_reader.py             # Train/evaluate BERT reader variants
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The scripts download datasets and models from Hugging Face on first run.

## Quick Start

Retrieve passages from the default local FAISS index:

```bash
python3 src/document_retriever.py "What is Atlantic City known for?" --k 5
```

Run retrieval followed by the default BERT reader:

```bash
python3 src/qa_pipeline.py "What is Atlantic City known for?" --top-k 5
```

Run the attention-augmented reader from a saved checkpoint:

```bash
python3 src/qa_pipeline.py \
  "What is Atlantic City known for?" \
  --top-k 5 \
  --model-kind bert_drqa_attention \
  --reader-checkpoint outputs/reader_squad_v2_1pct_saved_outputs/bert_drqa_attention/final
```

The large trained model weight files are not committed to GitHub. If a checkpoint path does not contain local weights, the script will need to load the base model instead or you will need to retrain the reader locally.

## Build a Retrieval Index

Build a full Wikipedia-article index from SQuAD-referenced pages:

```bash
python3 src/build_squad_index.py \
  --dataset-name rajpurkar/squad_v2 \
  --split validation \
  --max-documents 500 \
  --output-dir data/processed
```

By default, articles are split into 200-word chunks with a 20-word overlap. You can change this:

```bash
python3 src/build_squad_index.py \
  --dataset-name rajpurkar/squad_v2 \
  --split validation \
  --max-documents 500 \
  --chunk-words 150 \
  --overlap-words 30 \
  --output-dir data/processed
```

Build a small smoke-test index:

```bash
python3 src/build_squad_index.py --max-documents 5 --output-dir data/wiki_smoke
```

Reproduce the older context-only setup instead of downloading full Wikipedia articles:

```bash
python3 src/build_squad_index.py --use-squad-contexts --output-dir data/squad_contexts
```

## Train Reader Models

Train both reader variants on SQuAD v2:

```bash
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 src/train_reader.py \
  --dataset squad_v2 \
  --subset-fraction 0.05 \
  --model-kind all
```

Train both reader variants on SQuAD 1.1:

```bash
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 src/train_reader.py \
  --dataset squad_v1 \
  --subset-fraction 0.05 \
  --model-kind all
```

Use `--skip-save-model` if you only want metrics and do not want to save large checkpoint weights:

```bash
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 src/train_reader.py \
  --dataset squad_v2 \
  --subset-fraction 0.01 \
  --model-kind all \
  --skip-save-model
```

Useful training options:

- `--subset-fraction`: training/validation subset size, such as `0.01`, `0.05`, or `0.10`
- `--model-kind`: `bert_baseline`, `bert_drqa_attention`, or `all`
- `--epochs`: number of training epochs
- `--train-batch-size` and `--eval-batch-size`: batch sizes
- `--output-dir`: custom output directory

Training writes metrics to `outputs/.../results.json`. If model saving is enabled, final checkpoints are written under `outputs/.../<model_kind>/final/`.

## Evaluate Retrieval + Reader

Evaluate the full-corpus pipeline with the baseline reader:

```bash
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 src/evaluate_pipeline.py \
  --data-dir data/processed \
  --dataset-name rajpurkar/squad_v2 \
  --split validation \
  --sample-size 100 \
  --top-k 5 \
  --model-kind bert_baseline \
  --reader-checkpoint outputs/reader_squad_v2_1pct_saved_outputs/bert_baseline/final
```

Evaluate the attention-augmented reader:

```bash
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 src/evaluate_pipeline.py \
  --data-dir data/processed \
  --dataset-name rajpurkar/squad_v2 \
  --split validation \
  --sample-size 100 \
  --top-k 5 \
  --model-kind bert_drqa_attention \
  --reader-checkpoint outputs/reader_squad_v2_1pct_saved_outputs/bert_drqa_attention/final
```

Evaluate SQuAD 1.1 with the same full-corpus index:

```bash
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 src/evaluate_pipeline.py \
  --data-dir data/processed \
  --dataset-name rajpurkar/squad \
  --split validation \
  --sample-size 100 \
  --top-k 5 \
  --model-kind bert_baseline \
  --reader-checkpoint outputs/reader_squad_v1_5pct_saved_outputs/bert_baseline/final
```

The evaluation prints retrieval recall, end-to-end exact match, end-to-end F1, and no-answer accuracy when the dataset contains unanswerable questions.

## Final Report Artifacts

Main result figures:

- `reports/figures/image1_overall_metrics.pdf`
- `reports/figures/image2_squad2_split_metrics.pdf`

Final full-corpus evaluation JSON:

- `outputs/full_corpus_481_eval_top5.json`

The final full-corpus index used in the report contains 481 Wikipedia articles and 23,286 passage chunks. The target was 500 article titles, but some titles were duplicates, unresolved, or empty after Wikipedia API lookup.

## Notes on Large Files

GitHub does not accept the trained `model.safetensors` files because they are hundreds of megabytes each. Those files are intentionally ignored by `.gitignore`. The repository keeps lightweight configs, result JSON files, retrieval indexes, and report figures.
