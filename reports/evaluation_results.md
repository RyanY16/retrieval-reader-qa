# Evaluation Results

Evaluation setup:

- Dataset: `rajpurkar/squad_v2`
- Retrieval collection: first 500 unique validation contexts
- Evaluation sample: first 100 validation questions
- Retrieval: SentenceTransformers + FAISS, top 5 passages
- Readers: BERT baseline and BERT + DrQA-style attention

| Reader | Retrieval Recall@5 | End-to-End EM | End-to-End F1 | No-Answer Accuracy |
| --- | ---: | ---: | ---: | ---: |
| BERT baseline | 0.8667 (39/45) | 0.5100 | 0.5100 | 0.8909 (49/55) |
| BERT + DrQA attention | 0.8667 (39/45) | 0.5000 | 0.5050 | 0.7818 (43/55) |

The retriever finds an answer-containing passage for most answerable questions in the sample. The baseline reader is slightly stronger overall in this run because it handles no-answer examples better.

## 5% Reader Training Results

Setup:

- Dataset: `rajpurkar/squad_v2`
- Training subset: 5% of SQuAD v2 training examples
- Validation examples: configured validation subset from `src/train_reader.py`
- Readers: BERT baseline and BERT + DrQA-style attention

| Reader | Train Examples | Validation Examples | EM | F1 | Answerability Accuracy | Answerability Recall | Answerability F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT baseline | 6515 | 593 | 52.95 | 55.16 | 61.38 | 58.45 | 59.18 |
| BERT + DrQA attention | 6515 | 593 | 52.95 | 56.03 | 62.39 | 65.49 | 62.52 |

The 5% run gives the attention reader a small F1 and answerability-F1 edge, with exact match tied.
