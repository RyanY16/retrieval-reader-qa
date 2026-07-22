# All Reader Results

These are the reader-only SQuAD experiment results for BERT baseline and BERT + DrQA-style attention.

## Overall EM/F1

| Dataset | Training Size | Reader | Train Examples | Validation Examples | EM | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| SQuAD 1.1 | 1% | BERT baseline | 875 | 105 | 40.00 | 49.70 |
| SQuAD 1.1 | 1% | BERT + DrQA attention | 875 | 105 | 32.38 | 48.15 |
| SQuAD 1.1 | 5% | BERT baseline | 4379 | 528 | 60.80 | 72.98 |
| SQuAD 1.1 | 5% | BERT + DrQA attention | 4379 | 528 | 64.02 | 75.40 |
| SQuAD v2 | 1% | BERT baseline | 1303 | 118 | 50.85 | 51.00 |
| SQuAD v2 | 1% | BERT + DrQA attention | 1303 | 118 | 46.61 | 48.06 |
| SQuAD v2 | 5% | BERT baseline | 6515 | 593 | 52.95 | 55.16 |
| SQuAD v2 | 5% | BERT + DrQA attention | 6515 | 593 | 52.95 | 56.03 |

## SQuAD v2 Answerable vs Unanswerable Metrics

| Dataset | Training Size | Reader | Train Examples | Validation Examples | Answerability Accuracy | Answerability Precision | Answerability Recall | Answerability F1 | Answerable Examples | Answerable EM | Answerable F1 | Unanswerable Examples | Unanswerable EM | Unanswerable F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SQuAD v2 | 1% | BERT baseline | 1303 | 118 | 54.24 | 62.50 | 17.24 | 27.03 | 58 | 10.34 | 10.66 | 60 | 90.00 | 90.00 |
| SQuAD v2 | 1% | BERT + DrQA attention | 1303 | 118 | 54.24 | 56.67 | 29.31 | 38.64 | 58 | 13.79 | 16.74 | 60 | 78.33 | 78.33 |
| SQuAD v2 | 5% | BERT baseline | 6515 | 593 | 61.38 | 59.93 | 58.45 | 59.18 | 284 | 40.85 | 45.45 | 309 | 64.08 | 64.08 |
| SQuAD v2 | 5% | BERT + DrQA attention | 6515 | 593 | 62.39 | 59.81 | 65.49 | 62.52 | 284 | 45.77 | 52.20 | 309 | 59.55 | 59.55 |

Notes:

- SQuAD 1.1 has only answerable questions, so the second table is only for SQuAD v2.
- The strongest SQuAD 1.1 result is the 5% BERT + DrQA attention reader.
- In SQuAD v2, the 5% BERT + DrQA attention reader has the best overall F1 and answerability F1, while overall EM is tied with the 5% baseline.
