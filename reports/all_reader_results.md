# All Reader Results

These are the reader-only SQuAD experiment results for BERT baseline and BERT + DrQA-style attention.

| Dataset | Training Size | Reader | Train Examples | Validation Examples | EM | F1 | Answerability Accuracy | Answerability F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SQuAD 1.1 | 1% | BERT baseline | 875 | 105 | 40.00 | 49.70 | N/A | N/A |
| SQuAD 1.1 | 1% | BERT + DrQA attention | 875 | 105 | 32.38 | 48.15 | N/A | N/A |
| SQuAD 1.1 | 5% | BERT baseline | 4379 | 528 | 60.80 | 72.98 | N/A | N/A |
| SQuAD 1.1 | 5% | BERT + DrQA attention | 4379 | 528 | 64.02 | 75.40 | N/A | N/A |
| SQuAD v2 | 1% | BERT baseline | 1303 | 118 | 50.85 | 51.00 | 54.24 | 27.03 |
| SQuAD v2 | 1% | BERT + DrQA attention | 1303 | 118 | 46.61 | 48.06 | 54.24 | 38.64 |
| SQuAD v2 | 5% | BERT baseline | 6515 | 593 | 52.95 | 55.16 | 61.38 | 59.18 |
| SQuAD v2 | 5% | BERT + DrQA attention | 6515 | 593 | 52.95 | 56.03 | 62.39 | 62.52 |

Notes:

- SQuAD 1.1 has only answerable questions, so answerability metrics do not apply.
- SQuAD v2 includes unanswerable questions, so answerability accuracy and F1 are reported.
- The strongest SQuAD 1.1 result is the 5% BERT + DrQA attention reader.
- The strongest SQuAD v2 5% result is also the BERT + DrQA attention reader by F1 and answerability F1, with EM tied against the baseline.
