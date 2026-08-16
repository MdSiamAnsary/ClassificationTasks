# Model Performance Summary — AI-Generated Bengali Text Detection

**Source:** [`MdSiamAnsary/ClassificationTasks`](https://github.com/MdSiamAnsary/ClassificationTasks) — *AI Generated Text Detection / A Novel Dataset for Identifying AI Generated Bengali Texts / ipynb files*

**Task:** Binary classification of Bengali text as **human-written** vs. **AI-generated**.

---

## Results at a Glance

| # | Notebook | Model | Features / Backbone | Accuracy | Precision | Recall | F1 Score |
|---|----------|-------|----------------------|:--------:|:---------:|:------:|:--------:|
| 1 | `001_MultinomialNB.ipynb` | Multinomial Naive Bayes | TF-IDF | 0.7416 | 0.8693 | 0.5208 | 0.4648 |
| 2 | `002_RandomForest.ipynb` | Random Forest (100 trees, max depth 10) | TF-IDF | 0.7528 | 0.8736 | 0.5417 | 0.5046 |
| 3 | `003_SVC.ipynb` | SVM (polynomial kernel, degree 6) | TF-IDF | 0.7640 | 0.3820 | 0.5000 | 0.4331 |
| 4 | `004_KNN.ipynb` | K-Nearest Neighbors (k=15, Euclidean) | TF-IDF | 0.8090 | 0.7372 | 0.6940 | 0.7100 |
| 5 | `005_CNN_LSTM.ipynb` | CNN–LSTM ensemble head | Bangla-BERT embeddings (`sagorsarker/bangla-bert-base`) | 0.7640 | 0.3820 | 0.5000 | 0.4331 |
| 6 | `006_BERT.ipynb` | Fine-tuned BERT classifier | `sagorsarker/bangla-bert-base` | **0.9551** | **0.9722** | **0.9048** | **0.9331** |

*(Precision/Recall/F1 are macro-averaged where applicable, as computed in each notebook.)*

---

## Notes on Each Model

- **Multinomial Naive Bayes** — Baseline TF-IDF + NB classifier (`alpha=1.0`). Reasonable accuracy but low recall/F1, indicating it struggles to catch the minority class consistently.
- **Random Forest** — TF-IDF features with a bagged ensemble (100 trees, `max_depth=10`, `max_features='sqrt'`). Slightly better than Naive Bayes across the board.
- **SVM (Polynomial, degree 6)** — Despite a competitive accuracy (0.7640), precision/recall/F1 collapse toward the values expected of a majority-class predictor, suggesting the model is not discriminating well between classes.
- **KNN (k=15)** — The strongest of the classical/TF-IDF models by a clear margin, with the most balanced precision/recall trade-off among the non-transformer approaches.
- **CNN–LSTM over Bangla-BERT embeddings** — Training logs show `val_accuracy` frozen at 0.7619 across all 5 epochs with no real learning progress; final test metrics (0.7640 / 0.3820 / 0.5000 / 0.4331) match a degenerate, majority-class-only prediction pattern.
- **Fine-tuned Bangla-BERT** (`AutoModelForSequenceClassification`, full fine-tuning, `MAX_LEN=512`) — By far the best performer, with accuracy, precision, recall, and F1 all above 0.90.

---

## Best-Performing Model

**Fine-tuned Bangla-BERT (`006_BERT.ipynb`)** is the clear winner:
- Accuracy: **0.9551**
- Precision: **0.9722**
- Recall: **0.9048**
- F1 Score: **0.9331**

This suggests that full fine-tuning of a Bangla-pretrained transformer substantially outperforms both classical TF-IDF-based classifiers and a shallow CNN–LSTM head built on frozen/embedded BERT representations for this AI-generated Bengali text detection task.
