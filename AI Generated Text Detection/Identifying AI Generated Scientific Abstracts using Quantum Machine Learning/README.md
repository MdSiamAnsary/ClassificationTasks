# Quantum Machine Learning vs Classical and Transformer-Based Baselines

## Description of Proposed Quantum Machine Learning (QML) Model

The proposed approach utilizes a hybrid classical-quantum pipeline for binary text classification, aiming to distinguish between human-written and AI-generated text. The process begins with generating **sentence embeddings** using the `all-MiniLM-L6-v2` model, which are then reduced to a 4-dimensional representation via **PCA** and normalized to the range [0, π]. These features are fed into a **Parameterized Quantum Circuit (PQC)** implemented with PennyLane, consisting of rotational encoding, nearest-neighbor CNOT entanglement, and a variational layer of parameterized rotations. The PQC outputs a quantum expectation value that is passed through a **sigmoid function** to produce binary predictions. The model is trained using **stochastic gradient descent (SGD)** over 20 epochs with a mean squared error loss function.

## Baseline Models

Six baseline models were implemented for comparative evaluation:

1. **Logistic Regression** – A classical linear model for binary classification.
2. **SVM (RBF Kernel)** – A non-linear support vector machine with radial basis function kernel.
3. **Random Forest** – An ensemble of decision trees with majority voting.
4. **XGBoost** – Gradient boosting framework for optimized tree-based learning.
5. **BERT-base Fine-Tuning** – Transformer-based language model fine-tuned on the dataset for sequence classification.
6. **RoBERTa-base Fine-Tuning** – Robustly optimized BERT variant, also fine-tuned for sequence classification.

## Performance Comparison

| Model                     | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|----------------------------|---------|-----------|--------|----------|---------|
| Logistic Regression        | 0.88    | 0.89      | 0.88   | 0.88     | 0.981   |
| SVM (RBF)                  | 0.88    | 0.89      | 0.88   | 0.88     | 0.974   |
| Random Forest              | 0.90    | 0.91      | 0.90   | 0.90     | 0.983   |
| XGBoost                    | 0.93    | 0.93      | 0.93   | 0.93     | 0.983   |
| BERT-base Fine-Tuning      | 0.976   | 0.955     | 1.0    | 0.977    | -       |
| RoBERTa-base Fine-Tuning   | 0.976   | 1.0       | 0.95   | 0.974    | -       |
| Quantum ML (PQC)           | 0.854   | 0.76      | 1.0    | 0.864    | -       |

**Notes:**
- ROC-AUC values are reported for classical models; transformer models and QML were evaluated primarily with accuracy, precision, recall, and F1-score.
- The Quantum ML approach achieves competitive performance relative to classical models and demonstrates strong recall, capturing all AI-generated texts, though precision is lower due to some false positives.

