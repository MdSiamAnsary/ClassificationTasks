import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Load dataset
df = pd.read_csv("data.csv")  # text, label
texts = df["text"].astype(str)
labels = df["label"]

# Split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Baseline: TF-IDF + Logistic Regression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
probs = model.predict_proba(X_test_vec)[:,1]

print("TF-IDF + LR")
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))

#Baseline: Stylometric SVM 

from sklearn.svm import SVC
import re
from collections import Counter

def handcrafted(texts):
    feats = []
    for t in texts:
        tokens = re.findall(r"\w+", t.lower())
        freq = Counter(tokens)

        entropy = -sum((c/len(tokens))*np.log(c/len(tokens)+1e-9) for c in freq.values()) if tokens else 0

        feats.append([
            len(tokens),
            len(t),
            np.mean([len(x) for x in tokens]) if tokens else 0,
            len(set(tokens))/(len(tokens)+1e-9),
            entropy,
            len(set(tokens))
        ])
    return np.array(feats)

X_train_f = handcrafted(X_train)
X_test_f = handcrafted(X_test)

model = SVC(kernel='rbf', C=5, probability=True)
model.fit(X_train_f, y_train)

preds = model.predict(X_test_f)
probs = model.predict_proba(X_test_f)[:,1]

print("Stylometric SVM")
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))

#Baseline: XGBoost (Handcrafted Features)

import xgboost as xgb

X_train_f = handcrafted(X_train)
X_test_f = handcrafted(X_test)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.03
)

model.fit(X_train_f, y_train)

preds = model.predict(X_test_f)
probs = model.predict_proba(X_test_f)[:,1]

print("XGBoost (ΦH)")
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))

#Baseline: Sentence Transformer + Logistic Regression 

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

model_emb = SentenceTransformer("all-MiniLM-L6-v2")

X_train_emb = model_emb.encode(X_train.tolist())
X_test_emb = model_emb.encode(X_test.tolist())

model = LogisticRegression(max_iter=1000)
model.fit(X_train_emb, y_train)

preds = model.predict(X_test_emb)
probs = model.predict_proba(X_test_emb)[:,1]

print("Sentence Transformer + LR")
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))

#Baseline: BERT-base (Fine-tuned)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

from datasets import Dataset
train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

args = TrainingArguments(
    output_dir="bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_steps=100,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
)

trainer.train()

preds = trainer.predict(test_ds)
probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:,1].numpy()
labels_pred = np.argmax(preds.predictions, axis=1)

print("BERT-base")
print("Accuracy:", accuracy_score(y_test, labels_pred))
print("F1:", f1_score(y_test, labels_pred))
print("AUC:", roc_auc_score(y_test, probs))

#Baseline: RoBERTa-base (Fine-tuned)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

args = TrainingArguments(
    output_dir="roberta",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    save_strategy="no"
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds)
trainer.train()

preds = trainer.predict(test_ds)
probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:,1].numpy()
labels_pred = np.argmax(preds.predictions, axis=1)

print("RoBERTa-base")
print("Accuracy:", accuracy_score(y_test, labels_pred))
print("F1:", f1_score(y_test, labels_pred))
print("AUC:", roc_auc_score(y_test, probs))

