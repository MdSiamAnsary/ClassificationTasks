import numpy as np
import pandas as pd
import re
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# Load data
df = pd.read_csv("data.csv")
texts = df["text"].astype(str).tolist()
labels = df["label"].values

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# -------------------------
# FEATURE FUNCTIONS
# -------------------------

# ΦT (Transformer - single for simplicity)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model_t = AutoModel.from_pretrained("bert-base-uncased")

def get_transformer(texts):
    feats = []
    with torch.no_grad():
        for t in texts:
            inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=256)
            out = model_t(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
            feats.append(emb)
    return np.array(feats)

# ΦC
sent_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_contrastive(texts):
    return sent_model.encode(texts)

# ΦH
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

# ΦL (simple proxy for ablation)
def meta_features(texts):
    feats = []
    for t in texts:
        feats.append([
            len(t)/1000,
            sum(c.isupper() for c in t)/len(t),
            len(set(t.split()))/(len(t.split())+1e-9)
        ])
    return np.array(feats)

# -------------------------
# TRAIN FUNCTION
# -------------------------
def run_model(X_train_f, X_test_f):
    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(X_train_f)
    X_test_f = scaler.transform(X_test_f)

    mi = mutual_info_classif(X_train_f, y_train)
    idx = np.argsort(mi)[-500:]

    X_train_f = X_train_f[:, idx]
    X_test_f = X_test_f[:, idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_f, y_train)

    preds = model.predict(X_test_f)
    probs = model.predict_proba(X_test_f)[:,1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))

# FULL MODEL (ΦT + ΦC + ΦH + ΦL)
X_train_f = np.concatenate([
    get_transformer(X_train),
    get_contrastive(X_train),
    handcrafted(X_train),
    meta_features(X_train)
], axis=1)

X_test_f = np.concatenate([
    get_transformer(X_test),
    get_contrastive(X_test),
    handcrafted(X_test),
    meta_features(X_test)
], axis=1)

print("FULL MODEL")
run_model(X_train_f, X_test_f)

# w/o ΦL (No Meta Features)
X_train_f = np.concatenate([
    get_transformer(X_train),
    get_contrastive(X_train),
    handcrafted(X_train)
], axis=1)

X_test_f = np.concatenate([
    get_transformer(X_test),
    get_contrastive(X_test),
    handcrafted(X_test)
], axis=1)

print("w/o ΦL")
run_model(X_train_f, X_test_f)

#w/o ΦH (No Handcrafted)
X_train_f = np.concatenate([
    get_transformer(X_train),
    get_contrastive(X_train),
    meta_features(X_train)
], axis=1)

X_test_f = np.concatenate([
    get_transformer(X_test),
    get_contrastive(X_test),
    meta_features(X_test)
], axis=1)

print("w/o ΦH")
run_model(X_train_f, X_test_f)

#w/o ΦC (No Contrastive)
X_train_f = np.concatenate([
    get_transformer(X_train),
    handcrafted(X_train),
    meta_features(X_train)
], axis=1)

X_test_f = np.concatenate([
    get_transformer(X_test),
    handcrafted(X_test),
    meta_features(X_test)
], axis=1)

print("w/o ΦC")
run_model(X_train_f, X_test_f)

#w/o ΦT (No Transformer)
X_train_f = np.concatenate([
    get_contrastive(X_train),
    handcrafted(X_train),
    meta_features(X_train)
], axis=1)

X_test_f = np.concatenate([
    get_contrastive(X_test),
    handcrafted(X_test),
    meta_features(X_test)
], axis=1)

print("w/o ΦT")
run_model(X_train_f, X_test_f)

#TRANSFORMER ONLY (ΦT only)
X_train_f = get_transformer(X_train)
X_test_f = get_transformer(X_test)

print("Transformer Only")
run_model(X_train_f, X_test_f)