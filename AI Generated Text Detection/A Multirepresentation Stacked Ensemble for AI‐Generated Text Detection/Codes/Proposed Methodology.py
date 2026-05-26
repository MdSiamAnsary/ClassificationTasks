# =========================================================
# INSTALL 
# =========================================================
# pip install numpy pandas scikit-learn torch transformers sentence-transformers xgboost lightgbm catboost faiss-cpu tqdm

# =========================================================
# IMPORTS
# =========================================================
import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import faiss

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================================================
# DATA
# =========================================================
df = pd.read_csv("data.csv")  # columns: text, label
texts = df["text"].astype(str).tolist()
labels = df["label"].values

# Stratified split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# =========================================================
# UTILS
# =========================================================
def save_cache(name, arr):
    np.save(os.path.join(CACHE_DIR, name), arr)

def load_cache(name):
    path = os.path.join(CACHE_DIR, name)
    return np.load(path) if os.path.exists(path) else None

# =========================================================
# TRANSFORMER EMBEDDINGS (BATCH + GPU)
# =========================================================
class TransformerEmbedder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def encode(self, texts):
        all_embeddings = []

        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i:i+BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

# Load models
bert = TransformerEmbedder("bert-base-uncased")
roberta = TransformerEmbedder("roberta-base")
distilbert = TransformerEmbedder("distilbert-base-uncased")

def get_transformer_features(texts, name):
    cache = load_cache(name)
    if cache is not None:
        return cache

    e1 = bert.encode(texts)
    e2 = roberta.encode(texts)
    e3 = distilbert.encode(texts)

    out = np.concatenate([e1, e2, e3], axis=1)
    save_cache(name, out)
    return out

# =========================================================
# CONTRASTIVE FEATURES
# =========================================================
sent_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

def get_contrastive(texts, name):
    cache = load_cache(name)
    if cache is not None:
        return cache

    emb = sent_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    save_cache(name, emb)
    return emb

# =========================================================
# HANDCRAFTED FEATURES
# =========================================================
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

# =========================================================
# LLaMA META FEATURES (REAL)
# =========================================================
print("Loading LLaMA model...")

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

def parse_llama_output(text):
    try:
        a = float(re.search(r"AI:\s*([0-9.]+)", text).group(1))
        f = float(re.search(r"Formality:\s*([0-9.]+)", text).group(1))
        p = float(re.search(r"Predictability:\s*([0-9.]+)", text).group(1))
        return [a, f, p]
    except:
        return [0.5, 0.5, 0.5]

def llama_features(texts, name):
    cache = load_cache(name)
    if cache is not None:
        return cache

    feats = []
    for text in tqdm(texts):
        prompt = f"""
You are an expert linguistic analyst.

Text:
{text}

Return:
AI: <value>, Formality: <value>, Predictability: <value>
"""

        inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)

        with torch.no_grad():
            output = llama_model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.0
            )

        decoded = llama_tokenizer.decode(output[0], skip_special_tokens=True)
        feats.append(parse_llama_output(decoded))

    feats = np.array(feats)
    save_cache(name, feats)
    return feats

# =========================================================
# FEATURE PIPELINE
# =========================================================
def build_features(texts, prefix):
    fT = get_transformer_features(texts, f"{prefix}_T.npy")
    fC = get_contrastive(texts, f"{prefix}_C.npy")
    fH = handcrafted(texts)
    fL = llama_features(texts, f"{prefix}_L.npy")

    return np.concatenate([fT, fC, fH, fL], axis=1)

print("Extracting features...")
X_train_f = build_features(X_train, "train")
X_val_f = build_features(X_val, "val")
X_test_f = build_features(X_test, "test")

# =========================================================
# NORMALIZATION
# =========================================================
scaler = StandardScaler()
X_train_f = scaler.fit_transform(X_train_f)
X_val_f = scaler.transform(X_val_f)
X_test_f = scaler.transform(X_test_f)

# =========================================================
# FEATURE SELECTION
# =========================================================
mi = mutual_info_classif(X_train_f, y_train)
idx = np.argsort(mi)[-500:]

X_train_f = X_train_f[:, idx]
X_val_f = X_val_f[:, idx]
X_test_f = X_test_f[:, idx]

# =========================================================
# FAISS RETRIEVAL (ΦC)
# =========================================================
print("Building FAISS index...")

C_train = get_contrastive(X_train, "train_C.npy").astype("float32")

index = faiss.IndexFlatL2(C_train.shape[1])
index.add(C_train)

def retrieve_neighbors(vecs, k=3):
    D, I = index.search(vecs.astype("float32"), k)
    return I

# =========================================================
# MODELS
# =========================================================
models = [
    xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.03),
    lgb.LGBMClassifier(n_estimators=300, max_depth=9),
    RandomForestClassifier(n_estimators=300, max_depth=20),
    ExtraTreesClassifier(n_estimators=300, max_depth=20),
    GradientBoostingClassifier(n_estimators=250),
    SVC(probability=True, C=5),
    KNeighborsClassifier(n_neighbors=5),
    CatBoostClassifier(verbose=0)
]

# =========================================================
# STACKING (OOF)
# =========================================================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

meta_train = np.zeros((X_train_f.shape[0], len(models)))
meta_test = np.zeros((X_test_f.shape[0], len(models)))

for i, model in enumerate(models):
    oof = np.zeros(X_train_f.shape[0])
    test_preds = np.zeros(X_test_f.shape[0])

    for train_idx, val_idx in kf.split(X_train_f, y_train):
        X_tr, X_val_fold = X_train_f[train_idx], X_train_f[val_idx]
        y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val_fold)[:, 1]
        test_preds += model.predict_proba(X_test_f)[:, 1] / kf.n_splits

    meta_train[:, i] = oof
    meta_test[:, i] = test_preds

# =========================================================
# META MODEL
# =========================================================
meta_model = LogisticRegression()
meta_model.fit(meta_train, y_train)

final_preds = meta_model.predict(meta_test)
final_probs = meta_model.predict_proba(meta_test)[:, 1]

# =========================================================
# EVALUATION
# =========================================================
acc = accuracy_score(y_test, final_preds)
auc = roc_auc_score(y_test, final_probs)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, final_preds)

print("\nConfusion Matrix:")
print(cm)

# Pretty Print
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# -------------------------
# Classification Report
# -------------------------
print("\nClassification Report:")
print(classification_report(y_test, final_preds))