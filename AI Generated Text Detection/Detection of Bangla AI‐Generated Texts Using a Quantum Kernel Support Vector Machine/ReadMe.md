# Detection of Bangla AI-Generated Texts Using a Quantum Kernel SVM 

**"Detection of Bangla AI-Generated Texts Using a Quantum Kernel Support Vector Machine", *Applied AI Letters* 7:e70051 (2026), doi:10.1002/ail2.70051**.



## Environment & configuration

```bash
pip install pennylane==0.45.1 torch transformers scikit-learn pandas numpy matplotlib
```

```python
import os, re, time, random, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ---------------- Configuration ----------------
SEED            = 42                         # random_state = 42 throughout (§5)
DATA_PATH       = "dataset.csv"              # columns: SL, Text, Type, Source, Category, Label
BANGLABERT_PATH = "csebuetnlp/banglabert"    # or a local directory with the weights
N_QUBITS        = 8                          # = number of PCA components (§4.3)
MAX_LEN         = 128                        # BanglaBERT truncation length (§4.2)
TEST_SIZE       = 0.20                       # 480 train / 120 test (§5)
N_FOLDS         = 5
OUT_DIR         = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Paper: all models were run on a single CPU core (§5)
torch.set_num_threads(1)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()
```

---

## Dataset loading & statistics 

```python
df = pd.read_csv(DATA_PATH)
df["Text"] = df["Text"].astype(str)

# §3: no records had missing text or label, so no rows were removed
assert df[["Text", "Label"]].isna().sum().sum() == 0
print("Shape:", df.shape)

y      = df["Label"].to_numpy()          # 0 = HWT (human-written), 1 = AGT (AI-generated)
genres = df["Type"].to_numpy()

GENRE_ORDER = ["Newspaper Article", "Scientific Abstract", "Email",
               "Poem", "Short Story", "Office Order"]

# ---- Class balance ----
print("\nClass balance:\n", df["Category"].value_counts(normalize=True).mul(100).round(1))

# ---- Table 1: genre distribution ----
table1 = (df["Type"].value_counts()
            .reindex(GENRE_ORDER)
            .rename("Texts")
            .to_frame())
table1["% of corpus"] = (100 * table1["Texts"] / len(df)).round(1)
table1.loc["Total"] = [table1["Texts"].sum(), 100.0]
print("\nTable 1 — Genre distribution\n", table1)

# Each genre is split 50/50 between HWT and AGT
print("\nGenre × class:\n", pd.crosstab(df["Type"], df["Category"]).reindex(GENRE_ORDER))

# ---- Text-length statistics (characters, raw text) ----
lengths = df["Text"].str.len()
print(f"\nText length: min={lengths.min()}, max={lengths.max()}, "
      f"mean={lengths.mean():.0f}, median={lengths.median():.0f}")
print("\nMedian length per genre:\n",
      df.assign(len=lengths).groupby("Type")["len"].median().reindex(GENRE_ORDER))
```

**Table 1**

| Genre | Texts | % of corpus |
|---|---|---|
| Newspaper article | 318 | 53.0% |
| Scientific abstract | 162 | 27.0% |
| Email | 54 | 9.0% |
| Poem | 30 | 5.0% |
| Short story | 30 | 5.0% |
| Office order | 6 | 1.0% |
| **Total** | **600** | **100.0%** |

- Class balance: 50.0% HWT / 50.0% AGT (300 / 300); office orders = 3 HWT + 3 AGT
- Text length: **min 72, max 8184, mean 805, median 579** characters
- Median length: **poems 166**, **emails 197** characters (the shortest genres)

---

## Preprocessing 

Line breaks are collapsed to single spaces, Bangla sentence-final and clause punctuation is removed, and repeated whitespace is collapsed. Bangla Unicode text, digits and basic Latin characters are left as they are.

```python
# Bangla sentence-final (dari ।, double dari ॥) and clause punctuation
BN_PUNCT = r"[।॥,;:!?]"

def clean_text(text: str) -> str:
    text = re.sub(r"[\r\n]+", " ", text)     # carriage returns / line breaks -> space
    text = re.sub(BN_PUNCT, " ", text)       # strip sentence-final & clause punctuation
    text = re.sub(r"\s+", " ", text).strip() # collapse repeated whitespace
    return text

df["clean"] = df["Text"].map(clean_text)
texts = df["clean"].to_numpy()
print(df[["Text", "clean"]].head(3))
```

---

## BanglaBERT feature extraction 

BanglaBERT-base (ELECTRA discriminator, 12 layers, hidden size 768, 12 heads, WordPiece vocabulary of 32,000), truncated to 128 tokens. The **last hidden state is mean-pooled over non-padding tokens** (attention-mask-weighted average) to give one 768-d vector per text.

Embeddings are computed **once for the full corpus** and reused for every split and fold, because the encoder is frozen.

```python
from transformers import AutoTokenizer, AutoModel

@torch.no_grad()
def banglabert_embed(text_list, model_path=BANGLABERT_PATH, max_len=MAX_LEN,
                     batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    out = []
    for i in range(0, len(text_list), batch_size):
        enc = tok(list(text_list[i:i + batch_size]), padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        h    = model(**enc).last_hidden_state                 # (B, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()    # (B, T, 1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy())
    return np.vstack(out)

EMB_CACHE = os.path.join(OUT_DIR, "banglabert_mean_emb.npy")
if os.path.exists(EMB_CACHE):
    X_bert = np.load(EMB_CACHE)
else:
    t0 = time.time()
    X_bert = banglabert_embed(texts)
    np.save(EMB_CACHE, X_bert)
    print(f"Embedding time: {time.time() - t0:.1f}s")

print("BanglaBERT embedding matrix:", X_bert.shape)   # (600, 768)
```

Embedding matrix of shape **(600, 768)**.

---

## Dimensionality reduction, scaling & retained variance

PCA to 8 components (one per qubit), then MinMax scaling to **[0, π]**. Both are **fit on the training split only**; test values are clipped to the range. The TF-IDF representation used in the ablation combines the 3000 most frequent word 1–2-grams and the 3000 most frequent character 3–5-grams, fit on training data only.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

def fit_reduce(X_tr, X_te, n_components=N_QUBITS):
    """PCA(8) -> MinMax[0, pi]; fit on train, applied (and clipped) on test."""
    pca = PCA(n_components=n_components, random_state=SEED).fit(X_tr)
    Z_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)
    scaler = MinMaxScaler(feature_range=(0, np.pi), clip=True).fit(Z_tr)
    return scaler.transform(Z_tr), scaler.transform(Z_te), pca

def make_tfidf():
    """Word 1-2-grams (top 3000) + char 3-5-grams (top 3000).
    Whitespace tokenisation keeps Bangla vowel signs (matras) inside words;
    sklearn's default token_pattern would split words at combining marks."""
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=3000,
                                 tokenizer=str.split, token_pattern=None, lowercase=False)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=3000,
                                 lowercase=False)),
    ])

# ---- Feature builders used by every experiment (refit inside each split / fold) ----
def bert_features(tr_idx, te_idx):
    Xtr, Xte, _ = fit_reduce(X_bert[tr_idx], X_bert[te_idx])
    return Xtr, Xte

def tfidf_features(tr_idx, te_idx):
    vec = make_tfidf().fit(texts[tr_idx])
    Xtr = vec.transform(texts[tr_idx]).toarray()
    Xte = vec.transform(texts[te_idx]).toarray()
    Xtr8, Xte8, _ = fit_reduce(Xtr, Xte)
    return Xtr8, Xte8

# ---- Held-out split (§5): 480 train / 120 test, stratified, seed 42 ----
all_idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(all_idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
print("Train / test:", len(tr_idx), len(te_idx),
      "| test class balance:", np.bincount(y[te_idx]))

# ---- Retained variance of 8 PCs (descriptive, §4.3 / §6.3) ----
pca_bert_full = PCA(n_components=N_QUBITS, random_state=SEED).fit(X_bert)
print(f"BanglaBERT: 8 PCs retain {100 * pca_bert_full.explained_variance_ratio_.sum():.1f}% variance")

tfidf_vec = make_tfidf().fit(texts[tr_idx])              # vectoriser fit on training data only
X_tfidf_full = tfidf_vec.transform(texts).toarray()
pca_tfidf_full = PCA(n_components=N_QUBITS, random_state=SEED).fit(X_tfidf_full)
print(f"TF-IDF:     8 PCs retain {100 * pca_tfidf_full.explained_variance_ratio_.sum():.1f}% variance")
```



| Representation | Variance retained by 8 PCs (full corpus) |
|---|---|
| BanglaBERT (768-d) | **74.4%** |
| TF-IDF (3000 word 1–2-g + 3000 char 3–5-g) | **13.0%** (13.2% for the vectoriser configuration used in the original TF-IDF VQC run) |

Split: **480 train / 120 test**, 60 / 60 per class in the test set.

---

## Quantum feature map & fidelity kernel 

Feature map on 8 qubits starting from |0⟩^⊗8 (Eq. 1):

$$U(x) = \mathrm{AE}_Z(x)\cdot\prod_{i=0}^{7}\mathrm{CZ}\big(i,(i+1)\bmod 8\big)\cdot\mathrm{AE}_Y(x)$$

Kernel (Eq. 2): $K(x_1,x_2)=\left|\langle 0|^{\otimes 8}U^\dagger(x_2)U(x_1)|0\rangle^{\otimes 8}\right|^2$

Two equivalent implementations are given:
- **PennyLane `default.qubit`** (v0.45.1), used for the main held-out results. It runs batched with parameter broadcasting and computes only the upper triangle of the symmetric train–train matrix.
- **Exact NumPy state-vector simulation**, used for the revision experiments (classical-baseline CV, TF-IDF ablation, per-genre analysis). It is mathematically identical for this noiseless circuit.

```python
import pennylane as qml

WIRES = list(range(N_QUBITS))

# ---------------- PennyLane implementation ----------------
def feature_map(x):
    qml.AngleEmbedding(x, wires=WIRES, rotation="Y")          # AE_Y(x)
    for i in range(N_QUBITS):                                 # CZ ring
        qml.CZ(wires=[i, (i + 1) % N_QUBITS])
    qml.AngleEmbedding(x, wires=WIRES, rotation="Z")          # AE_Z(x)

kernel_dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(kernel_dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    # probability of returning to |0...0> via a projector
    return qml.expval(qml.Projector([0] * N_QUBITS, wires=WIRES))

def qkernel_matrix_pl(A, B=None, batch=4096):
    """Kernel matrix with PennyLane, broadcasting many (x1, x2) pairs per simulator call.
    If B is None, the symmetric A-A matrix is built from its upper triangle only."""
    symmetric = B is None
    B = A if symmetric else B
    if symmetric:
        ii, jj = np.triu_indices(len(A))
    else:
        ii, jj = (g.ravel() for g in np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij"))
    vals = np.empty(len(ii))
    for s in range(0, len(ii), batch):
        vals[s:s + batch] = np.asarray(kernel_circuit(A[ii[s:s + batch]], B[jj[s:s + batch]]))
    K = np.zeros((len(A), len(B)))
    K[ii, jj] = vals
    if symmetric:
        K = K + K.T - np.diag(np.diag(K))
    return K

# ---------------- Exact NumPy implementation ----------------
_IDX  = np.arange(2 ** N_QUBITS)
BITS  = (_IDX[:, None] >> (N_QUBITS - 1 - np.arange(N_QUBITS))) & 1          # wire 0 = MSB
CZ_PHASE = (-1.0) ** np.sum(BITS * np.roll(BITS, -1, axis=1), axis=1)         # ring (i, i+1 mod 8)

def feature_states(X):
    """|psi(x)> = U(x)|0>^8 for every row of X -> (N, 256) complex."""
    X = np.asarray(X, dtype=float)
    c, s = np.cos(X / 2), np.sin(X / 2)                                        # RY(x)|0>
    amp = np.where(BITS[None] == 1, s[:, None, :], c[:, None, :]).prod(axis=2) # product state
    psi = amp * CZ_PHASE[None, :]                                              # CZ ring
    rz  = np.exp(1j * ((2 * BITS[None] - 1) * X[:, None, :] / 2).sum(axis=2))  # RZ(x) phases
    return psi * rz

def qkernel_matrix_np(A, B=None):
    SA = feature_states(A)
    SB = SA if B is None else feature_states(B)
    return np.abs(SA.conj() @ SB.T) ** 2

# ---------------- Sanity check: both implementations agree ----------------
_rng = np.random.default_rng(SEED)
_xa, _xb = _rng.uniform(0, np.pi, (5, N_QUBITS)), _rng.uniform(0, np.pi, (4, N_QUBITS))
assert np.allclose(qkernel_matrix_pl(_xa, _xb), qkernel_matrix_np(_xa, _xb), atol=1e-8)
assert np.allclose(np.diag(qkernel_matrix_np(_xa)), 1.0)
print("PennyLane and NumPy kernels agree.")
print(qml.draw(kernel_circuit)(_xa[0], _xb[0]))
```

On a single CPU core, the full **480 × 480 train–train** and **120 × 480 test–train** kernel matrices take **about 15 s** with PennyLane. Kernel diagonal = 1.

---

## Models 

All four models are trained on the **same PCA(8) → [0, π] features**.

- **QSVM**: SVC with `kernel="precomputed"`, C = 1.0, Platt-scaled probabilities
- **Logistic Regression**: L2, scikit-learn defaults, `max_iter=1000`
- **Classical SVM (RBF)**: `SVC(kernel="rbf")`
- **VQC**: 8-qubit AngleEmbedding, then a 3-layer StronglyEntanglingLayers ansatz, Pauli-Z readout, sigmoid output; Adam, BCE loss, 60 epochs, batch size 16 (PennyLane + PyTorch)

```python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class QSVM:
    """Fidelity quantum kernel + classical SVM (no trainable circuit parameters)."""
    def __init__(self, C=1.0, kernel_fn=qkernel_matrix_np):
        self.C, self.kernel_fn = C, kernel_fn

    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        t0 = time.time()
        K_train = self.kernel_fn(self.X_train_)
        self.kernel_time_train_ = time.time() - t0
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True,
                        random_state=SEED).fit(K_train, y)
        return self

    def predict_with_score(self, X):
        t0 = time.time()
        K = self.kernel_fn(np.asarray(X), self.X_train_)          # (n_test, n_train)
        self.kernel_time_test_ = time.time() - t0
        return self.svc_.predict(K), self.svc_.predict_proba(K)[:, 1]


class _VQCNet(torch.nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")   # same encoding as QSVM
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))                                 # Pauli-Z readout

        shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}
        self.q = qml.qnn.TorchLayer(circuit, shapes)
        # Trainable affine rescaling of <Z> in [-1, 1] before the sigmoid, so the output
        # can approach 0/1. (The paper specifies only "Pauli-Z readout + sigmoid".)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.bias  = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.scale * self.q(x).reshape(-1) + self.bias     # logit


class VQC:
    def __init__(self, n_qubits=N_QUBITS, n_layers=3, epochs=60, batch_size=16, lr=0.01, seed=SEED):
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed

    def fit(self, X, y, verbose=False):
        set_seed(self.seed)
        self.net_ = _VQCNet(self.n_qubits, self.n_layers)
        opt  = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()                    # sigmoid + binary cross-entropy
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        g  = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                             batch_size=self.batch_size, shuffle=True, generator=g)
        self.history_ = []
        for ep in range(self.epochs):
            self.net_.train(); total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.net_(xb), yb)
                loss.backward(); opt.step()
                total += loss.item() * len(xb)
            self.history_.append(total / len(Xt))
            if verbose and (ep + 1) % 10 == 0:
                print(f"  epoch {ep + 1:02d}  loss {self.history_[-1]:.4f}")
        return self

    @torch.no_grad()
    def predict_with_score(self, X):
        self.net_.eval()
        p = torch.sigmoid(self.net_(torch.tensor(X, dtype=torch.float32))).numpy()
        return (p >= 0.5).astype(int), p


class SklearnWrapper:
    def __init__(self, est): self.est = est
    def fit(self, X, y): self.est.fit(X, y); return self
    def predict_with_score(self, X):
        return self.est.predict(X), self.est.predict_proba(X)[:, 1]


def make_models(qsvm_kernel=qkernel_matrix_np, include_vqc=True):
    """Factories so each split / fold gets a fresh model."""
    m = {
        "Logistic Regression": lambda: SklearnWrapper(LogisticRegression(max_iter=1000)),
        "Classical SVM (RBF)": lambda: SklearnWrapper(SVC(kernel="rbf", probability=True, random_state=SEED)),
    }
    if include_vqc:
        m["VQC"] = lambda: VQC()
    m["QSVM"] = lambda: QSVM(C=1.0, kernel_fn=qsvm_kernel)
    return m
```

---

## Evaluation helpers 

Metrics are computed with **AGT (Label = 1) as the positive class**. CV spread is the **population SD** (`ddof=0`) over the five folds. Per-genre accuracy uses **95% Wilson score intervals**.

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold

def compute_metrics(y_true, y_pred, y_score=None):
    m = {
        "Acc.":  accuracy_score(y_true, y_pred),
        "Prec.": precision_score(y_true, y_pred, pos_label=1),
        "Rec.":  recall_score(y_true, y_pred, pos_label=1),
        "F1":    f1_score(y_true, y_pred, pos_label=1),
        "AUC":   roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
    }
    m["CM"] = confusion_matrix(y_true, y_pred, labels=[0, 1])   # rows: true HWT/AGT
    return m

def wilson_ci(k, n, z=1.959963984540054):
    """95% Wilson score interval for k successes out of n."""
    p = k / n
    denom  = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half

def run_holdout(feature_fn, model_factory, tr=tr_idx, te=te_idx):
    Xtr, Xte = feature_fn(tr, te)
    model = model_factory().fit(Xtr, y[tr])
    y_pred, y_score = model.predict_with_score(Xte)
    return compute_metrics(y[te], y_pred, y_score), y_pred, y_score, model

def run_cv(feature_fn, model_factory, n_splits=N_FOLDS):
    """5-fold stratified CV; feature pipeline refit in every fold (no leakage).
    Returns per-fold accuracies plus pooled out-of-fold predictions/scores."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_acc = []
    oof_pred  = np.full(len(y), -1)
    oof_score = np.full(len(y), np.nan)
    for tr, va in skf.split(np.zeros(len(y)), y):
        Xtr, Xva = feature_fn(tr, va)
        model = model_factory().fit(Xtr, y[tr])
        p, s = model.predict_with_score(Xva)
        oof_pred[va], oof_score[va] = p, s
        fold_acc.append(accuracy_score(y[va], p))
    fold_acc = np.array(fold_acc)
    return {"folds": fold_acc, "mean": fold_acc.mean(), "sd": fold_acc.std(ddof=0),
            "oof_pred": oof_pred, "oof_score": oof_score}

def fmt_cv(r): return f"{r['mean']:.3f} ± {r['sd']:.3f}"
```

---

## Held-out test split 

The main held-out QSVM result uses the **PennyLane `default.qubit`** kernel.

```python
holdout_bert = {}
for name, factory in make_models(qsvm_kernel=qkernel_matrix_pl).items():
    t0 = time.time()
    metrics, y_pred, y_score, model = run_holdout(bert_features, factory)
    holdout_bert[name] = {"metrics": metrics, "pred": y_pred, "score": y_score, "model": model}
    print(f"{name:22s} done in {time.time() - t0:6.1f}s  "
          f"(correct = {int(round(metrics['Acc.'] * len(te_idx)))}/{len(te_idx)})")

q = holdout_bert["QSVM"]["model"]
print(f"\nQSVM kernel time: train-train {q.kernel_time_train_:.1f}s, test-train {q.kernel_time_test_:.1f}s")

table2 = pd.DataFrame({k: {m: v["metrics"][m] for m in ["Acc.", "Prec.", "Rec.", "F1", "AUC"]}
                       for k, v in holdout_bert.items()}).T.round(3)
table2 = table2.loc[["Logistic Regression", "Classical SVM (RBF)", "VQC", "QSVM"]]
print("\nTable 2 — Held-out test performance (BanglaBERT, 8 qubits / 8 PCs)\n", table2)

for name in ["QSVM", "VQC", "Logistic Regression", "Classical SVM (RBF)"]:
    print(f"\nConfusion matrix — {name} (rows true HWT/AGT, cols predicted HWT/AGT)\n",
          holdout_bert[name]["metrics"]["CM"])
table2.to_csv(os.path.join(OUT_DIR, "table2_holdout.csv"))
```

**Table 2**

| Model | Acc. | Prec. | Rec. | F1 | AUC | Correct / 120 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.958 | 0.951 | 0.967 | 0.959 | 0.972 | 115 |
| Classical SVM (RBF) | 0.933 | 0.948 | 0.917 | 0.932 | 0.987 | 112 |
| VQC (trainable circuit) | 0.875 | 0.881 | 0.867 | 0.874 | 0.943 | 105 |
| **QSVM (proposed)** | **0.950** | **0.950** | **0.950** | **0.950** | **0.989** | **114** |

**Confusion matrices** (rows = true HWT, AGT; columns = predicted HWT, AGT)

| Model | Matrix | Source |
|---|---|---|
| QSVM | `[[57, 3], [3, 57]]` (TN 57, FP 3, FN 3, TP 57) | Fig. 2 |
| VQC | `[[53, 7], [8, 52]]` (TN 53, FP 7, FN 8, TP 52) | Fig. 2 |
| Logistic Regression | `[[57, 3], [2, 58]]` | derived from Table 2 |
| Classical SVM (RBF) | `[[57, 3], [5, 55]]` | derived from Table 2 |

---

## 5-fold stratified cross-validation 

PCA and the scaler are **refit inside every fold**. QSVM uses the PennyLane default.qubit kernel, with the NumPy kernel as an equivalent option. Pooled out-of-fold predictions are kept for the per-genre analysis.

```python
cv_bert = {}
for name, factory in make_models(qsvm_kernel=qkernel_matrix_pl).items():   # or qkernel_matrix_np
    t0 = time.time()
    cv_bert[name] = run_cv(bert_features, factory)
    print(f"{name:22s} {fmt_cv(cv_bert[name])}   ({time.time() - t0:.1f}s)")

rows = ["QSVM", "Logistic Regression", "Classical SVM (RBF)", "VQC"]
table3 = pd.DataFrame(
    [[*np.round(cv_bert[r]["folds"], 3), fmt_cv(cv_bert[r])] for r in rows],
    index=rows, columns=["F1", "F2", "F3", "F4", "F5", "Mean ± SD"])
print("\nTable 3 — 5-fold stratified CV accuracy (population SD)\n", table3)

# Per-fold QSVM-vs-VQC margin (paper: QSVM better in every fold, by 7.5-12.5 points)
margin = 100 * (cv_bert["QSVM"]["folds"] - cv_bert["VQC"]["folds"])
print("\nQSVM − VQC per fold (points):", np.round(margin, 1))
print("QSVM worst fold:", cv_bert["QSVM"]["folds"].min().round(3),
      "| VQC mean:", cv_bert["VQC"]["mean"].round(3))
table3.to_csv(os.path.join(OUT_DIR, "table3_cv.csv"))
```

**Table 3**

| Model | F1 | F2 | F3 | F4 | F5 | Mean ± SD |
|---|---|---|---|---|---|---|
| **QSVM** | 0.942 | 0.967 | 0.992 | 0.908 | 0.950 | **0.952 ± 0.028** |
| Logistic Regression | 0.942 | 0.967 | 0.975 | 0.917 | 0.950 | 0.950 ± 0.020 |
| Classical SVM (RBF) | 0.925 | 0.958 | 0.967 | 0.900 | 0.942 | 0.938 ± 0.024 |
| VQC | 0.867 | 0.867 | 0.908 | 0.783 | 0.867 | 0.858 ± 0.041 |

- Each fold has 120 texts, so one text = 0.83 points
- QSVM beats VQC in **every fold by 7.5–12.5 points**; QSVM's worst fold (**0.908, Fold 4**) is above the VQC mean (0.858)

---

## Feature-extractor ablation: TF-IDF vs BanglaBERT 

The pipeline is repeated (PCA → quantum feature map → classifier) with TF-IDF in place of BanglaBERT. Qubit count, circuit and training stay the same, and the **TF-IDF vectoriser is refit within each split and fold**. In the paper, the VQC on TF-IDF was evaluated **on the held-out split only**.

```python
# ---- Held-out split with TF-IDF features ----
holdout_tfidf = {}
for name, factory in make_models(qsvm_kernel=qkernel_matrix_np).items():
    metrics, y_pred, y_score, _ = run_holdout(tfidf_features, factory)
    holdout_tfidf[name] = {"metrics": metrics, "pred": y_pred, "score": y_score}
    print(f"{name:22s} Acc {metrics['Acc.']:.3f}  AGT-Rec {metrics['Rec.']:.3f}  AUC {metrics['AUC']:.3f}")
print("\nVQC (TF-IDF) confusion matrix:\n", holdout_tfidf["VQC"]["metrics"]["CM"])

# ---- 5-fold CV with TF-IDF features (VQC not run under CV, as in the paper) ----
cv_tfidf = {}
for name, factory in make_models(qsvm_kernel=qkernel_matrix_np, include_vqc=False).items():
    cv_tfidf[name] = run_cv(tfidf_features, factory)
    print(f"CV {name:22s} {fmt_cv(cv_tfidf[name])}")

# ---- Assemble Table 4 ----
rows = []
for name in ["QSVM", "VQC", "Logistic Regression", "Classical SVM (RBF)"]:
    for feat, ho, cv in [("BanglaBERT", holdout_bert, cv_bert), ("TF-IDF", holdout_tfidf, cv_tfidf)]:
        m = ho[name]["metrics"]
        is_vqc_tfidf = (name == "VQC" and feat == "TF-IDF")
        rows.append({
            "Model": name, "Features": feat,
            "Acc.": round(m["Acc."], 3), "AGT Rec.": round(m["Rec."], 3),
            "AUC": "n/a" if is_vqc_tfidf else round(m["AUC"], 3),      # paper reports n/a
            "5-fold CV Acc.": fmt_cv(cv[name]) if name in cv else "n/a",
        })
table4 = pd.DataFrame(rows)
print("\nTable 4 — Feature-extractor ablation\n", table4.to_string(index=False))
table4.to_csv(os.path.join(OUT_DIR, "table4_ablation.csv"), index=False)
```

**Table 4**

| Model | Features | Acc. | AGT Rec. | AUC | 5-fold CV Acc. |
|---|---|---|---|---|---|
| QSVM | BanglaBERT | 0.950 | 0.950 | 0.989 | 0.952 ± 0.028 |
| QSVM | TF-IDF | 0.892 | 0.833 | 0.984 | 0.937 ± 0.027 |
| VQC | BanglaBERT | 0.875 | 0.867 | 0.943 | 0.858 ± 0.041 |
| VQC | TF-IDF | 0.758 | 0.533 | n/a | n/a |
| Logistic Regression | BanglaBERT | 0.958 | 0.967 | 0.972 | 0.950 ± 0.020 |
| Logistic Regression | TF-IDF | 0.875 | 0.817 | 0.951 | 0.910 ± 0.029 |
| Classical SVM (RBF) | BanglaBERT | 0.933 | 0.917 | 0.987 | 0.938 ± 0.024 |
| Classical SVM (RBF) | TF-IDF | 0.883 | 0.817 | 0.979 | 0.935 ± 0.032 |

- QSVM: held-out accuracy 95.0% → 89.2%, AGT recall 95.0% → 83.3%, AUC 0.989 → 0.984; CV 95.2% → 93.7% (about 1.5 points, less than one SD)
- VQC: accuracy 87.5% → 75.8%, AGT recall 86.7% → 53.3%; the confusion matrix skews heavily towards predicting HWT
- The paper's TF-IDF VQC row comes from the original run, whose vectoriser retained **13.2%** variance (13.0% for the other rows)

---

## Per-genre accuracy with 95% Wilson intervals 

Computed from the **pooled out-of-fold predictions of the 5-fold protocol** (BanglaBERT features), so every text is predicted exactly once by a model that has not seen it.

```python
def per_genre_table(cv_results, model_names=("QSVM", "Logistic Regression", "Classical SVM (RBF)")):
    rows = []
    for g in GENRE_ORDER:
        mask = genres == g
        n = int(mask.sum())
        row = {"Genre": g, "n": n}
        for name in model_names:
            k = int((cv_results[name]["oof_pred"][mask] == y[mask]).sum())
            lo, hi = wilson_ci(k, n)
            row[name] = f"{100 * k / n:.1f} ({100 * lo:.0f}–{100 * hi:.0f})"
            row[f"{name} correct"] = f"{k}/{n}"
        rows.append(row)
    return pd.DataFrame(rows)

table5 = per_genre_table(cv_bert)
print("Table 5 — Per-genre accuracy, % (95% Wilson interval)\n")
print(table5[["Genre", "n", "QSVM", "Logistic Regression", "Classical SVM (RBF)"]].to_string(index=False))
print("\nRaw counts:\n", table5[["Genre", "QSVM correct", "Logistic Regression correct",
                                "Classical SVM (RBF) correct"]].to_string(index=False))

# Pooled OOF accuracy overall = mean of the five equal-size folds
print("\nQSVM pooled OOF accuracy:", accuracy_score(y, cv_bert["QSVM"]["oof_pred"]).round(4))
table5.to_csv(os.path.join(OUT_DIR, "table5_per_genre.csv"), index=False)
```

**Table 5**

| Genre | n | QSVM | LR | RBF-SVM |
|---|---|---|---|---|
| Newspaper article | 318 | 97.5 (95–99) | 97.2 (95–99) | 96.2 (94–98) |
| Scientific abstract | 162 | 98.1 (95–99) | 98.1 (95–99) | 97.5 (94–99) |
| Email | 54 | 85.2 (73–92) | 83.3 (71–91) | 81.5 (69–90) |
| Poem | 30 | 80.0 (63–90) | 83.3 (66–93) | 80.0 (63–90) |
| Short story | 30 | 90.0 (74–97) | 90.0 (74–97) | 86.7 (70–95) |
| Office order | 6 | 83.3 (44–97) | 83.3 (44–97) | 83.3 (44–97) |

- Implied QSVM counts: 310/318, 159/162, 46/54, 24/30, 27/30, 5/6, a **total of 571/600 = 95.2%**, matching the CV mean
- For office orders, one error moves accuracy by about 17 points
- Lowest accuracy is on poems and emails, the genres with the shortest texts

---

## Figures 2, 3 and 4

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

LABELS = ["HWT", "AGT"]

# ---- Figure 2: confusion matrices on the 120-sample held-out split ----
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name, title in zip(axes, ["QSVM", "VQC"], ["QSVM", "VQC"]):
    ConfusionMatrixDisplay(holdout_bert[name]["metrics"]["CM"], display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig2_confusion_matrices.png"), dpi=300); plt.show()

# ---- Figure 3: ROC curves for QSVM, LR, RBF-SVM (VQC omitted, as in the paper) ----
ROC_MODELS = {"QSVM": "QSVM", "Logistic Regression": "LR", "Classical SVM (RBF)": "RBF-SVM"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

# (a) held-out split
for name, short in ROC_MODELS.items():
    s = holdout_bert[name]["score"]
    fpr, tpr, _ = roc_curve(y[te_idx], s)
    axes[0].plot(fpr, tpr, label=f"{short} (AUC = {roc_auc_score(y[te_idx], s):.3f})")
axes[0].set_title("(a) Held-out split")

# (b) pooled out-of-fold predictions of the 5-fold protocol
for name, short in ROC_MODELS.items():
    s = cv_bert[name]["oof_score"]
    fpr, tpr, _ = roc_curve(y, s)
    axes[1].plot(fpr, tpr, label=f"{short} (AUC = {roc_auc_score(y, s):.3f})")
axes[1].set_title("(b) Pooled out-of-fold (5-fold)")

for ax in axes:
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig3_roc_curves.png"), dpi=300); plt.show()

# ---- Figure 4: held-out accuracy, F1 and ROC-AUC of the four models ----
order  = ["Logistic Regression", "Classical SVM (RBF)", "VQC", "QSVM"]
metrics_to_plot = ["Acc.", "F1", "AUC"]
x = np.arange(len(order)); w = 0.25
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, m in enumerate(metrics_to_plot):
    vals = [holdout_bert[n]["metrics"][m] for n in order]
    bars = ax.bar(x + (i - 1) * w, vals, w, label={"Acc.": "Accuracy", "F1": "F1-score", "AUC": "ROC-AUC"}[m])
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
ax.set_xticks(x); ax.set_xticklabels(["LR", "RBF-SVM", "VQC", "QSVM"])
ax.set_ylim(0.80, 1.00)                                  # vertical axis starts at 0.80 (paper)
ax.set_ylabel("Score"); ax.legend(ncol=3, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig4_summary_bars.png"), dpi=300); plt.show()
```



- **Fig. 2**: QSVM `[[57, 3], [3, 57]]`; VQC `[[53, 7], [8, 52]]`
- **Fig. 3a** (held-out AUC): QSVM 0.989, RBF-SVM 0.987, LR 0.972. The VQC is omitted because its per-sample scores were not retained in the original run; its held-out AUC is 0.943
- **Fig. 3b**: pooled out-of-fold ROC for the same three models (the paper gives no AUC values)
- **Fig. 4**: held-out accuracy, F1 and AUC from Table 2, y-axis starting at 0.80

---

## Comparison with existing literature on the same dataset 

These baselines come from the paper that introduced the dataset (Ansary, QPAIN 2025, ref. [11]). They were trained under **a different protocol** (70/30 split, stratified for SVM and KNN only, no PCA restriction), so this is **not re-run**. The values are listed for reference next to the QSVM result from this run.

```python
table6 = pd.DataFrame([
    ["Multinomial Naive Bayes [11]",   "TF-IDF",                               0.742, 0.869, 0.521, 0.465],
    ["Random Forest [11]",             "TF-IDF",                               0.753, 0.874, 0.542, 0.505],
    ["SVM (polynomial, degree 6) [11]","TF-IDF",                               0.764, 0.382, 0.500, 0.433],
    ["KNN (k = 15) [11]",              "TF-IDF",                               0.809, 0.737, 0.694, 0.710],
    ["CNN–LSTM [11]",                  "BanglaBERT embeddings",                0.764, 0.382, 0.500, 0.433],
    ["Fine-tuned BanglaBERT [11]",     "BanglaBERT (fine-tuned)",              0.955, 0.972, 0.905, 0.933],
], columns=["Model", "Features", "Accuracy", "Precision", "Recall", "F1-score"])

# Macro-averaged metrics for the QSVM from this run (held-out split)
yt, yp = y[te_idx], holdout_bert["QSVM"]["pred"]
table6.loc[len(table6)] = ["QSVM (proposed, this run)", "BanglaBERT + PCA(8) + quantum kernel",
                           round(accuracy_score(yt, yp), 3),
                           round(precision_score(yt, yp, average="macro"), 3),
                           round(recall_score(yt, yp, average="macro"), 3),
                           round(f1_score(yt, yp, average="macro"), 3)]
print("Table 6 — Comparison with baselines on the same dataset (macro-averaged)\n",
      table6.to_string(index=False))
table6.to_csv(os.path.join(OUT_DIR, "table6_literature.csv"), index=False)
```

QSVM (proposed) **0.950 / 0.950 / 0.950 / 0.950** (Acc / Prec / Rec / F1). Fine-tuned BanglaBERT [11] is the strongest at 0.955 accuracy and 0.933 F1.

---

## Results 

| Experiment | Key reported values |
|---|---|
| Dataset | 600 texts, 300 HWT / 300 AGT, 6 genres; length 72–8184 chars (mean 805, median 579) |
| PCA retained variance (8 PCs) | BanglaBERT 74.4%; TF-IDF 13.0% |
| Held-out QSVM | Acc 0.950, Prec 0.950, Rec 0.950, F1 0.950, AUC 0.989; CM `[[57,3],[3,57]]` |
| Held-out LR / RBF-SVM / VQC | Acc 0.958 / 0.933 / 0.875; AUC 0.972 / 0.987 / 0.943 |
| 5-fold CV (mean ± pop. SD) | QSVM 0.952 ± 0.028; LR 0.950 ± 0.020; RBF-SVM 0.938 ± 0.024; VQC 0.858 ± 0.041 |
| TF-IDF ablation (QSVM) | held-out Acc 0.892, AGT Rec 0.833, AUC 0.984; CV 0.937 ± 0.027 |
| TF-IDF ablation (VQC) | held-out Acc 0.758, AGT Rec 0.533 |
| TF-IDF ablation (LR / RBF) | held-out Acc 0.875 / 0.883; CV 0.910 ± 0.029 / 0.935 ± 0.032 |
| Per-genre QSVM (OOF) | News 97.5, Sci. abstract 98.1, Email 85.2, Poem 80.0, Short story 90.0, Office order 83.3 |
| Runtime | Kernel matrices about 15 s; full pipeline about 8–10 min on one CPU core |


