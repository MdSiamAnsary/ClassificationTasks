# =========================================================
# CROSS-DATASET GENERALIZATION EXPERIMENT
# =========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# =========================================================
# LOAD DATASETS
# =========================================================
# Dataset 1
df1 = pd.read_csv("dataset1.csv")  # DAIGT V2
X1 = df1["text"].astype(str).tolist()
y1 = df1["label"].values

# Dataset 2
df2 = pd.read_csv("dataset2.csv")  # LLM Detect dataset
X2 = df2["text"].astype(str).tolist()
y2 = df2["label"].values

# =========================================================
# IMPORT FEATURE PIPELINE FROM MAIN CODE
# (Make sure these functions exist in same file or import)
# =========================================================
# REQUIRED:
# - build_features()
# - get_contrastive()
# - caching system (optional but recommended)

# =========================================================
# STACKING FUNCTION
# =========================================================
def train_stacked_model(X_train, y_train, X_test):

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

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    meta_train = np.zeros((X_train.shape[0], len(models)))
    meta_test = np.zeros((X_test.shape[0], len(models)))

    for i, model in enumerate(models):
        oof = np.zeros(X_train.shape[0])
        test_preds = np.zeros(X_test.shape[0])

        for tr_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

        meta_train[:, i] = oof
        meta_test[:, i] = test_preds

    meta_model = LogisticRegression()
    meta_model.fit(meta_train, y_train)

    final_preds = meta_model.predict(meta_test)
    final_probs = meta_model.predict_proba(meta_test)[:, 1]

    return final_preds, final_probs


# =========================================================
# CORE FUNCTION: TRAIN → TEST PIPELINE
# =========================================================
def cross_dataset_experiment(train_texts, train_labels, test_texts, test_labels, tag):

    print(f"\n==============================")
    print(f"Experiment: {tag}")
    print(f"==============================")

    # -------------------------
    # Feature extraction
    # -------------------------
    X_train_f = build_features(train_texts, f"{tag}_train")
    X_test_f = build_features(test_texts, f"{tag}_test")

    # -------------------------
    # Normalize (ONLY TRAIN FIT)
    # -------------------------
    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(X_train_f)
    X_test_f = scaler.transform(X_test_f)

    # -------------------------
    # Feature Selection (ONLY TRAIN)
    # -------------------------
    mi = mutual_info_classif(X_train_f, train_labels)
    idx = np.argsort(mi)[-500:]

    X_train_f = X_train_f[:, idx]
    X_test_f = X_test_f[:, idx]

    # -------------------------
    # Train + Predict
    # -------------------------
    preds, probs = train_stacked_model(X_train_f, train_labels, X_test_f)

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    auc = roc_auc_score(test_labels, probs)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, f1, auc


# =========================================================
# RUN BOTH DIRECTIONS
# =========================================================

# D1 → D2
acc_12, f1_12, auc_12 = cross_dataset_experiment(
    X1, y1,
    X2, y2,
    "D1_to_D2"
)

# D2 → D1
acc_21, f1_21, auc_21 = cross_dataset_experiment(
    X2, y2,
    X1, y1,
    "D2_to_D1"
)

# =========================================================
# SUMMARY TABLE
# =========================================================
print("\n==============================")
print("CROSS-DATASET SUMMARY")
print("==============================")

print(f"D1 → D2 | Acc: {acc_12:.4f} | F1: {f1_12:.4f} | AUC: {auc_12:.4f}")
print(f"D2 → D1 | Acc: {acc_21:.4f} | F1: {f1_21:.4f} | AUC: {auc_21:.4f}")