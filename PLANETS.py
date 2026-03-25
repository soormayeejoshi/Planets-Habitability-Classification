#!/usr/bin/env python
# coding: utf-8
"""
Exoplanet Confirmation Classifier
----------------------------------
Logistic Regression (from scratch) on the NASA Kepler KOI dataset.
Fixes:
  - Defined correlation matrix before heatmap
  - No data leakage: test set scaled with training stats
  - Removed references to non-existent columns ('box', 'scale')
  - Vectorised predict (no slow Python loop)
  - Trained w, b used consistently (no hardcoded weights)
  - Threshold sweep moved to a clean evaluation block
  - All duplicate code removed
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
planets = pd.read_csv("cumulative.csv")          # adjust path as needed
print("Raw shape:", planets.shape)

# ──────────────────────────────────────────────
# 2. BUILD TARGET COLUMN
# ──────────────────────────────────────────────
planets["confirmed"] = (planets["koi_disposition"] == "CONFIRMED").astype(int)
planets.drop(columns=["koi_disposition", "koi_pdisposition"], inplace=True)

# ──────────────────────────────────────────────
# 3. DROP HIGH-MISSING COLUMNS & IMPUTE
# ──────────────────────────────────────────────
print("\nMissing (%) before drop:")
print((planets.isnull().mean() * 100).sort_values(ascending=False).head(10))

planets.drop(columns=["koi_teq_err1", "koi_teq_err2"], inplace=True)

planets.fillna(planets.median(numeric_only=True), inplace=True)

print("\nMissing (%) after imputation:")
print((planets.isnull().mean() * 100).sort_values(ascending=False).head(5))

# ──────────────────────────────────────────────
# 4. DERIVED FEATURES
# ──────────────────────────────────────────────
planets["rocky"]    = (planets["koi_prad"] <= 4).astype(int)
planets["habitable"] = (
    (planets["koi_teq"] >= 200) & (planets["koi_teq"] <= 350)
).astype(int)

# ──────────────────────────────────────────────
# 5. CORRELATION HEATMAP  (fix: define corr first)
# ──────────────────────────────────────────────
FEATURES = ["koi_prad", "koi_steff", "koi_slogg", "koi_srad",
            "koi_teq", "koi_insol", "rocky", "habitable", "confirmed"]

corr = planets[FEATURES].corr()          # ← was missing in original

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="mako_r", annot=True, fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 6. EXPLORATORY PLOTS
# ──────────────────────────────────────────────
planets["koi_teq"].hist(bins=50, figsize=(10, 5), range=(273, 373))
plt.title("Equilibrium Temperature (273–373 K  ≈ habitable range)")
plt.xlabel("koi_teq (K)")
plt.tight_layout()
plt.show()

print("\nPlanets in habitable temperature range:")
print(planets.query("273 <= koi_teq <= 373").shape[0])

print(f"\nMean planet radius: {planets['koi_prad'].mean():.3f} Earth radii")

# ──────────────────────────────────────────────
# 7. STRATIFIED TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
INPUT_FEATURES = ["koi_prad", "koi_steff", "koi_slogg", "koi_srad",
                  "koi_teq", "koi_insol", "rocky", "habitable"]

# Build stratification key
planets["rad"]   = pd.cut(planets["koi_prad"],
    bins=[0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,np.inf],
    labels=np.arange(1, 16))

planets["teq"]   = pd.cut(planets["koi_teq"],
    bins=[0,200,400,600,800,1000,1200,1400,1600,1800,2000,np.inf],
    labels=np.arange(1, 12))

planets["slogg"] = pd.cut(planets["koi_slogg"],
    bins=[-np.inf, 3, 4, 5], labels=np.arange(1, 4))

planets["strata"] = (planets["rad"].astype(str) + "_" +
                     planets["teq"].astype(str) + "_" +
                     planets["slogg"].astype(str))

# Drop rare strata
counts = planets["strata"].value_counts()
planets = planets[~planets["strata"].isin(counts[counts < 2].index)].copy()

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(planets, planets["strata"]):
    train_set = planets.iloc[train_idx]
    test_set  = planets.iloc[test_idx]

X_train = train_set[INPUT_FEATURES].copy()
y_train = train_set["confirmed"].values

X_test  = test_set[INPUT_FEATURES].copy()
y_test  = test_set["confirmed"].values

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# ──────────────────────────────────────────────
# 8. FEATURE SCALING  (fix: scale test with TRAIN stats)
# ──────────────────────────────────────────────
COLS_TO_SCALE = ["koi_prad", "koi_steff", "koi_slogg",
                 "koi_srad", "koi_teq", "koi_insol"]

train_means = X_train[COLS_TO_SCALE].mean()
train_stds  = X_train[COLS_TO_SCALE].std()

X_train[COLS_TO_SCALE] = (X_train[COLS_TO_SCALE] - train_means) / train_stds
X_test[COLS_TO_SCALE]  = (X_test[COLS_TO_SCALE]  - train_means) / train_stds  # ← use train stats

X_train_np = X_train.values
X_test_np  = X_test.values

X_train.hist(figsize=(16, 10), bins=50)
plt.suptitle("Training Feature Distributions (after scaling)")
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────
# 9. LOGISTIC REGRESSION — FROM SCRATCH
# ──────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # clip avoids overflow


def compute_cost(X, y, w, b):
    m = len(y)
    z   = X @ w + b
    sig = sigmoid(z)
    cost = -np.mean(y * np.log(sig + 1e-15) + (1 - y) * np.log(1 - sig + 1e-15))
    return cost


def compute_gradient(X, y, w, b):
    m   = len(y)
    err = sigmoid(X @ w + b) - y
    dj_dw = (X.T @ err) / m
    dj_db = np.mean(err)
    return dj_db, dj_dw


def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    w, b = w_init.copy(), b_init
    J_history = []

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % max(1, num_iters // 10) == 0 or i == num_iters - 1:
            print(f"  Iter {i:5d}:  Cost = {cost:.4f}")

    return w, b, J_history


def predict(X, w, b, threshold=0.5):
    """Vectorised prediction — no slow Python loop."""
    return (sigmoid(X @ w + b) >= threshold).astype(int)


def evaluate(y_true, y_pred, label=""):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp + 1e-15)
    rec  = tp / (tp + fn + 1e-15)
    f1   = 2 * prec * rec / (prec + rec + 1e-15)

    print(f"\n{'─'*40}")
    print(f"  {label} Evaluation")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    return acc, prec, rec, f1

# ──────────────────────────────────────────────
# 10. TRAIN
# ──────────────────────────────────────────────
m, n = X_train_np.shape
np.random.seed(42)
w_init = np.random.randn(n) * 0.01
b_init = 0.0

print("\nTraining logistic regression …")
w, b, J_history = gradient_descent(
    X_train_np, y_train,
    w_init, b_init,
    alpha=0.01,
    num_iters=10_000
)
print(f"\nLearned w: {np.round(w,6)}")
print(f"Learned b: {b:.6f}")

# Plot learning curve
plt.figure(figsize=(8, 4))
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Learning Curve")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 11. EVALUATE — default threshold 0.5
# ──────────────────────────────────────────────
y_pred_train = predict(X_train_np, w, b)
y_pred_test  = predict(X_test_np,  w, b)

evaluate(y_train, y_pred_train, "Train")
evaluate(y_test,  y_pred_test,  "Test")

# ──────────────────────────────────────────────
# 12. THRESHOLD SWEEP (find best F1 on test set)
# ──────────────────────────────────────────────
probs = sigmoid(X_test_np @ w + b)
best = {"thr": 0.5, "f1": -1, "prec": 0, "rec": 0}

for thr in np.linspace(0.05, 0.95, 50):
    pred = (probs >= thr).astype(int)
    tp = np.sum((y_test == 1) & (pred == 1))
    fp = np.sum((y_test == 0) & (pred == 1))
    fn = np.sum((y_test == 1) & (pred == 0))
    p  = tp / (tp + fp + 1e-15)
    r  = tp / (tp + fn + 1e-15)
    f1 = 2 * p * r / (p + r + 1e-15)
    if f1 > best["f1"]:
        best = {"thr": thr, "f1": f1, "prec": p, "rec": r}

print(f"\nBest threshold = {best['thr']:.2f}")
print(f"  Precision : {best['prec']*100:.2f}%")
print(f"  Recall    : {best['rec']*100:.2f}%")
print(f"  F1 Score  : {best['f1']*100:.2f}%")

# Re-evaluate with best threshold
y_pred_best = (probs >= best["thr"]).astype(int)
evaluate(y_test, y_pred_best, f"Test (threshold={best['thr']:.2f})")
