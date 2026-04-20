"""
train.py — Preprocessing + Training + Evaluation + Visualization
=================================================================
Run this file first:
    python train.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from model import NeuralNetwork


# ─────────────────────────────────────────────
#  STEP 1: LOAD MULTIPLE DATASETS
# ─────────────────────────────────────────────

print("=" * 55)
print("  FROM-SCRATCH NEURAL NETWORK — TRAINING")
print("=" * 55)

BASE = os.path.dirname(__file__)

# 🔥 Load all CSV files in folder
files = [f for f in os.listdir(BASE) if f.endswith(".csv")]
DATA_PATHS = [os.path.join(BASE, f) for f in files]

dfs = [pd.read_csv(path) for path in DATA_PATHS]
df = pd.concat(dfs, ignore_index=True)

print(f"\n[1] Combined Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3).to_string())


# ─────────────────────────────────────────────
#  STEP 2: PREPROCESSING
# ─────────────────────────────────────────────

print("\n[2] Preprocessing...")

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Handle missing values
missing = df.isnull().sum()
if missing.any():
    print(f"    Missing values found:\n{missing[missing>0]}")
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("    → Filled with column medians")
else:
    print("    No missing values found ✓")


# ─────────────────────────────────────────────
#  FEATURE SELECTION (FIXED FOR YOUR DATASET)
# ─────────────────────────────────────────────

FEATURES = ['highbp', 'highchol', 'bmi', 'smoker', 'physactivity']

X_raw = df[FEATURES].values.astype(float)


# ─────────────────────────────────────────────
#  NORMALIZATION (SAFE)
# ─────────────────────────────────────────────

X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)

range_ = X_max - X_min
range_[range_ == 0] = 1

X = (X_raw - X_min) / range_

np.save(os.path.join(BASE, "scaler.npy"), np.stack([X_min, X_max]))
print("    Scaler saved ✓")


# ─────────────────────────────────────────────
#  TARGETS (FIXED)
# ─────────────────────────────────────────────

y_diabetes = df['diabetes_binary'].values.reshape(-1, 1).astype(float)
y_heart    = df['heartdiseaseorattack'].values.reshape(-1, 1).astype(float)


# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

np.random.seed(42)
idx = np.random.permutation(len(X))

split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test   = X[train_idx], X[test_idx]
yd_train, yd_test = y_diabetes[train_idx], y_diabetes[test_idx]
yh_train, yh_test = y_heart[train_idx], y_heart[test_idx]

print(f"    Train: {len(X_train)} | Test: {len(X_test)}")


# ─────────────────────────────────────────────
#  STEP 3: TRAIN DIABETES MODEL
# ─────────────────────────────────────────────

print("\n[3] Training Diabetes Model...")

nn_diabetes = NeuralNetwork(input_size=5, hidden1=16, hidden2=8, learning_rate=0.05)
nn_diabetes.train(X_train, yd_train, epochs=1000, verbose=True)

d_train_acc = nn_diabetes.accuracy(X_train, yd_train)
d_test_acc  = nn_diabetes.accuracy(X_test, yd_test)

print(f"  ✓ Diabetes | Train: {d_train_acc:.2%} | Test: {d_test_acc:.2%}")

nn_diabetes.save_weights(os.path.join(BASE, "weights_diabetes.npz"))


# ─────────────────────────────────────────────
#  STEP 4: TRAIN HEART MODEL
# ─────────────────────────────────────────────

print("\n[4] Training Heart Disease Model...")

nn_heart = NeuralNetwork(input_size=5, hidden1=16, hidden2=8, learning_rate=0.05)
nn_heart.train(X_train, yh_train, epochs=1000, verbose=True)

h_train_acc = nn_heart.accuracy(X_train, yh_train)
h_test_acc  = nn_heart.accuracy(X_test, yh_test)

print(f"  ✓ Heart | Train: {h_train_acc:.2%} | Test: {h_test_acc:.2%}")

nn_heart.save_weights(os.path.join(BASE, "weights_heart.npz"))


# ─────────────────────────────────────────────
#  STEP 5: PLOTS
# ─────────────────────────────────────────────

print("\n[5] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(nn_diabetes.loss_history)
axes[0,0].set_title("Diabetes Loss")

axes[0,1].plot(nn_heart.loss_history)
axes[0,1].set_title("Heart Loss")

axes[1,0].hist(nn_diabetes.predict_proba(X_test), bins=30)
axes[1,0].set_title("Diabetes Predictions")

axes[1,1].hist(nn_heart.predict_proba(X_test), bins=30)
axes[1,1].set_title("Heart Predictions")

plt.tight_layout()
plt.savefig(os.path.join(BASE, "training_plots.png"))
plt.close()


# ─────────────────────────────────────────────
#  DONE
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print(f"  Diabetes Test Accuracy : {d_test_acc:.2%}")
print(f"  Heart    Test Accuracy : {h_test_acc:.2%}")
print("=" * 55)

print("\nNext step: streamlit run app.py")
