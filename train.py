"""
train.py — Preprocessing + Training + Evaluation + Visualization
=================================================================
Run this file first to train the models and save weights + plots.

Usage:
    python train.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import os

from model import NeuralNetwork


# ─────────────────────────────────────────────
#  STEP 1: LOAD DATASET
# ─────────────────────────────────────────────

print("=" * 55)
print("  FROM-SCRATCH NEURAL NETWORK — TRAINING")
print("=" * 55)

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3).to_string())


# ─────────────────────────────────────────────
#  STEP 2: PREPROCESSING
# ─────────────────────────────────────────────

print("\n[2] Preprocessing...")

# 2a. Check & handle missing values
missing = df.isnull().sum()
if missing.any():
    print(f"    Missing values found:\n{missing[missing>0]}")
    df.fillna(df.median(), inplace=True)  # fill with column median
    print("    → Filled with column medians")
else:
    print("    No missing values found ✓")

# 2b. Feature columns
FEATURES = ['age', 'blood_pressure', 'glucose', 'cholesterol', 'heart_rate']
X_raw = df[FEATURES].values.astype(float)

# 2c. Min-Max Normalization: scale each feature to [0, 1]
#     Formula: X_norm = (X - X_min) / (X_max - X_min)
#     Why? Neural networks train much faster on small uniform numbers.
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min + 1e-9)

# Save scaler stats so the app can normalize user inputs
np.save(os.path.join(os.path.dirname(__file__), "scaler.npy"),
        np.stack([X_min, X_max]))
print(f"    Feature ranges saved (scaler.npy)")

# 2d. Targets
y_diabetes = df['diabetes'].values.reshape(-1, 1).astype(float)
y_heart    = df['heart_disease'].values.reshape(-1, 1).astype(float)

# 2e. Train / Test split (80 / 20) — done manually, no sklearn
np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test     = X[train_idx], X[test_idx]
yd_train, yd_test   = y_diabetes[train_idx], y_diabetes[test_idx]
yh_train, yh_test   = y_heart[train_idx],    y_heart[test_idx]

print(f"    Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"    Diabetes prevalence  (train): {yd_train.mean():.1%}")
print(f"    Heart disease prev.  (train): {yh_train.mean():.1%}")


# ─────────────────────────────────────────────
#  STEP 3: TRAIN DIABETES MODEL
# ─────────────────────────────────────────────

print("\n[3] Training Diabetes Model (1000 epochs)...")
nn_diabetes = NeuralNetwork(input_size=5, hidden1=16, hidden2=8, learning_rate=0.05)
nn_diabetes.train(X_train, yd_train, epochs=1000, verbose=True)

d_train_acc = nn_diabetes.accuracy(X_train, yd_train)
d_test_acc  = nn_diabetes.accuracy(X_test,  yd_test)
print(f"\n  ✓ Diabetes Model  |  Train Acc: {d_train_acc:.2%}  |  Test Acc: {d_test_acc:.2%}")
nn_diabetes.save_weights(os.path.join(os.path.dirname(__file__), "weights_diabetes.npz"))


# ─────────────────────────────────────────────
#  STEP 4: TRAIN HEART DISEASE MODEL
# ─────────────────────────────────────────────

print("\n[4] Training Heart Disease Model (1000 epochs)...")
nn_heart = NeuralNetwork(input_size=5, hidden1=16, hidden2=8, learning_rate=0.05, seed=99)
nn_heart.train(X_train, yh_train, epochs=1000, verbose=True)

h_train_acc = nn_heart.accuracy(X_train, yh_train)
h_test_acc  = nn_heart.accuracy(X_test,  yh_test)
print(f"\n  ✓ Heart Model     |  Train Acc: {h_train_acc:.2%}  |  Test Acc: {h_test_acc:.2%}")
nn_heart.save_weights(os.path.join(os.path.dirname(__file__), "weights_heart.npz"))


# ─────────────────────────────────────────────
#  STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────────

print("\n[5] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('#0f172a')
for ax in axes.flat:
    ax.set_facecolor('#1e293b')

C_DIAB  = '#38bdf8'   # sky blue
C_HEART = '#f87171'  # rose red
C_TEXT  = '#e2e8f0'
C_GRID  = '#334155'

# ── Plot 1: Loss vs Epochs — Diabetes ──
ax = axes[0, 0]
epochs_range = range(1, len(nn_diabetes.loss_history) + 1)
ax.plot(epochs_range, nn_diabetes.loss_history, color=C_DIAB, linewidth=2)
ax.set_title('Diabetes Model — Loss vs Epochs', color=C_TEXT, fontsize=12, pad=10)
ax.set_xlabel('Epoch', color=C_TEXT)
ax.set_ylabel('Binary Cross-Entropy Loss', color=C_TEXT)
ax.tick_params(colors=C_TEXT)
ax.grid(color=C_GRID, linestyle='--', linewidth=0.5)
ax.spines[:].set_color(C_GRID)
ax.annotate(f'Final: {nn_diabetes.loss_history[-1]:.4f}',
            xy=(1000, nn_diabetes.loss_history[-1]),
            xytext=(700, nn_diabetes.loss_history[0]*0.7),
            color=C_DIAB, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_DIAB))

# ── Plot 2: Loss vs Epochs — Heart ──
ax = axes[0, 1]
epochs_range = range(1, len(nn_heart.loss_history) + 1)
ax.plot(epochs_range, nn_heart.loss_history, color=C_HEART, linewidth=2)
ax.set_title('Heart Disease Model — Loss vs Epochs', color=C_TEXT, fontsize=12, pad=10)
ax.set_xlabel('Epoch', color=C_TEXT)
ax.set_ylabel('Binary Cross-Entropy Loss', color=C_TEXT)
ax.tick_params(colors=C_TEXT)
ax.grid(color=C_GRID, linestyle='--', linewidth=0.5)
ax.spines[:].set_color(C_GRID)
ax.annotate(f'Final: {nn_heart.loss_history[-1]:.4f}',
            xy=(1000, nn_heart.loss_history[-1]),
            xytext=(700, nn_heart.loss_history[0]*0.7),
            color=C_HEART, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_HEART))

# ── Plot 3: Prediction Distribution — Diabetes ──
ax = axes[1, 0]
d_probs = nn_diabetes.predict_proba(X_test)
ax.hist(d_probs[yd_test.flatten()==0], bins=30, alpha=0.7, color='#22c55e', label='No Diabetes')
ax.hist(d_probs[yd_test.flatten()==1], bins=30, alpha=0.7, color=C_DIAB,   label='Diabetes')
ax.axvline(0.5, color='white', linestyle='--', linewidth=1.5, label='Threshold 0.5')
ax.set_title('Diabetes — Prediction Distribution', color=C_TEXT, fontsize=12, pad=10)
ax.set_xlabel('Predicted Probability', color=C_TEXT)
ax.set_ylabel('Count', color=C_TEXT)
ax.tick_params(colors=C_TEXT)
ax.legend(facecolor='#0f172a', labelcolor=C_TEXT, fontsize=9)
ax.grid(color=C_GRID, linestyle='--', linewidth=0.5)
ax.spines[:].set_color(C_GRID)

# ── Plot 4: Prediction Distribution — Heart ──
ax = axes[1, 1]
h_probs = nn_heart.predict_proba(X_test)
ax.hist(h_probs[yh_test.flatten()==0], bins=30, alpha=0.7, color='#a78bfa', label='No Heart Disease')
ax.hist(h_probs[yh_test.flatten()==1], bins=30, alpha=0.7, color=C_HEART,  label='Heart Disease')
ax.axvline(0.5, color='white', linestyle='--', linewidth=1.5, label='Threshold 0.5')
ax.set_title('Heart Disease — Prediction Distribution', color=C_TEXT, fontsize=12, pad=10)
ax.set_xlabel('Predicted Probability', color=C_TEXT)
ax.set_ylabel('Count', color=C_TEXT)
ax.tick_params(colors=C_TEXT)
ax.legend(facecolor='#0f172a', labelcolor=C_TEXT, fontsize=9)
ax.grid(color=C_GRID, linestyle='--', linewidth=0.5)
ax.spines[:].set_color(C_GRID)

plt.suptitle('From-Scratch Neural Network — Training Results',
             color=C_TEXT, fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "training_plots.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
print(f"  Plots saved → training_plots.png")
plt.close()

print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print(f"  Diabetes Test Accuracy : {d_test_acc:.2%}")
print(f"  Heart    Test Accuracy : {h_test_acc:.2%}")
print("=" * 55)
print("\nNext step: streamlit run app.py")
