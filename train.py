import numpy as np
import pandas as pd
import os
from model import NeuralNetwork

# Load dataset

df = pd.read_csv("dataset.csv")

# Rename columns if needed (VERY IMPORTANT)

df.columns = [c.lower() for c in df.columns]

# Adjust based on your dataset columns

X = df[['age','blood_pressure','glucose','cholesterol','heart_rate']].values
y_diabetes = df['diabetes'].values.reshape(-1,1)
y_heart = df['heart_disease'].values.reshape(-1,1)

# Normalize

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-9)

np.save("scaler.npy", np.array([X_min, X_max]))

# Split

split = int(0.8 * len(X))
X_train, X_test = X_norm[:split], X_norm[split:]
yd_train, yd_test = y_diabetes[:split], y_diabetes[split:]
yh_train, yh_test = y_heart[:split], y_heart[split:]

# Train models

nn_d = NeuralNetwork()
nn_d.train(X_train, yd_train, epochs=500)
nn_d.save_weights("weights_diabetes.npz")

nn_h = NeuralNetwork()
nn_h.train(X_train, yh_train, epochs=500)
nn_h.save_weights("weights_heart.npz")

print("Training complete!")
