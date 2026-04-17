import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Path setup

BASE = os.path.dirname(os.path.abspath(**file**))
sys.path.insert(0, BASE)

from model import NeuralNetwork

# Page config

st.set_page_config(page_title="Health Risk Predictor", page_icon="🫀", layout="wide")

# Load models

@st.cache_resource
def load_models():
scaler = np.load(os.path.join(BASE, "scaler.npy"))

```
nn_d = NeuralNetwork()
nn_d.load_weights(os.path.join(BASE, "weights_diabetes.npz"))

nn_h = NeuralNetwork()
nn_h.load_weights(os.path.join(BASE, "weights_heart.npz"))

return nn_d, nn_h, scaler[0], scaler[1]
```

nn_diabetes, nn_heart, X_min, X_max = load_models()

# Normalize

def normalize(raw):
return (raw - X_min) / (X_max - X_min + 1e-9)

# Risk label

def risk_label(pct):
if pct < 30: return "LOW"
elif pct < 70: return "MEDIUM"
else: return "HIGH"

# UI

st.title("🫀 Health Risk Predictor")
st.markdown("### Enter Patient Details")

# FORM INPUT

with st.form("patient_form"):

```
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure (mmHg)", 50, 200, 120)
    glucose = st.number_input("Glucose (mg/dL)", 50, 400, 100)

with col2:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)

submit = st.form_submit_button("🔬 Predict Risk")
```

# Prediction

if submit:
raw = np.array([[age, bp, glucose, cholesterol, heart_rate]], dtype=float)
X_in = normalize(raw)

```
d_prob = float(nn_diabetes.predict_proba(X_in)[0])
h_prob = float(nn_heart.predict_proba(X_in)[0])

d_pct = d_prob * 100
h_pct = h_prob * 100

st.subheader("Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("🩸 Diabetes Risk", f"{d_pct:.2f}%", risk_label(d_pct))

with col2:
    st.metric("❤️ Heart Disease Risk", f"{h_pct:.2f}%", risk_label(h_pct))

# Graph
fig, ax = plt.subplots()
ax.bar(["Diabetes", "Heart"], [d_pct, h_pct])
ax.set_ylabel("Risk %")
st.pyplot(fig)
```
