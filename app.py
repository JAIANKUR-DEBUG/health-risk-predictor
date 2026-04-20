import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from model import NeuralNetwork

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Health AI", page_icon="🫀", layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    scaler = np.load(os.path.join(BASE, "scaler.npy"))

    nn_d = NeuralNetwork()
    nn_d.load_weights(os.path.join(BASE, "weights_diabetes.npz"))

    nn_h = NeuralNetwork()
    nn_h.load_weights(os.path.join(BASE, "weights_heart.npz"))

    return nn_d, nn_h, scaler[0], scaler[1]

nn_diabetes, nn_heart, X_min, X_max = load_models()

# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize(x):
    range_ = X_max - X_min
    range_[range_ == 0] = 1
    return (x - X_min) / range_

# -----------------------------
# CUSTOM STYLE (COOL UI)
# -----------------------------
st.markdown("""
<style>
body { background-color: #0f172a; }

.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 15px;
}

.big-text {
    font-size: 28px;
    font-weight: bold;
}

.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 class='center'>🫀 Health Risk AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='center'>Smart Prediction using Neural Network</p>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# INPUT SECTION (CENTERED)
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    highbp = st.selectbox("High BP", ["No", "Yes"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])

with col2:
    highchol = st.selectbox("High Cholesterol", ["No", "Yes"])
    phys = st.selectbox("Physically Active", ["No", "Yes"])

with col3:
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)

# Convert values
highbp = 1 if highbp == "Yes" else 0
highchol = 1 if highchol == "Yes" else 0
smoker = 1 if smoker == "Yes" else 0
phys = 1 if phys == "Yes" else 0

st.markdown("---")

# -----------------------------
# BUTTON CENTERED
# -----------------------------
center_btn = st.columns([1,2,1])[1]
with center_btn:
    predict = st.button("🔍 Analyze Health Risk", use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
if predict:
    raw = np.array([[highbp, highchol, bmi, smoker, phys]])
    X = normalize(raw)

    d = float(nn_diabetes.predict_proba(X)[0]) * 100
    h = float(nn_heart.predict_proba(X)[0]) * 100

    # -------------------------
    # RESULT CARDS
    # -------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div>🩸 Diabetes Risk</div>
            <div class="big-text">{d:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div>❤️ Heart Risk</div>
            <div class="big-text">{h:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # CHART
    # -------------------------
    fig, ax = plt.subplots()
    ax.bar(["Diabetes", "Heart"], [d, h])
    ax.set_title("Risk Comparison")
    ax.set_ylabel("Probability %")

    st.pyplot(fig)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("<p class='center'>Built with NumPy Neural Network</p>", unsafe_allow_html=True)
