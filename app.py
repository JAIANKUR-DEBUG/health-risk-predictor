"""
app.py — Streamlit Web Dashboard
=================================
Interactive risk prediction dashboard for Diabetes & Heart Disease.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

# Add project folder to path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from model import NeuralNetwork


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .block-container { padding-top: 1.5rem; }

    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 { margin: 0; font-size: 2.4rem; }
    .metric-card p  { margin: 0.2rem 0 0; color: #94a3b8; font-size: 0.9rem; }

    .risk-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.4rem;
    }
    .low    { background: #14532d; color: #4ade80; }
    .medium { background: #713f12; color: #fbbf24; }
    .high   { background: #7f1d1d; color: #f87171; }

    .explain-box {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.92rem;
        color: #cbd5e1;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS (cached so they load once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load pre-trained weights into NeuralNetwork objects."""
    scaler = np.load(os.path.join(BASE, "scaler.npy"))   # shape (2, 5)

    nn_d = NeuralNetwork()
    nn_d.load_weights(os.path.join(BASE, "weights_diabetes.npz"))

    nn_h = NeuralNetwork()
    nn_h.load_weights(os.path.join(BASE, "weights_heart.npz"))

    return nn_d, nn_h, scaler[0], scaler[1]   # models + X_min + X_max

nn_diabetes, nn_heart, X_min, X_max = load_models()


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def normalize(raw: np.ndarray) -> np.ndarray:
    """Apply the same min-max scaling used during training."""
    return (raw - X_min) / (X_max - X_min + 1e-9)

def risk_label(pct: float):
    """Return (label, css_class) for a given percentage."""
    if pct < 30:   return "🟢 LOW",    "low"
    if pct < 70:   return "🟡 MEDIUM", "medium"
    return "🔴 HIGH", "high"

def gauge_color(pct):
    if pct < 30:  return "#4ade80"
    if pct < 70:  return "#fbbf24"
    return "#f87171"


# ─────────────────────────────────────────────
#  SIDEBAR INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Patient Parameters")
    st.markdown("Move the sliders to enter patient data:")
    st.markdown("---")

    age   = st.slider("🎂 Age (years)",              25, 80, 45)
    bp    = st.slider("💉 Blood Pressure (mmHg)",    60, 180, 120)
    gluc  = st.slider("🍬 Glucose / Sugar (mg/dL)", 70, 300, 100)
    chol  = st.slider("🧈 Cholesterol (mg/dL)",     120, 400, 200)
    hr    = st.slider("💓 Heart Rate (bpm)",          50, 150, 75)

    st.markdown("---")
    predict_btn = st.button("🔬 Predict Risk", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Normal Ranges:**")
    st.markdown("""
    - BP: 90–120 mmHg  
    - Glucose: 70–100 mg/dL  
    - Cholesterol: <200 mg/dL  
    - Heart Rate: 60–100 bpm
    """)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# 🫀 Health Risk Predictor")
st.markdown("**From-Scratch Neural Network** · Diabetes & Heart Disease · Built with only NumPy")
st.markdown("---")


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
raw   = np.array([[age, bp, gluc, chol, hr]], dtype=float)
X_in  = normalize(raw)

d_prob = float(nn_diabetes.predict_proba(X_in)[0])
h_prob = float(nn_heart.predict_proba(X_in)[0])
d_pct  = d_prob * 100
h_pct  = h_prob * 100

d_label, d_cls = risk_label(d_pct)
h_label, h_cls = risk_label(h_pct)


# ─────────────────────────────────────────────
#  RESULTS CARDS
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p>🩸 Diabetes Risk</p>
        <h2 style="color:{gauge_color(d_pct)}">{d_pct:.1f}%</h2>
        <span class="risk-badge {d_cls}">{d_label}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p>❤️ Heart Attack Risk</p>
        <h2 style="color:{gauge_color(h_pct)}">{h_pct:.1f}%</h2>
        <span class="risk-badge {h_cls}">{h_label}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
#  PROBABILITY BAR CHARTS
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))
fig.patch.set_facecolor('#0f172a')

def draw_bar(ax, pct, title, color):
    ax.set_facecolor('#1e293b')
    # Risk zones
    ax.barh(0, 30,  height=0.5, color='#14532d', left=0)
    ax.barh(0, 40,  height=0.5, color='#713f12', left=30)
    ax.barh(0, 30,  height=0.5, color='#7f1d1d', left=70)
    # Pointer line
    ax.axvline(pct, color=color, linewidth=3, zorder=5)
    ax.plot(pct, 0, 'o', color=color, markersize=12, zorder=6)
    # Labels
    ax.text(15, -0.45, 'LOW',    ha='center', color='#4ade80', fontsize=8, fontweight='bold')
    ax.text(50, -0.45, 'MEDIUM', ha='center', color='#fbbf24', fontsize=8, fontweight='bold')
    ax.text(85, -0.45, 'HIGH',   ha='center', color='#f87171', fontsize=8, fontweight='bold')
    ax.text(pct, 0.45, f'{pct:.1f}%', ha='center', color=color, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, 0.7)
    ax.set_title(title, color='#e2e8f0', fontsize=12, pad=8)
    ax.set_xlabel('Risk Probability (%)', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8')
    ax.set_yticks([])
    ax.spines[:].set_color('#334155')

draw_bar(ax1, d_pct, '🩸 Diabetes Risk', gauge_color(d_pct))
draw_bar(ax2, h_pct, '❤️ Heart Attack Risk', gauge_color(h_pct))

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("---")


# ─────────────────────────────────────────────
#  TRAINING HISTORY PLOTS
# ─────────────────────────────────────────────
st.markdown("### 📉 Training History")
tab1, tab2 = st.tabs(["Loss vs Epochs", "Prediction Distribution"])

with tab1:
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.patch.set_facecolor('#0f172a')
    for ax in (a1, a2):
        ax.set_facecolor('#1e293b')

    a1.plot(nn_diabetes.loss_history, color='#38bdf8', lw=2)
    a1.set_title('Diabetes — BCE Loss', color='#e2e8f0')
    a1.set_xlabel('Epoch', color='#94a3b8')
    a1.set_ylabel('Loss', color='#94a3b8')
    a1.tick_params(colors='#94a3b8')
    a1.grid(color='#334155', linestyle='--', lw=0.5)
    a1.spines[:].set_color('#334155')
    a1.annotate(f"Start: {nn_diabetes.loss_history[0]:.3f}",
                xy=(0, nn_diabetes.loss_history[0]), xytext=(80, nn_diabetes.loss_history[0]*0.95),
                color='#94a3b8', fontsize=8)
    a1.annotate(f"End: {nn_diabetes.loss_history[-1]:.3f}",
                xy=(999, nn_diabetes.loss_history[-1]),
                xytext=(700, nn_diabetes.loss_history[-1]*1.3),
                color='#38bdf8', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#38bdf8'))

    a2.plot(nn_heart.loss_history, color='#f87171', lw=2)
    a2.set_title('Heart Disease — BCE Loss', color='#e2e8f0')
    a2.set_xlabel('Epoch', color='#94a3b8')
    a2.set_ylabel('Loss', color='#94a3b8')
    a2.tick_params(colors='#94a3b8')
    a2.grid(color='#334155', linestyle='--', lw=0.5)
    a2.spines[:].set_color('#334155')
    a2.annotate(f"Start: {nn_heart.loss_history[0]:.3f}",
                xy=(0, nn_heart.loss_history[0]), xytext=(80, nn_heart.loss_history[0]*0.95),
                color='#94a3b8', fontsize=8)
    a2.annotate(f"End: {nn_heart.loss_history[-1]:.3f}",
                xy=(999, nn_heart.loss_history[-1]),
                xytext=(700, nn_heart.loss_history[-1]*1.3),
                color='#f87171', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#f87171'))

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
    st.caption("Loss decreases over epochs — the network is learning! Lower loss = better predictions.")

with tab2:
    img_path = os.path.join(BASE, "training_plots.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Prediction distributions from the test set", use_container_width=True)
    else:
        st.info("Run `python train.py` first to generate the distribution plots.")


# ─────────────────────────────────────────────
#  VIVA EXPLANATION SECTION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📚 How It Works — Viva Explanation")

e1, e2, e3 = st.columns(3)

with e1:
    st.markdown("""
    <div class="explain-box">
    <div class="section-title">🔵 Forward Propagation</div>
    Data travels <b>input → hidden layers → output</b>.
    At each layer:<br>
    <code>z = X·W + b</code> (linear)<br>
    <code>a = activation(z)</code> (non-linear)<br><br>
    Like signals passing through neurons in a brain.
    The final neuron outputs a probability (0–1).
    </div>
    """, unsafe_allow_html=True)

with e2:
    st.markdown("""
    <div class="explain-box">
    <div class="section-title">🔴 Backpropagation</div>
    After forward pass, we compute the <b>error</b> (loss).
    We then flow this error <b>backwards</b> through the network
    using the chain rule to find how much each weight
    contributed to the mistake.<br><br>
    Think of it as finding who is "to blame" for
    a wrong prediction.
    </div>
    """, unsafe_allow_html=True)

with e3:
    st.markdown("""
    <div class="explain-box">
    <div class="section-title">🟢 Gradient Descent</div>
    Using the gradients from backprop, we <b>update weights</b>:<br>
    <code>W = W − lr × gradient</code><br><br>
    The <b>learning rate</b> (e.g. 0.05) controls the step size.
    Too large → overshoot. Too small → slow convergence.
    After many steps, loss reaches a minimum = trained model!
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="explain-box" style="margin-top:0.5rem">
<div class="section-title">🏗️ Neural Network Architecture</div>
<b>Input Layer (5 neurons)</b>: Age, Blood Pressure, Glucose, Cholesterol, Heart Rate<br>
→ <b>Hidden Layer 1 (16 neurons, ReLU)</b>: Learns low-level patterns<br>
→ <b>Hidden Layer 2 (8 neurons, ReLU)</b>: Learns higher-level combinations<br>
→ <b>Output Layer (1 neuron, Sigmoid)</b>: Outputs probability 0–1
<br><br>
<b>Loss Function</b>: Binary Cross-Entropy — measures how wrong our probability prediction is.<br>
<b>Activation</b>: ReLU in hidden layers (kills negatives, faster training); Sigmoid in output (squashes to 0–1 for probability).<br>
<b>Weight Init</b>: He initialization — scales weights by √(2/n) to prevent vanishing/exploding gradients.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<center style="color:#475569; font-size:0.85rem">
Built from scratch with NumPy only · No sklearn, no Keras, no TensorFlow · 
Diabetes Accuracy: ~97% · Heart Accuracy: ~80%
</center>
""", unsafe_allow_html=True)
