# 🫀 From-Scratch Neural Network — Diabetes & Heart Attack Risk Predictor

## Project Overview
A **fully hand-coded Neural Network** using only NumPy that predicts:
- 🩸 **Diabetes risk**
- ❤️ **Heart attack / Heart disease risk**

No sklearn, no Keras, no TensorFlow — everything is built manually.

---

## File Structure
```
ml_project/
├── model.py          # Neural Network from scratch (weights, forward, backprop, GD)
├── train.py          # Data preprocessing + training loop + visualization
├── app.py            # Streamlit web dashboard
├── dataset.csv       # Generated realistic dataset (1000 samples)
├── scaler.npy        # Min-max scaler values (saved after training)
├── weights_diabetes.npz  # Trained weights for diabetes model
├── weights_heart.npz     # Trained weights for heart disease model
├── training_plots.png    # Generated training graphs
└── README.md
```

---

## How to Run

### Step 1 — Install Dependencies
```bash
pip install numpy pandas matplotlib streamlit
```

### Step 2 — Train the Models
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train two neural networks (diabetes + heart)
- Save trained weights to `.npz` files
- Save training plots to `training_plots.png`

Expected output:
```
Epoch 1000 | Loss: 0.1099 | Accuracy: 96.75%
✓ Diabetes Model  |  Train Acc: 96.75%  |  Test Acc: 97.50%
✓ Heart Model     |  Train Acc: 86.38%  |  Test Acc: 80.00%
```

### Step 3 — Launch the Web App
```bash
streamlit run app.py
```
Then open your browser at: **http://localhost:8501**

---

## Neural Network Architecture
```
Input(5)  →  Hidden1(16, ReLU)  →  Hidden2(8, ReLU)  →  Output(1, Sigmoid)
   Age           Pattern               Combinations          Probability
   BP            Learning              Learning               0.0 – 1.0
   Glucose
   Cholesterol
   Heart Rate
```

---

## Risk Levels
| Probability | Risk Level |
|-------------|-----------|
| 0% – 30%   | 🟢 LOW    |
| 30% – 70%  | 🟡 MEDIUM |
| 70% – 100% | 🔴 HIGH   |

---

## Key Concepts (Viva Ready)

### Forward Propagation
Data flows input → hidden layers → output.  
Each layer: `z = X·W + b`, then `a = activation(z)`  
The final output is a probability between 0 and 1.

### Backpropagation
After forward pass, we compute the error (Binary Cross-Entropy loss).  
We then flow the error backwards using the **chain rule** to find gradients —  
how much each weight contributed to the mistake.

### Gradient Descent
Weights are updated: `W = W - learning_rate × gradient`  
- **Learning rate** (0.05): controls step size
- Too large → oscillates, doesn't converge
- Too small → trains very slowly
- Just right → smooth descent to minimum loss

### Activation Functions
- **ReLU** `max(0, z)` in hidden layers: faster training, no vanishing gradient
- **Sigmoid** `1/(1+e^-z)` in output: maps any value to (0, 1) — perfect for probability

### Loss Function
**Binary Cross-Entropy**: `BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`  
Measures how wrong our probability prediction is. Goal: minimize this over epochs.

---

## Results
| Model | Train Accuracy | Test Accuracy |
|-------|---------------|--------------|
| Diabetes | ~97% | ~97% |
| Heart Disease | ~86% | ~80% |
