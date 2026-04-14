"""
model.py — Neural Network from Scratch using only NumPy
=========================================================
No sklearn, no Keras, no TensorFlow. Pure math + NumPy.

Architecture:  Input(5) → Hidden1(16, ReLU) → Hidden2(8, ReLU) → Output(1, Sigmoid)
"""

import numpy as np


# ─────────────────────────────────────────────
#  ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

def relu(z):
    """ReLU: max(0, z) — kills negative values, keeps positives."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU: 1 if z>0 else 0 (used in backprop)."""
    return (z > 0).astype(float)

def sigmoid(z):
    """Sigmoid: squashes any value into (0,1) — perfect for probability output."""
    # Clip to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid: a * (1 - a)  where a = sigmoid(z)."""
    return a * (1.0 - a)


# ─────────────────────────────────────────────
#  LOSS FUNCTION
# ─────────────────────────────────────────────

def binary_cross_entropy(y_true, y_pred):
    """
    BCE Loss = -[ y*log(ŷ) + (1-y)*log(1-ŷ) ]
    Measures how wrong our predictions are.
    Lower loss = better model.
    """
    eps = 1e-9  # tiny value to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ─────────────────────────────────────────────
#  NEURAL NETWORK CLASS
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    A 3-layer fully connected Neural Network built from scratch.

    Layers:
        Input  →  Hidden1(16 neurons, ReLU)  →  Hidden2(8 neurons, ReLU)  →  Output(1, Sigmoid)

    What happens during training:
        1. FORWARD PASS  : data flows input→output, producing a prediction
        2. LOSS          : we measure how wrong the prediction is
        3. BACKPROP      : we calculate how much each weight contributed to the error
        4. GRADIENT DESC : we nudge each weight in the direction that reduces error
    """

    def __init__(self, input_size=5, hidden1=16, hidden2=8, learning_rate=0.01, seed=42):
        """
        Initialize weights and biases.

        Weights are initialized with 'He initialization' (scaled by sqrt(2/n))
        which works well with ReLU activations — prevents vanishing/exploding gradients.
        """
        np.random.seed(seed)
        self.lr = learning_rate  # How big a step we take when updating weights

        # ── Layer 1: Input(5) → Hidden1(16) ──
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1))  # biases start at 0

        # ── Layer 2: Hidden1(16) → Hidden2(8) ──
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        # ── Layer 3: Hidden2(8) → Output(1) ──
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, 1))

        # Store training history for plotting
        self.loss_history = []

    # ─────────────────────────────────────────
    #  FORWARD PROPAGATION
    # ─────────────────────────────────────────
    def forward(self, X):
        """
        FORWARD PROPAGATION:
        Data flows from input layer → hidden layers → output layer.
        Each layer: z = X·W + b  (linear transformation)
                    a = activation(z)  (non-linear squash)

        Think of it like passing a signal through a network of neurons.
        """
        # Layer 1: linear + ReLU
        self.Z1 = X @ self.W1 + self.b1       # (n, 5) · (5, 16) = (n, 16)
        self.A1 = relu(self.Z1)                # Apply ReLU activation

        # Layer 2: linear + ReLU
        self.Z2 = self.A1 @ self.W2 + self.b2  # (n, 16) · (16, 8) = (n, 8)
        self.A2 = relu(self.Z2)                # Apply ReLU activation

        # Output layer: linear + Sigmoid (to get probability 0–1)
        self.Z3 = self.A2 @ self.W3 + self.b3  # (n, 8) · (8, 1) = (n, 1)
        self.A3 = sigmoid(self.Z3)             # Final probability

        return self.A3

    # ─────────────────────────────────────────
    #  BACKPROPAGATION
    # ─────────────────────────────────────────
    def backward(self, X, y):
        """
        BACKPROPAGATION:
        We calculate how much each weight contributed to the error
        by flowing the error signal BACKWARDS through the network.

        Uses the chain rule of calculus:
            dLoss/dW = dLoss/dA · dA/dZ · dZ/dW

        Think of it like finding who's to blame when a team loses — you trace
        each player's contribution to the final result.
        """
        n = X.shape[0]  # number of training samples

        # ── Output layer error ──
        # dLoss/dZ3 = (prediction - true_label) / n
        dZ3 = (self.A3 - y) / n                    # (n, 1)
        dW3 = self.A2.T @ dZ3                      # (8, n) · (n, 1) = (8, 1)
        db3 = np.sum(dZ3, axis=0, keepdims=True)   # (1, 1)

        # ── Layer 2 error ──
        dA2 = dZ3 @ self.W3.T                      # (n, 1) · (1, 8) = (n, 8)
        dZ2 = dA2 * relu_derivative(self.Z2)        # Apply ReLU derivative
        dW2 = self.A1.T @ dZ2                      # (16, n) · (n, 8) = (16, 8)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # ── Layer 1 error ──
        dA1 = dZ2 @ self.W2.T                      # (n, 8) · (8, 16) = (n, 16)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T @ dZ1                            # (5, n) · (n, 16) = (5, 16)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    # ─────────────────────────────────────────
    #  GRADIENT DESCENT UPDATE
    # ─────────────────────────────────────────
    def update_weights(self, dW1, db1, dW2, db2, dW3, db3):
        """
        GRADIENT DESCENT:
        Update each weight by moving it a small step in the direction
        that reduces the loss.

        Rule:  W = W - learning_rate × gradient
               (gradient tells us which direction is 'uphill',
                so we subtract to go 'downhill' toward minimum loss)

        learning_rate (e.g. 0.01) controls step size:
            Too large  → overshoot the minimum, loss oscillates
            Too small  → takes forever to converge
        """
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    # ─────────────────────────────────────────
    #  TRAINING LOOP
    # ─────────────────────────────────────────
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Full training loop:
          For each epoch:
            1. Forward pass  → get predictions
            2. Compute loss  → measure error
            3. Backprop      → compute gradients
            4. Update weights → reduce error
        """
        for epoch in range(1, epochs + 1):
            # Step 1: Forward propagation
            predictions = self.forward(X)

            # Step 2: Compute loss
            loss = binary_cross_entropy(y, predictions)
            self.loss_history.append(loss)

            # Step 3: Backpropagation
            grads = self.backward(X, y)

            # Step 4: Gradient descent weight update
            self.update_weights(*grads)

            # Print progress every 100 epochs
            if verbose and epoch % 100 == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    def predict_proba(self, X):
        """Return probability (0–1) for each sample."""
        return self.forward(X).flatten()

    def accuracy(self, X, y):
        """Fraction of correct predictions (threshold = 0.5)."""
        preds = (self.predict_proba(X) >= 0.5).astype(int)
        return np.mean(preds == y.flatten())

    def save_weights(self, path):
        """Save all weights/biases to a .npz file."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3,
                 loss_history=np.array(self.loss_history))
        print(f"  Weights saved → {path}")

    def load_weights(self, path):
        """Load weights from a saved .npz file."""
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']
        self.loss_history = list(data['loss_history'])
