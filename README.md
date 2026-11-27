# Deep Neural Network from Scratch in Pure NumPy

A modular, production-grade Deep Learning framework built entirely from scratch in **pure Python and NumPy**, without PyTorch, TensorFlow, or Scikit-Learn dependencies.

Included is a complete optical digit recognition model ($0 - 9$) achieving **99.5% accuracy** on unseen test data in less than 5 seconds of training.

---

## Architecture Overview

```
                        Input Vector (64 features, 8x8 image)
                                        │
                                        ▼
                           Dense (64 -> 128) [He Normal]
                                        │
                                        ▼
                                 BatchNorm1d(128)
                                        │
                                        ▼
                                      ReLU()
                                        │
                                        ▼
                                  Dropout(p=0.15)
                                        │
                                        ▼
                           Dense (128 -> 64) [He Normal]
                                        │
                                        ▼
                                 BatchNorm1d(64)
                                        │
                                        ▼
                                      ReLU()
                                        │
                                        ▼
                                  Dropout(p=0.10)
                                        │
                                        ▼
                           Dense (64 -> 10) [He Normal]
                                        │
                                        ▼
                     Categorical Cross-Entropy (Fused Logits)
                                        │
                                        ▼
                         Predicted Digit Probability (0-9)
```

---

## Core Components Implemented

### 1. Layers (`src/core/layers.py`)
- **`Dense`**: Fully connected linear transformation $Y = XW + b$ with He (Kaiming) & Xavier (Glorot) weight initializations, analytical gradient backpropagation for inputs, weights, and biases.
- **`BatchNorm1d`**: 1D Batch Normalization with learnable scale $\gamma$, shift $\beta$, running mean/variance tracking for inference, and the analytical backward gradient pass.
- **`Dropout`**: Inverted Dropout for regularization during training with automatic bypass during inference.

### 2. Activation Functions (`src/core/activations.py`)
- **`ReLU`**: Rectified Linear Unit ($f(x) = \max(0, x)$).
- **`LeakyReLU`**: Leaky ReLU with custom $\alpha$ slope for negative inputs.
- **`Sigmoid`**: Numerically clamped Sigmoid activation ($f(x) = \frac{1}{1 + e^{-x}}$).
- **`Tanh`**: Hyperbolic tangent activation.
- **`Softmax`**: Numerically stable Softmax with coordinate shift ($\max(z)$ subtraction).

### 3. Loss Functions (`src/core/losses.py`)
- **`CrossEntropyLoss`**: Categorical cross-entropy with fused log-softmax for numerical stability, computing $\frac{\partial \mathcal{L}}{\partial z} = \frac{p - y}{N}$.
- **`MSELoss`**: Mean Squared Error regression loss.
- **`BinaryCrossEntropyLoss`**: Binary classification loss with stable log-sum-exp formulation.

### 4. Optimizers (`src/core/optimizers.py`)
- **`Adam`**: Adaptive Moment Estimation with bias correction for 1st moment ($m_t$), 2nd raw moment ($v_t$), and decoupled weight decay (AdamW).
- **`SGD`**: Stochastic Gradient Descent with Nesterov/Polyak momentum and $L_2$ weight decay.

### 5. High-Level Model Container (`src/core/model.py`)
- **`Sequential`**: Chainable neural network container with `.compile()`, `.fit()`, `.evaluate()`, `.predict()`, `.summary()`, `.save_weights()`, and `.load_weights()`.

---

## Directory Structure

```
neural-network-from-scratch/
├── src/
│   ├── core/
│   │   ├── __init__.py        # Core exports
│   │   ├── layers.py          # Dense, BatchNorm1d, Dropout
│   │   ├── activations.py     # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│   │   ├── losses.py          # CrossEntropyLoss, MSELoss, BinaryCrossEntropyLoss
│   │   ├── optimizers.py      # Adam, SGD
│   │   └── model.py           # Sequential model container
│   └── data/
│       ├── __init__.py        # Data utilities exports
│       └── datasets.py        # 8x8 digit dataset generator, ASCII visualizer, train/test split
├── tests/
│   └── test_model.py          # Unit tests & numerical gradient checks
├── weights/
│   └── digit_classifier.npz   # Trained model weights & batchnorm buffers
├── train.py                   # Model training and validation script
├── infer.py                   # Inference script with ASCII digit rendering
├── interactive_predict.py     # Demonstration script on custom digits
└── README.md                  # Project documentation
```

---

## Quickstart

### 1. Run Unit Tests (Numerical Gradient Checks & Layer Integrity)
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### 2. Train the Model
Trains a 3-layer deep neural network with Batch Normalization and Adam optimizer:
```powershell
python train.py
```
**Sample Output:**
```
Training Neural Network on 2800 samples | Batch size: 64 | Epochs: 20
Epoch 01/20 [0.20s] - loss: 0.8395 | acc: 75.18% | val_loss: 0.2084 | val_acc: 97.00%
...
Epoch 20/20 [0.19s] - loss: 0.0111 | acc: 99.68% | val_loss: 0.0123 | val_acc: 99.33%
Evaluation -> Loss: 0.0085 | Accuracy: 99.50%
```

### 3. Run Inference on Unseen Test Digits
Runs forward pass, prints ASCII visualization and class probability distribution:
```powershell
python infer.py --samples 3
```

### 4. Test Handcrafted Digits
```powershell
python interactive_predict.py
```
