"""
Loss functions implemented with NumPy.
Provides forward computation and gradient backward passes.
"""

from typing import Optional
import numpy as np


class Loss:
    """Base class for loss functions."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    Categorical Cross Entropy Loss with fused Softmax.
    
    Operating directly on raw logits prevents numerical instability:
        log(softmax(z)_i) = z_i - log(sum(exp(z)))
    Gradient wrt logits:
        dL / dz = (softmax(z) - y_one_hot) / batch_size
    """

    def __init__(self, from_logits: bool = True):
        self.from_logits = from_logits
        self.probs = None
        self.y_one_hot = None
        self.batch_size = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]

        # Convert 1D integer targets to one-hot if needed
        if y_true.ndim == 1:
            self.y_one_hot = np.zeros((self.batch_size, num_classes))
            self.y_one_hot[np.arange(self.batch_size), y_true.astype(int)] = 1.0
        else:
            self.y_one_hot = y_true.astype(float)

        if self.from_logits:
            # Numerically stable log-softmax using max subtraction
            max_logits = np.max(y_pred, axis=-1, keepdims=True)
            shifted = y_pred - max_logits
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-12)
            log_probs = shifted - log_sum_exp
            self.probs = np.exp(log_probs)
            # Loss = - sum(y * log_p) / N
            loss = -np.sum(self.y_one_hot * log_probs) / self.batch_size
        else:
            eps = 1e-12
            self.probs = np.clip(y_pred, eps, 1.0 - eps)
            loss = -np.sum(self.y_one_hot * np.log(self.probs)) / self.batch_size

        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Gradient of categorical cross-entropy wrt logits:
        dL/dz = (p - y) / N
        """
        grad = (self.probs - self.y_one_hot) / self.batch_size
        return grad


class MSELoss(Loss):
    """Mean Squared Error Loss: L = 1/(2*N) * sum((y_pred - y_true)^2)"""

    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.batch_size = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_pred.shape[0]
        loss = 0.5 * np.mean((y_pred - y_true) ** 2)
        return float(loss)

    def backward(self) -> np.ndarray:
        return (self.y_pred - self.y_true) / self.y_pred.size


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross Entropy Loss with Sigmoid stability:
    L = -1/N * sum(y * log(p) + (1-y) * log(1-p))
    """

    def __init__(self, from_logits: bool = True):
        self.from_logits = from_logits
        self.probs = None
        self.y_true = None
        self.batch_size = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.batch_size = y_pred.shape[0]
        self.y_true = y_true.astype(float)
        eps = 1e-12

        if self.from_logits:
            clipped = np.clip(y_pred, -100.0, 100.0)
            self.probs = 1.0 / (1.0 + np.exp(-clipped))
            # Stable BCE from logits: max(x,0) - x*z + log(1 + exp(-abs(x)))
            loss = np.maximum(clipped, 0) - clipped * self.y_true + np.log(1.0 + np.exp(-np.abs(clipped)))
            return float(np.mean(loss))
        else:
            self.probs = np.clip(y_pred, eps, 1.0 - eps)
            loss = -(self.y_true * np.log(self.probs) + (1.0 - self.y_true) * np.log(1.0 - self.probs))
            return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        return (self.probs - self.y_true) / self.batch_size
