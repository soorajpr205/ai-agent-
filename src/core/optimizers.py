"""
Optimizers implemented with NumPy.
Provides parameter updates using SGD with Momentum and Adam/AdamW.
"""

from typing import List, Tuple
import numpy as np


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, params: List[Tuple[str, np.ndarray, np.ndarray]]):
        """Update model parameters using computed gradients."""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with Momentum and Weight Decay.
    
    Update rule:
        v = momentum * v + grad + weight_decay * param
        param = param - lr * v
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        self.momentum = momentum
        self.velocities = {}

    def step(self, params: List[Tuple[str, np.ndarray, np.ndarray]]):
        for i, (name, param, grad) in enumerate(params):
            if grad is None:
                continue

            if i not in self.velocities:
                self.velocities[i] = np.zeros_like(param)

            # Apply L2 weight decay
            effective_grad = grad
            if self.weight_decay > 0.0:
                effective_grad = effective_grad + self.weight_decay * param

            # Momentum update
            v = self.momentum * self.velocities[i] + effective_grad
            self.velocities[i] = v

            # Parameter update
            param -= self.learning_rate * v


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam / AdamW) optimizer.
    
    Maintains running exponential averages of gradients (first moment m)
    and squared gradients (second moment v), with bias correction.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params: List[Tuple[str, np.ndarray, np.ndarray]]):
        self.t += 1

        # Precompute bias corrections
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        for i, (name, param, grad) in enumerate(params):
            if grad is None:
                continue

            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            # Weight decay (AdamW decoupled style: directly decays weights)
            if self.weight_decay > 0.0:
                param -= self.learning_rate * self.weight_decay * param

            # Update biased 1st moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            # Update biased 2nd raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            # Parameter update
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
