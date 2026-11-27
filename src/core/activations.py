"""
Activation functions implemented with NumPy.
Provides forward and backward gradient passes for:
- ReLU
- LeakyReLU
- Sigmoid
- Tanh
- Softmax
"""
# TODO: Add GELU activation

import numpy as np
from src.core.layers import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    df/dx = 1 if x > 0 else 0
    """

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0.0, inputs)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.inputs > 0.0)


class LeakyReLU(Layer):
    """
    Leaky Rectified Linear Unit activation function.
    f(x) = x if x > 0 else alpha * x
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.inputs = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.inputs = inputs
        return np.where(inputs > 0.0, inputs, inputs * self.alpha)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        dx = np.where(self.inputs > 0.0, 1.0, self.alpha)
        return grad_output * dx


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    df/dx = f(x) * (1 - f(x))
    """

    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        clipped = np.clip(inputs, -500.0, 500.0)
        self.output = 1.0 / (1.0 + np.exp(-clipped))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.output * (1.0 - self.output)


class Tanh(Layer):
    """
    Hyperbolic tangent activation function.
    f(x) = tanh(x)
    df/dx = 1 - f(x)^2
    """

    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1.0 - (self.output ** 2))


class Softmax(Layer):
    """
    Numerically stable Softmax activation.
    sigma(z)_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis
        self.output = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        shifted_inputs = inputs - np.max(inputs, axis=self.axis, keepdims=True)
        exp_vals = np.exp(shifted_inputs)
        self.output = exp_vals / np.sum(exp_vals, axis=self.axis, keepdims=True)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        sum_grad_s = np.sum(grad_output * self.output, axis=self.axis, keepdims=True)
        return self.output * (grad_output - sum_grad_s)
