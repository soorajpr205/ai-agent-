"""
Core Neural Network Layers implemented from scratch with NumPy.
"""

from typing import List, Tuple, Optional
import numpy as np


class Layer:
    """Base class for all neural network layers."""

    def __init__(self):
        self.training = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Returns list of tuples: (param_name, param_array, grad_array)
        for trainable parameters.
        """
        return []

    def set_training(self, mode: bool = True):
        self.training = mode


class Dense(Layer):
    """
    Fully Connected (Linear) Layer: Y = X * W + b
    
    Supports:
    - He (Kaiming) normal initialization (ideal for ReLU/LeakyReLU)
    - Xavier (Glorot) normal initialization (ideal for Sigmoid/Tanh)
    - Gradient computation for weights, biases, and input activations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_method: str = "he",
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Weight initialization
        if init_method.lower() == "he":
            std = np.sqrt(2.0 / in_features)
        elif init_method.lower() == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            std = 0.01

        self.W = (np.random.randn(in_features, out_features) * std).astype(np.float64)
        self.grad_W = np.zeros_like(self.W)

        if self.use_bias:
            self.b = np.zeros((1, out_features), dtype=np.float64)
            self.grad_b = np.zeros_like(self.b)
        else:
            self.b = None
            self.grad_b = None

        self.inputs = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.inputs = inputs
        out = np.dot(inputs, self.W)
        if self.use_bias:
            out = out + self.b
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # grad_W = X^T * grad_output
        self.grad_W = np.dot(self.inputs.T, grad_output)

        if self.use_bias:
            # grad_b = sum over batch axis
            self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # grad_input = grad_output * W^T
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        params = [("W", self.W, self.grad_W)]
        if self.use_bias:
            params.append(("b", self.b, self.grad_b))
        return params


class Dropout(Layer):
    """
    Inverted Dropout layer for regularization.
    
    Scales activations by 1 / (1 - drop_rate) during training so that
    no scaling is needed during evaluation / inference.
    """

    def __init__(self, drop_rate: float = 0.2):
        super().__init__()
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError("drop_rate must be between 0.0 and 1.0")
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.drop_rate > 0.0:
            retain_prob = 1.0 - self.drop_rate
            self.mask = (np.random.rand(*inputs.shape) >= self.drop_rate) / retain_prob
            return inputs * self.mask
        else:
            self.mask = None
            return inputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.mask is not None:
            return grad_output * self.mask
        return grad_output


class BatchNorm1d(Layer):
    """
    1D Batch Normalization layer.
    
    Normalizes mini-batch feature distributions to mean 0, var 1,
    and applies learnable affine transformation (gamma, beta).
    Tracks exponential running mean and variance for inference.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable scale and shift
        self.gamma = np.ones((1, num_features), dtype=np.float64)
        self.beta = np.zeros((1, num_features), dtype=np.float64)
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        # Running statistics
        self.running_mean = np.zeros((1, num_features), dtype=np.float64)
        self.running_var = np.ones((1, num_features), dtype=np.float64)

        # Cached tensors for backprop
        self.x_hat = None
        self.inv_std = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            batch_size = inputs.shape[0]
            if batch_size > 1:
                mean = np.mean(inputs, axis=0, keepdims=True)
                var = np.var(inputs, axis=0, keepdims=True)
                self.inv_std = 1.0 / np.sqrt(var + self.eps)
                self.x_hat = (inputs - mean) * self.inv_std

                # Update running statistics
                self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var
            else:
                self.x_hat = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
        else:
            self.x_hat = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        N = grad_output.shape[0]

        # Gradients wrt gamma and beta
        self.grad_gamma = np.sum(grad_output * self.x_hat, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        # Backpropagation through batch normalization
        dxhat = grad_output * self.gamma
        dx = (self.inv_std / N) * (
            N * dxhat
            - np.sum(dxhat, axis=0, keepdims=True)
            - self.x_hat * np.sum(dxhat * self.x_hat, axis=0, keepdims=True)
        )
        return dx

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        return [
            ("gamma", self.gamma, self.grad_gamma),
            ("beta", self.beta, self.grad_beta),
        ]
