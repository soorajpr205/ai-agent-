"""
Unit tests and gradient checks for Neural Network components.
Tests:
- Analytical vs Numerical gradient verification (Finite Difference check)
- Forward and backward shape integrity
- Loss calculations and stability
- Weight save and reload consistency
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.core import (
    Dense,
    ReLU,
    Sigmoid,
    Tanh,
    Dropout,
    BatchNorm1d,
    CrossEntropyLoss,
    MSELoss,
    BinaryCrossEntropyLoss,
    Adam,
    SGD,
    Sequential,
)


class TestNeuralNetwork(unittest.TestCase):

    def test_dense_forward_backward_shapes(self):
        batch_size = 16
        in_features = 32
        out_features = 8

        layer = Dense(in_features, out_features)
        X = np.random.randn(batch_size, in_features)
        out = layer.forward(X)

        self.assertEqual(out.shape, (batch_size, out_features))

        grad_out = np.random.randn(batch_size, out_features)
        grad_in = layer.backward(grad_out)

        self.assertEqual(grad_in.shape, (batch_size, in_features))
        self.assertEqual(layer.grad_W.shape, (in_features, out_features))
        self.assertEqual(layer.grad_b.shape, (1, out_features))

    def test_dense_gradient_check(self):
        """Numerical gradient check using finite difference."""
        layer = Dense(4, 2, use_bias=True)
        X = np.random.randn(2, 4)

        # Loss function: L = 0.5 * sum((Y - target)^2)
        target = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Forward & analytical backward
        out = layer.forward(X)
        dL_dout = out - target
        layer.backward(dL_dout)

        analytical_grad_W = layer.grad_W.copy()

        # Numerical gradient for W
        eps = 1e-5
        numerical_grad_W = np.zeros_like(layer.W)

        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                orig = layer.W[i, j]

                layer.W[i, j] = orig + eps
                out_pos = layer.forward(X)
                loss_pos = 0.5 * np.sum((out_pos - target) ** 2)

                layer.W[i, j] = orig - eps
                out_neg = layer.forward(X)
                loss_neg = 0.5 * np.sum((out_neg - target) ** 2)

                numerical_grad_W[i, j] = (loss_pos - loss_neg) / (2.0 * eps)
                layer.W[i, j] = orig

        # Relative error
        rel_error = np.linalg.norm(analytical_grad_W - numerical_grad_W) / (
            np.linalg.norm(analytical_grad_W) + np.linalg.norm(numerical_grad_W) + 1e-12
        )
        self.assertLess(rel_error, 1e-4)

    def test_activations_forward_backward(self):
        X = np.array([[-1.0, 0.0, 2.0]], dtype=float)

        # ReLU
        relu = ReLU()
        out = relu.forward(X)
        np.testing.assert_array_almost_equal(out, [[0.0, 0.0, 2.0]])
        grad = relu.backward(np.array([[1.0, 1.0, 1.0]]))
        np.testing.assert_array_almost_equal(grad, [[0.0, 0.0, 1.0]])

        # Sigmoid
        sig = Sigmoid()
        out = sig.forward(np.array([[0.0]]))
        self.assertAlmostEqual(out[0, 0], 0.5)

        # Tanh
        tanh = Tanh()
        out = tanh.forward(np.array([[0.0]]))
        self.assertAlmostEqual(out[0, 0], 0.0)

    def test_cross_entropy_stability(self):
        loss_fn = CrossEntropyLoss(from_logits=True)
        # Logits with extreme values
        logits = np.array([[1000.0, -1000.0, 0.0], [-500.0, 500.0, 10.0]])
        y_true = np.array([0, 1])

        loss = loss_fn.forward(logits, y_true)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
        self.assertGreaterEqual(loss, 0.0)

        grad = loss_fn.backward()
        self.assertEqual(grad.shape, logits.shape)
        self.assertFalse(np.any(np.isnan(grad)))

    def test_save_and_load_weights(self):
        model1 = Sequential([
            Dense(10, 20),
            ReLU(),
            Dense(20, 5),
        ])

        temp_path = os.path.join(BASE_DIR, "weights", "test_weights.npz")
        model1.save_weights(temp_path)

        model2 = Sequential([
            Dense(10, 20),
            ReLU(),
            Dense(20, 5),
        ])
        model2.load_weights(temp_path)

        # Check weights match
        for l1, l2 in zip(model1.layers, model2.layers):
            p1 = l1.get_params()
            p2 = l2.get_params()
            for (_, arr1, _), (_, arr2, _) in zip(p1, p2):
                np.testing.assert_array_almost_equal(arr1, arr2)

        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
