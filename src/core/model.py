"""
Sequential Model container for Deep Neural Networks in pure NumPy.
Provides model definition, forward/backward loops, training with mini-batching,
evaluation, weight persistence (.npz), and summary.
"""

from typing import List, Tuple, Optional, Dict
import os
import time
import numpy as np

from src.core.layers import Layer
from src.core.losses import Loss
from src.core.optimizers import Optimizer


class Sequential:
    """
    Sequential Neural Network container.
    Manages linear stack of layers, execution pipeline, and optimization loop.
    """

    def __init__(self, layers: Optional[List[Layer]] = None):
        self.layers: List[Layer] = layers if layers is not None else []
        self.optimizer: Optional[Optimizer] = None
        self.loss_fn: Optional[Loss] = None
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def add(self, layer: Layer) -> "Sequential":
        """Add a layer to the neural network."""
        self.layers.append(layer)
        return self

    def compile(self, optimizer: Optimizer, loss_fn: Loss):
        """Configure training optimizer and loss function."""
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Collects latest parameter references and gradients from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through all stacked layers."""
        out = X
        for layer in self.layers:
            layer.set_training(training)
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad_loss: np.ndarray):
        """Backward pass through all stacked layers in reverse."""
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
        shuffle: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model using mini-batch gradient descent.
        """
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Model must be compiled with compile(optimizer, loss_fn) before training.")

        num_samples = X_train.shape[0]

        if verbose:
            print("\n" + "=" * 65)
            print(f"Training Neural Network on {num_samples} samples | Batch size: {batch_size} | Epochs: {epochs}")
            print("=" * 65)

        total_start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            indices = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(indices)

            epoch_losses = []
            correct_preds = 0

            # Mini-batch loop
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # 1. Forward Pass
                logits = self.forward(X_batch, training=True)

                # 2. Compute Loss
                batch_loss = self.loss_fn.forward(logits, y_batch)
                epoch_losses.append(batch_loss)

                # Track training accuracy
                pred_classes = np.argmax(logits, axis=-1)
                true_classes = y_batch if y_batch.ndim == 1 else np.argmax(y_batch, axis=-1)
                correct_preds += np.sum(pred_classes == true_classes)

                # 3. Backward Pass
                grad_loss = self.loss_fn.backward()
                self.backward(grad_loss)

                # 4. Parameter Update with live parameter references
                self.optimizer.step(self.get_params())

            train_loss = float(np.mean(epoch_losses))
            train_acc = float(correct_preds / num_samples)
            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)

            val_str = ""
            if validation_data is not None:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val, verbose=False)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                val_str = f" | val_loss: {val_metrics['loss']:.4f} | val_acc: {val_metrics['accuracy'] * 100:.2f}%"

            epoch_time = time.time() - epoch_start
            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.2f}s] - "
                    f"loss: {train_loss:.4f} | acc: {train_acc * 100:.2f}%"
                    f"{val_str}"
                )

        total_time = time.time() - total_start_time
        if verbose:
            print("=" * 65)
            print(f"Training completed in {total_time:.2f} seconds.")
            print("=" * 65 + "\n")

        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """Evaluate model loss and accuracy on test/validation dataset."""
        logits = self.forward(X, training=False)
        loss = 0.0
        if self.loss_fn is not None:
            loss = self.loss_fn.forward(logits, y)

        preds = np.argmax(logits, axis=-1)
        targets = y if y.ndim == 1 else np.argmax(y, axis=-1)
        accuracy = float(np.mean(preds == targets))

        metrics = {"loss": loss, "accuracy": accuracy}
        if verbose:
            print(f"Evaluation -> Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate softmax class probabilities for inputs."""
        logits = self.forward(X, training=False)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels with highest probability."""
        probs = self.predict(X)
        return np.argmax(probs, axis=-1)

    def summary(self):
        """Display a detailed table of model layers and parameter counts."""
        print("-" * 65)
        print(f"{'Layer (Type)':<25}{'Output Shape / Config':<25}{'Param #':<15}")
        print("=" * 65)

        total_params = 0
        trainable_params = 0

        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}_{i+1}"
            params = layer.get_params()
            layer_param_count = sum(p[1].size for p in params)
            total_params += layer_param_count
            trainable_params += layer_param_count

            extra_info = ""
            if hasattr(layer, "out_features"):
                extra_info = f"out_features={layer.out_features}"
            elif hasattr(layer, "drop_rate"):
                extra_info = f"drop_rate={layer.drop_rate}"
            elif hasattr(layer, "num_features"):
                extra_info = f"num_features={layer.num_features}"

            print(f"{layer_name:<25}{extra_info:<25}{layer_param_count:<15,}")

        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print("-" * 65)

    def save_weights(self, filepath: str):
        """Save all model parameter arrays and layer state to a compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        param_dict = {}
        for layer_idx, layer in enumerate(self.layers):
            params = layer.get_params()
            for p_name, p_array, _ in params:
                key = f"layer_{layer_idx}_{p_name}"
                param_dict[key] = p_array

            # Save batchnorm running statistics if present
            if hasattr(layer, "running_mean"):
                param_dict[f"layer_{layer_idx}_running_mean"] = layer.running_mean
                param_dict[f"layer_{layer_idx}_running_var"] = layer.running_var

        np.savez_compressed(filepath, **param_dict)
        print(f"Successfully saved model weights to: {filepath}")

    def load_weights(self, filepath: str):
        """Load parameter arrays and layer state from a .npz archive."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file not found: {filepath}")

        archive = np.load(filepath)
        for layer_idx, layer in enumerate(self.layers):
            params = layer.get_params()
            for p_name, p_array, _ in params:
                key = f"layer_{layer_idx}_{p_name}"
                if key in archive:
                    np.copyto(p_array, archive[key])
                else:
                    print(f"Warning: Key {key} not found in saved weights.")

            # Restore batchnorm running statistics if present
            if hasattr(layer, "running_mean"):
                rm_key = f"layer_{layer_idx}_running_mean"
                rv_key = f"layer_{layer_idx}_running_var"
                if rm_key in archive:
                    np.copyto(layer.running_mean, archive[rm_key])
                if rv_key in archive:
                    np.copyto(layer.running_var, archive[rv_key])

        print(f"Successfully loaded model weights from: {filepath}")
