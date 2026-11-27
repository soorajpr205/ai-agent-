"""
Inference Script for Handwritten Digit Classifier.
Loads pre-trained weights from .npz archive, runs forward prediction,
and visualizes output probabilities with confidence bar chart and ASCII digit render.
"""

import os
import sys
import argparse
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.core import Sequential, Dense, ReLU, Dropout, BatchNorm1d
from src.data import generate_digit_dataset, render_ascii_digit


def build_model():
    """Builds identical architecture to training."""
    model = Sequential([
        Dense(in_features=64, out_features=128, init_method="he"),
        BatchNorm1d(num_features=128),
        ReLU(),
        Dropout(drop_rate=0.15),
        Dense(in_features=128, out_features=64, init_method="he"),
        BatchNorm1d(num_features=64),
        ReLU(),
        Dropout(drop_rate=0.1),
        Dense(in_features=64, out_features=10, init_method="he"),
    ])
    return model


def print_probability_bar_chart(probs: np.ndarray, pred_class: int):
    """Prints a neat terminal horizontal bar chart of probabilities."""
    print("\nClass Probability Distribution:")
    print("-" * 45)
    for digit in range(10):
        prob = probs[digit]
        bar_len = int(prob * 30)
        bar = "#" * bar_len
        marker = " <== PREDICTION" if digit == pred_class else ""
        print(f"Digit {digit}: [{bar:<30}] {prob * 100:6.2f}%{marker}")
    print("-" * 45)


def run_inference(num_samples: int = 5):
    weights_path = os.path.join(BASE_DIR, "weights", "digit_classifier.npz")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' does not exist.")
        print("Please run `python train.py` first to train the model and save weights.")
        return

    print("=" * 65)
    print("   NEURAL NETWORK INFERENCE ENGINE")
    print("=" * 65)

    # 1. Instantiate model and load weights
    model = build_model()
    model.load_weights(weights_path)

    # 2. Generate random test samples
    print(f"\nGenerating {num_samples} unseen test digits for inference...")
    X_test, y_test = generate_digit_dataset(n_samples_per_class=10, noise_level=0.15, random_state=123)

    indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)

    for i, idx in enumerate(indices, start=1):
        sample = X_test[idx]
        true_label = y_test[idx]

        probs = model.predict(sample.reshape(1, -1))[0]
        pred_label = int(np.argmax(probs))
        confidence = probs[pred_label] * 100.0

        is_correct = "CORRECT" if pred_label == true_label else "INCORRECT"

        print(f"\n[{i}/{num_samples}] Sample #{idx} | True Label: {true_label} | Prediction: {pred_label} ({is_correct})")
        print(render_ascii_digit(sample))
        print_probability_bar_chart(probs, pred_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on handwritten digit recognition model.")
    parser.add_argument("--samples", type=int, default=3, help="Number of random test samples to run inference on.")
    args = parser.parse_args()

    run_inference(num_samples=args.samples)
