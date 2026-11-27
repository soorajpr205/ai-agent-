"""
Interactive custom digit test and demonstration script.
Allows predicting any custom 8x8 ASCII or numpy array digit.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from infer import build_model, print_probability_bar_chart
from src.data import render_ascii_digit


def test_handcrafted_digits():
    weights_path = os.path.join(BASE_DIR, "weights", "digit_classifier.npz")
    model = build_model()
    model.load_weights(weights_path)

    # Let's test a handcrafted digit '7' and digit '8'
    custom_7 = np.array([
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32).flatten()

    custom_8 = np.array([
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32).flatten()

    tests = [
        ("Handcrafted Digit 7", 7, custom_7),
        ("Handcrafted Digit 8", 8, custom_8),
    ]

    for name, expected, img_vec in tests:
        print("\n" + "=" * 55)
        print(f"Testing {name} (Expected: {expected}):")
        print("=" * 55)
        print(render_ascii_digit(img_vec))
        probs = model.predict(img_vec.reshape(1, -1))[0]
        pred = int(np.argmax(probs))
        print_probability_bar_chart(probs, pred)
        print(f"Result: Predicted {pred} with {probs[pred]*100:.2f}% confidence!")


if __name__ == "__main__":
    test_handcrafted_digits()
