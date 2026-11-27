"""
Training Script for Deep Neural Network on Handwritten Digits (0-9).
Constructs a multi-layer deep network, trains it with Adam and mini-batching,
evaluates on unseen test data, and saves weights to disk.
"""

import os
import sys
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.core import (
    Sequential,
    Dense,
    ReLU,
    Dropout,
    BatchNorm1d,
    CrossEntropyLoss,
    Adam,
)
from src.data import generate_digit_dataset, train_test_split, render_ascii_digit


def main():
    print("=" * 65)
    print("   DEEP LEARNING MODEL FROM SCRATCH (Pure NumPy)")
    print("   Application: Handwritten Digit Recognition (0-9)")
    print("=" * 65)

    # 1. Generate Dataset
    print("\n[1/5] Generating dataset of 8x8 handwritten digits...")
    X, y = generate_digit_dataset(n_samples_per_class=400, noise_level=0.12, random_state=42)
    print(f"Total samples: {len(X)} | Features: {X.shape[1]} | Classes: 10")

    # Split into Train, Validation, and Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Display sample digit
    sample_idx = 0
    print(f"\nVisual preview of training sample #{sample_idx} (Label = {y_train[sample_idx]}):")
    print(render_ascii_digit(X_train[sample_idx]))

    # 2. Build Deep Neural Network Architecture
    print("\n[2/5] Constructing Deep Neural Network Architecture...")
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

    model.summary()

    # 3. Compile Model
    print("\n[3/5] Compiling model with Adam optimizer and CrossEntropy loss...")
    optimizer = Adam(learning_rate=0.005, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss(from_logits=True)
    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    # 4. Train Model
    print("\n[4/5] Training model...")
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=True,
    )

    # 5. Evaluate on Unseen Test Set
    print("\n[5/5] Evaluating on independent Test Set...")
    test_metrics = model.evaluate(X_test, y_test, verbose=True)

    # Save weights
    weights_path = os.path.join(BASE_DIR, "weights", "digit_classifier.npz")
    model.save_weights(weights_path)

    # Quick test demo
    print("\n" + "=" * 65)
    print("DEMO INFERENCE ON 3 TEST SAMPLES:")
    print("=" * 65)
    demo_indices = [5, 25, 45]
    for idx in demo_indices:
        sample_img = X_test[idx]
        actual_label = y_test[idx]
        probs = model.predict(sample_img.reshape(1, -1))[0]
        pred_label = int(np.argmax(probs))
        confidence = probs[pred_label] * 100.0

        print(f"\nTest Sample #{idx}: Actual = {actual_label} | Predicted = {pred_label} (Confidence: {confidence:.2f}%)")
        print(render_ascii_digit(sample_img))

    print("\n Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
