"""
Dataset utilities and generators for training Deep Neural Networks.
Includes:
- Synthetic Handwritten Digits Dataset (0-9) rendered on 8x8 pixel grids with ASCII viewer
- Non-linear multi-class spiral dataset
- Train/test splitting and standardization
"""

from typing import Tuple
import numpy as np


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, shuffle: bool = True, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset into train and test subsets."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n_samples * (1.0 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def generate_digit_dataset(
    n_samples_per_class: int = 300, noise_level: float = 0.12, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a realistic 8x8 optical handwritten digit dataset (0-9)
    using canonical digit stroke templates with random deformations,
    translations, and Gaussian noise.
    
    Total samples = 10 * n_samples_per_class.
    Shape of X: (Total, 64) in range [0, 1].
    Shape of y: (Total,) with integer class labels 0..9.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Canonical 8x8 bitmap templates for digits 0 through 9
    templates = {
        0: [
            "  XXXX  ",
            " X    X ",
            " X    X ",
            " X    X ",
            " X    X ",
            " X    X ",
            " X    X ",
            "  XXXX  ",
        ],
        1: [
            "   XX   ",
            "  XXX   ",
            "   XX   ",
            "   XX   ",
            "   XX   ",
            "   XX   ",
            "   XX   ",
            "  XXXX  ",
        ],
        2: [
            "  XXXX  ",
            " X    X ",
            "      X ",
            "    XX  ",
            "   XX   ",
            "  XX    ",
            " X      ",
            " XXXXXX ",
        ],
        3: [
            " XXXXX  ",
            "      X ",
            "      X ",
            "   XXXX ",
            "      X ",
            "      X ",
            " X    X ",
            "  XXXX  ",
        ],
        4: [
            "    XX  ",
            "   XXX  ",
            "  X XX  ",
            " X  XX  ",
            " XXXXXX ",
            "    XX  ",
            "    XX  ",
            "    XX  ",
        ],
        5: [
            " XXXXXX ",
            " X      ",
            " XXXXX  ",
            "      X ",
            "      X ",
            "      X ",
            " X    X ",
            "  XXXX  ",
        ],
        6: [
            "  XXXX  ",
            " X    X ",
            " X      ",
            " XXXXX  ",
            " X    X ",
            " X    X ",
            " X    X ",
            "  XXXX  ",
        ],
        7: [
            " XXXXXX ",
            "     XX ",
            "    XX  ",
            "   XX   ",
            "   XX   ",
            "  XX    ",
            "  XX    ",
            "  XX    ",
        ],
        8: [
            "  XXXX  ",
            " X    X ",
            " X    X ",
            "  XXXX  ",
            " X    X ",
            " X    X ",
            " X    X ",
            "  XXXX  ",
        ],
        9: [
            "  XXXX  ",
            " X    X ",
            " X    X ",
            "  XXXXX ",
            "      X ",
            "      X ",
            " X    X ",
            "  XXXX  ",
        ],
    }

    base_bitmaps = {}
    for digit, rows in templates.items():
        mat = np.array([[1.0 if ch == "X" else 0.0 for ch in r] for r in rows], dtype=np.float32)
        base_bitmaps[digit] = mat

    X_list = []
    y_list = []

    for digit in range(10):
        base = base_bitmaps[digit]
        for _ in range(n_samples_per_class):
            sample = base.copy()

            # Random intensity variation
            intensity_variation = np.random.uniform(0.75, 1.25, size=base.shape)
            sample = sample * intensity_variation

            # Random pixel shift (translation by -1, 0, or 1 in row and col)
            shift_r = int(np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]))
            shift_c = int(np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]))
            sample = np.roll(sample, shift_r, axis=0)
            sample = np.roll(sample, shift_c, axis=1)

            # Zero out rolled edges if shifted
            if shift_r == 1:
                sample[0, :] = 0.0
            elif shift_r == -1:
                sample[-1, :] = 0.0
            if shift_c == 1:
                sample[:, 0] = 0.0
            elif shift_c == -1:
                sample[:, -1] = 0.0

            # Add Gaussian noise and clip to [0, 1]
            noise = np.random.normal(0.0, noise_level, size=base.shape)
            sample = np.clip(sample + noise, 0.0, 1.0)

            X_list.append(sample.flatten())
            y_list.append(digit)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def render_ascii_digit(image_vector: np.ndarray, width: int = 8, height: int = 8) -> str:
    """
    Renders an 8x8 or 16x16 image vector as a stylized ASCII art graphic
    using standard ASCII characters compatible with all terminal encodings.
    """
    chars = [" ", ".", ":", "*", "#"]
    img = image_vector.reshape((height, width))
    lines = []
    lines.append("+" + "--" * width + "+")
    for r in range(height):
        row_str = "|"
        for c in range(width):
            val = img[r, c]
            idx = int(np.clip(val * (len(chars) - 1), 0, len(chars) - 1))
            char = chars[idx]
            row_str += char + char  # Double width for square aspect ratio
        row_str += "|"
        lines.append(row_str)
    lines.append("+" + "--" * width + "+")
    return "\n".join(lines)


def generate_spiral_dataset(n_samples_per_class: int = 200, n_classes: int = 3, noise: float = 0.2):
    """Generates 2D non-linear multi-class spiral dataset for benchmarking."""
    np.random.seed(42)
    X = np.zeros((n_samples_per_class * n_classes, 2))
    y = np.zeros(n_samples_per_class * n_classes, dtype=int)

    for j in range(n_classes):
        ix = range(n_samples_per_class * j, n_samples_per_class * (j + 1))
        r = np.linspace(0.0, 1.0, n_samples_per_class)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, n_samples_per_class) + np.random.randn(n_samples_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y
