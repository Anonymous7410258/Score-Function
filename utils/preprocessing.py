"""
Preprocessing Utilities
=======================
Data normalization and preprocessing functions.

Usage:
    python -m utils.preprocessing --input data/raw/ --output data/processed/
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional


def normalize_data(
    data: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, dict]:
    """Min-max normalize data to a specified range.

    Args:
        data: Input array of shape (n, d).
        feature_range: Target range (min, max).

    Returns:
        Tuple of (normalized_data, stats_dict).
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    scale = data_max - data_min
    scale[scale == 0] = 1.0  # Avoid division by zero

    lo, hi = feature_range
    normalized = lo + (data - data_min) / scale * (hi - lo)

    stats = {
        "min": data_min,
        "max": data_max,
        "scale": scale,
        "feature_range": feature_range,
    }

    return normalized.astype(np.float32), stats


def standardize_data(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """Standardize data to zero mean and unit variance.

    Args:
        data: Input array of shape (n, d).
        mean: Pre-computed mean (for applying to test data).
        std: Pre-computed std (for applying to test data).

    Returns:
        Tuple of (standardized_data, stats_dict).
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std == 0] = 1.0

    standardized = (data - mean) / std

    stats = {"mean": mean, "std": std}
    return standardized.astype(np.float32), stats


def inverse_standardize(
    data: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """Reverse standardization.

    Args:
        data: Standardized array.
        stats: Dictionary with 'mean' and 'std'.

    Returns:
        Original-scale data.
    """
    return (data * stats["std"] + stats["mean"]).astype(np.float32)


def apply_mask(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply missingness mask to data.

    Args:
        data: Input array of shape (n, d).
        mask: Binary mask of shape (n, d).
        fill_value: Value for missing entries.

    Returns:
        Masked data with missing values replaced.
    """
    masked = data.copy()
    masked[mask == 0] = fill_value
    return masked


def main():
    """CLI for preprocessing raw data."""
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory or file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="standardize",
        choices=["standardize", "normalize"],
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".npy")
        ]

    for filepath in files:
        data = np.load(filepath)
        basename = os.path.basename(filepath)

        if args.method == "standardize":
            processed, stats = standardize_data(data)
        else:
            processed, stats = normalize_data(data)

        np.save(os.path.join(args.output, basename), processed)
        np.save(
            os.path.join(args.output, basename.replace(".npy", "_stats.npy")),
            stats,
        )
        print(f"Processed {basename}: {data.shape} → {processed.shape}")


if __name__ == "__main__":
    main()
