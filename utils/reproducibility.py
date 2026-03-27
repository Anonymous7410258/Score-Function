"""
Reproducibility Utilities
=========================
Ensures deterministic behavior across runs by setting random seeds
for all relevant libraries and enabling deterministic CUDA operations.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for Python, NumPy, PyTorch (CPU and CUDA),
    and enables deterministic CUDA operations.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device_str: One of 'auto', 'cuda', 'cpu'.

    Returns:
        torch.device for computation.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
