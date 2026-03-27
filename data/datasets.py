"""
Dataset Module
==============
Generates and loads datasets for score matching experiments.

Implements synthetic data generators matching the experimental protocol
in Section 5.1 of the paper:
- Multivariate Gaussian (truncated and non-truncated)
- ICA-style non-Gaussian mixtures
- Gaussian Graphical Models (GGM)
- Financial time-series (synthetic proxy)
- Biological expression data (synthetic proxy)

Reference: Section 5.1 (Datasets) of the accompanying paper.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict


def generate_gaussian_data(
    n_samples: int = 5000,
    dim: int = 50,
    truncated: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate multivariate Gaussian data.

    Creates samples from a multivariate normal distribution with a
    randomly generated positive-definite covariance matrix.

    Args:
        n_samples: Number of samples to generate.
        dim: Data dimensionality d.
        truncated: If True, truncate samples to positive orthant.
        seed: Random seed for reproducibility.

    Returns:
        Data array of shape (n_samples, dim).
    """
    rng = np.random.RandomState(seed)

    # Generate random positive-definite covariance matrix
    A = rng.randn(dim, dim) * 0.5
    cov = A @ A.T + np.eye(dim) * 0.1  # Ensure positive definiteness
    mean = rng.randn(dim) * 0.1

    # Sample from multivariate Gaussian
    data = rng.multivariate_normal(mean, cov, size=n_samples)

    if truncated:
        # Truncate to positive orthant via rejection sampling
        accepted = data[np.all(data > 0, axis=1)]
        while accepted.shape[0] < n_samples:
            extra = rng.multivariate_normal(mean, cov, size=n_samples * 2)
            accepted = np.vstack([accepted, extra[np.all(extra > 0, axis=1)]])
        data = accepted[:n_samples]

    return data.astype(np.float32)


def generate_ica_data(
    n_samples: int = 5000,
    dim: int = 20,
    n_components: int = 5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate ICA-style non-Gaussian mixture data.

    Creates data from an Independent Component Analysis model:
    X = A · S, where S are non-Gaussian independent sources
    (Laplacian, uniform, sub/super-Gaussian mixtures).

    Args:
        n_samples: Number of samples.
        dim: Data dimensionality.
        n_components: Number of independent components.
        seed: Random seed.

    Returns:
        Data array of shape (n_samples, dim).
    """
    rng = np.random.RandomState(seed)

    # Generate non-Gaussian sources
    sources = []
    for i in range(n_components):
        source_type = i % 3
        if source_type == 0:
            # Laplacian (super-Gaussian)
            s = rng.laplace(0, 1, size=n_samples)
        elif source_type == 1:
            # Uniform (sub-Gaussian)
            s = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n_samples)
        else:
            # Student-t (heavy-tailed)
            s = rng.standard_t(df=5, size=n_samples)
        sources.append(s)

    S = np.stack(sources, axis=1)  # (n_samples, n_components)

    # Random mixing matrix
    A = rng.randn(dim, n_components) * 0.5

    data = S @ A.T  # (n_samples, dim)

    return data.astype(np.float32)


def generate_ggm_data(
    n_samples: int = 5000,
    dim: int = 50,
    sparsity: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian Graphical Model data.

    Creates data from a sparse precision matrix model, useful for
    evaluating structure recovery (AUC-ROC, SHD).

    Args:
        n_samples: Number of samples.
        dim: Data dimensionality.
        sparsity: Fraction of non-zero off-diagonal entries in the
                 precision matrix.
        seed: Random seed.

    Returns:
        Tuple of:
            - Data array of shape (n_samples, dim).
            - True adjacency matrix of shape (dim, dim).
    """
    rng = np.random.RandomState(seed)

    # Generate sparse precision matrix (inverse covariance)
    # Start with random sparse symmetric matrix
    B = rng.randn(dim, dim) * 0.3
    mask = rng.rand(dim, dim) < sparsity
    B = B * mask
    B = (B + B.T) / 2  # Symmetrize

    # Make positive definite by adding diagonal dominance
    precision = B + np.eye(dim) * (np.abs(B).sum(axis=1).max() + 0.5)
    precision = (precision + precision.T) / 2  # Ensure symmetry

    # Adjacency matrix: non-zero off-diagonal entries
    adjacency = (np.abs(precision) > 1e-10).astype(float)
    np.fill_diagonal(adjacency, 0)

    # Covariance and sampling
    cov = np.linalg.inv(precision)
    mean = np.zeros(dim)
    data = rng.multivariate_normal(mean, cov, size=n_samples)

    return data.astype(np.float32), adjacency


def generate_financial_data(
    n_samples: int = 5000,
    dim: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic financial time-series proxy data.

    Creates correlated multivariate data with factor structure
    that mimics properties of financial returns.

    Args:
        n_samples: Number of samples.
        dim: Number of assets / features.
        seed: Random seed.

    Returns:
        Data array of shape (n_samples, dim).
    """
    rng = np.random.RandomState(seed)

    n_factors = min(10, dim // 5)
    # Factor model: X = F · L^T + ε
    F = rng.randn(n_samples, n_factors)  # Factors
    L = rng.randn(dim, n_factors) * 0.5  # Loadings
    epsilon = rng.randn(n_samples, dim) * 0.2  # Idiosyncratic noise

    data = F @ L.T + epsilon

    # Add heavy tails (Student-t noise)
    heavy_tail = rng.standard_t(df=5, size=(n_samples, dim)) * 0.1
    data += heavy_tail

    return data.astype(np.float32)


def generate_biological_data(
    n_samples: int = 2000,
    dim: int = 500,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic biological expression data.

    Creates high-dimensional data with sparse structure mimicking
    gene expression data properties.

    Args:
        n_samples: Number of samples.
        dim: Number of genes / features.
        seed: Random seed.

    Returns:
        Data array of shape (n_samples, dim).
    """
    rng = np.random.RandomState(seed)

    # Low-rank structure with sparse perturbation
    rank = 20
    U = rng.randn(n_samples, rank)
    V = rng.randn(dim, rank) * 0.3

    # Sparse activations (many genes are near-zero)
    activity = rng.binomial(1, 0.3, size=(n_samples, dim)).astype(float)
    noise = rng.randn(n_samples, dim) * 0.1

    data = (U @ V.T) * activity + noise

    return data.astype(np.float32)


# Dataset registry for easy access
DATASET_GENERATORS = {
    "gaussian": generate_gaussian_data,
    "gaussian_truncated": lambda **kw: generate_gaussian_data(
        truncated=True, **kw
    ),
    "ica": generate_ica_data,
    "ggm": generate_ggm_data,
    "financial": generate_financial_data,
    "biological": generate_biological_data,
}


class ScoreMatchingDataset(Dataset):
    """PyTorch Dataset wrapping generated data with missingness masks.

    Each item returns (x_observed, mask, x_full) where:
        x_observed = x_full * mask (missing values zeroed out)
        mask = binary MCAR mask
        x_full = complete data (for oracle evaluation only)
    """

    def __init__(
        self,
        data: np.ndarray,
        masks: np.ndarray,
    ):
        """Initialize dataset.

        Args:
            data: Full data array of shape (n, d).
            masks: Binary masks of shape (n, d).
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x_full = self.data[idx]
        mask = self.masks[idx]
        x_observed = x_full * mask
        return x_observed, mask, x_full


def create_task_distribution(
    dataset_name: str,
    n_samples: int = 5000,
    dim: int = 50,
    missing_rates: Optional[list] = None,
    n_tasks: int = 100,
    seed: int = 42,
    **kwargs,
) -> list:
    """Create a distribution of tasks for meta-learning.

    Each task corresponds to a specific missingness pattern applied
    to the same underlying data. Tasks vary in missingness rate
    and specific mask patterns.

    Args:
        dataset_name: Name of dataset generator.
        n_samples: Total samples in the dataset.
        dim: Data dimensionality.
        missing_rates: List of missingness rates to sample from.
        n_tasks: Number of tasks to generate.
        seed: Random seed.

    Returns:
        List of task dictionaries, each containing:
            'data': observed data (n_per_task, d)
            'masks': masks (n_per_task, d)
            'full_data': complete data
            'missing_rate': float
    """
    from data.missingness import generate_mcar_mask

    if missing_rates is None:
        missing_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    rng = np.random.RandomState(seed)

    # Generate base dataset
    gen_fn = DATASET_GENERATORS[dataset_name]
    gen_kwargs = {"n_samples": n_samples, "dim": dim, "seed": seed}

    result = gen_fn(**gen_kwargs)

    if isinstance(result, tuple):
        data = result[0]  # For GGM which returns (data, adjacency)
    else:
        data = result

    # Create tasks with different missingness patterns
    tasks = []
    samples_per_task = max(50, n_samples // n_tasks)

    for t in range(n_tasks):
        # Random missingness rate
        rate = rng.choice(missing_rates)

        # Random subset of data
        indices = rng.choice(n_samples, size=samples_per_task, replace=True)
        task_data = data[indices]

        # Generate MCAR mask
        task_masks = generate_mcar_mask(
            task_data.shape, missing_rate=rate, seed=seed + t
        )

        tasks.append(
            {
                "data": task_data * task_masks,
                "masks": task_masks,
                "full_data": task_data,
                "missing_rate": rate,
            }
        )

    return tasks
