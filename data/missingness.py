"""
Missingness Module
==================
Implements MCAR (Missing Completely At Random) mask generation and
task sampling for meta-learning over missingness-induced topologies.

Reference: Section 3.1 (Equation 1-2) and Section 5.1 of the paper.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


def generate_mcar_mask(
    shape: Tuple[int, ...],
    missing_rate: float = 0.4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate MCAR (Missing Completely At Random) binary masks.

    Each entry is independently set to 0 (missing) with probability
    equal to `missing_rate`, and 1 (observed) otherwise.

    M ∈ {0,1}^d where M_j = 1 indicates j-th coordinate is observed.

    Reference: Equation 1 in the paper.

    Args:
        shape: Shape of the mask, typically (n_samples, dim).
        missing_rate: Probability of each coordinate being missing.
                     Must be in [0, 1).
        seed: Random seed for reproducibility.

    Returns:
        Binary mask array of given shape (1=observed, 0=missing).

    Raises:
        ValueError: If missing_rate is not in [0, 1).
    """
    if not 0 <= missing_rate < 1:
        raise ValueError(
            f"missing_rate must be in [0, 1), got {missing_rate}"
        )

    rng = np.random.RandomState(seed)
    mask = (rng.rand(*shape) > missing_rate).astype(np.float32)

    # Ensure at least one observed coordinate per sample
    if len(shape) == 2:
        for i in range(shape[0]):
            if mask[i].sum() == 0:
                # Force at least one dimension to be observed
                j = rng.randint(0, shape[1])
                mask[i, j] = 1.0

    return mask


def generate_structured_mask(
    shape: Tuple[int, int],
    missing_rate: float = 0.4,
    block_size: int = 5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate structured (block) missingness masks.

    Used to test the framework under missingness shift scenarios
    where test-time patterns differ structurally from training.
    Blocks of consecutive dimensions are missing together.

    Args:
        shape: Shape (n_samples, dim).
        missing_rate: Approximate overall missingness rate.
        block_size: Size of contiguous blocks to mask.
        seed: Random seed.

    Returns:
        Binary mask of shape (n_samples, dim).
    """
    rng = np.random.RandomState(seed)
    n_samples, dim = shape
    mask = np.ones(shape, dtype=np.float32)

    n_blocks = dim // block_size
    blocks_to_mask = int(n_blocks * missing_rate)

    for i in range(n_samples):
        # Randomly choose blocks to mask for each sample
        chosen_blocks = rng.choice(n_blocks, size=blocks_to_mask, replace=False)
        for b in chosen_blocks:
            start = b * block_size
            end = min(start + block_size, dim)
            mask[i, start:end] = 0.0

        # Ensure at least one observed
        if mask[i].sum() == 0:
            j = rng.randint(0, dim)
            mask[i, j] = 1.0

    return mask


class MissingnessTaskSampler:
    """Samples missingness tasks for meta-learning.

    Each task corresponds to a unique missingness pattern (or family
    of patterns) that induces a distinct observed data manifold.
    The sampler produces tasks with varying missingness rates and
    patterns to build topological diversity.

    Reference: Section 4.0.1 of the paper.
    """

    def __init__(
        self,
        data: np.ndarray,
        missing_rates: Optional[List[float]] = None,
        samples_per_task: int = 100,
        use_structured: bool = False,
        seed: int = 42,
    ):
        """Initialize task sampler.

        Args:
            data: Full dataset of shape (n_samples, dim).
            missing_rates: List of missingness rates to sample from.
            samples_per_task: Number of samples per task.
            use_structured: Include structured missingness patterns.
            seed: Base random seed.
        """
        self.data = data
        self.missing_rates = missing_rates or [
            0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]
        self.samples_per_task = samples_per_task
        self.use_structured = use_structured
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._task_counter = 0

    def sample_task(self) -> Dict[str, np.ndarray]:
        """Sample a single task.

        Returns:
            Dictionary with:
                'observed_data': masked data (samples_per_task, dim)
                'masks': binary masks (samples_per_task, dim)
                'full_data': complete data (samples_per_task, dim)
                'missing_rate': float
        """
        # Randomly select a missingness rate
        rate = self.rng.choice(self.missing_rates)

        # Sample data subset
        n_total = self.data.shape[0]
        indices = self.rng.choice(
            n_total, size=self.samples_per_task, replace=True
        )
        task_data = self.data[indices].copy()

        # Generate mask
        task_seed = self.seed + self._task_counter
        self._task_counter += 1

        if self.use_structured and self.rng.rand() < 0.3:
            # 30% chance of structured missingness
            masks = generate_structured_mask(
                task_data.shape, rate, seed=task_seed
            )
        else:
            masks = generate_mcar_mask(
                task_data.shape, rate, seed=task_seed
            )

        return {
            "observed_data": task_data * masks,
            "masks": masks,
            "full_data": task_data,
            "missing_rate": rate,
        }

    def sample_batch(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Sample a batch of tasks.

        Args:
            batch_size: Number of tasks per batch.

        Returns:
            List of task dictionaries.
        """
        return [self.sample_task() for _ in range(batch_size)]

    def sample_shifted_task(
        self,
        train_rate: float = 0.4,
        test_rate: float = 0.8,
    ) -> Tuple[Dict, Dict]:
        """Sample a task pair with missingness shift.

        Creates training and test tasks with different missingness
        patterns to evaluate test-time adaptation under distributional
        shift (topological drift scenario).

        Args:
            train_rate: Training missingness rate.
            test_rate: Test missingness rate (higher = more shift).

        Returns:
            Tuple of (train_task, test_task) dictionaries.
        """
        task_seed = self.seed + self._task_counter
        self._task_counter += 2

        n_total = self.data.shape[0]
        indices = self.rng.choice(
            n_total, size=self.samples_per_task * 2, replace=True
        )

        train_data = self.data[indices[: self.samples_per_task]]
        test_data = self.data[indices[self.samples_per_task :]]

        # Different missingness patterns
        train_masks = generate_mcar_mask(
            train_data.shape, train_rate, seed=task_seed
        )
        test_masks = generate_mcar_mask(
            test_data.shape, test_rate, seed=task_seed + 1
        )

        train_task = {
            "observed_data": train_data * train_masks,
            "masks": train_masks,
            "full_data": train_data,
            "missing_rate": train_rate,
        }

        test_task = {
            "observed_data": test_data * test_masks,
            "masks": test_masks,
            "full_data": test_data,
            "missing_rate": test_rate,
        }

        return train_task, test_task
