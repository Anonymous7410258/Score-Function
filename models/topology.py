"""
Topology Module
===============
Implements topological feature extraction using persistent homology.

This module computes:
1. Persistence diagrams via Vietoris-Rips filtration on observed point clouds.
2. Persistence image vectorization Φ for stable finite-dimensional embedding.
3. Wasserstein distance between persistence diagrams for drift detection.

Reference: Sections 3.4–3.6, 4.0.2 of the accompanying paper.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict


def _compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix.

    Args:
        X: Point cloud of shape (n, d).

    Returns:
        Distance matrix of shape (n, n).
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _rips_persistence_diagram(
    distance_matrix: np.ndarray,
    max_dim: int = 1,
    max_edge: float = 2.0,
) -> List[np.ndarray]:
    """Compute persistence diagrams via Vietoris-Rips filtration.

    Uses a simplified ripser-like algorithm. For production use with
    large datasets, install and use the `ripser` or `giotto-tda` package.

    Args:
        distance_matrix: Pairwise distance matrix of shape (n, n).
        max_dim: Maximum homological dimension.
        max_edge: Maximum edge length for filtration.

    Returns:
        List of persistence diagrams, one per dimension.
        Each diagram is an array of shape (num_features, 2) with
        columns [birth, death].
    """
    try:
        from gtda.homology import VietorisRipsPersistence

        # Use giotto-tda if available
        vr = VietorisRipsPersistence(
            homology_dimensions=list(range(max_dim + 1)),
            max_edge_length=max_edge,
            metric="precomputed",
        )
        # giotto-tda expects 3D input: (n_samples, n_points, n_points)
        diagrams = vr.fit_transform(distance_matrix[np.newaxis, :, :])[0]

        # Split by dimension
        result = []
        for dim in range(max_dim + 1):
            mask = diagrams[:, 2] == dim
            dim_diagram = diagrams[mask, :2]
            # Filter infinite deaths
            finite_mask = np.isfinite(dim_diagram[:, 1])
            result.append(dim_diagram[finite_mask])

        return result

    except ImportError:
        # Fallback: simplified H0 computation using union-find
        return _simple_persistence(distance_matrix, max_dim, max_edge)


def _simple_persistence(
    distance_matrix: np.ndarray,
    max_dim: int = 1,
    max_edge: float = 2.0,
) -> List[np.ndarray]:
    """Simplified persistence computation (H0 connected components).

    This is a fallback when giotto-tda is not installed. It computes
    H0 (connected components) using a union-find based approach.

    Args:
        distance_matrix: Pairwise distance matrix.
        max_dim: Maximum homological dimension (only H0 computed here).
        max_edge: Maximum filtration value.

    Returns:
        List of persistence diagrams per dimension.
    """
    n = distance_matrix.shape[0]

    # Get sorted edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] <= max_edge:
                edges.append((distance_matrix[i, j], i, j))
    edges.sort()

    # Union-Find for H0
    parent = list(range(n))
    birth = [0.0] * n  # All components born at 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    h0_diagram = []
    for weight, u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            # Merge: the younger component dies
            # Convention: birth=0 for all, death=edge weight
            h0_diagram.append([0.0, weight])
            parent[rv] = ru

    h0 = np.array(h0_diagram) if h0_diagram else np.zeros((0, 2))

    # For H1 and above, return empty diagrams in fallback mode
    result = [h0]
    for _ in range(max_dim):
        result.append(np.zeros((0, 2)))

    return result


def compute_persistence_image(
    diagram: np.ndarray,
    resolution: Tuple[int, int] = (10, 10),
    sigma: float = 0.1,
    birth_range: Optional[Tuple[float, float]] = None,
    persistence_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Convert a persistence diagram to a persistence image.

    Transforms (birth, death) pairs into a stable, fixed-size vector
    representation using Gaussian kernel density estimation on a grid
    in birth-persistence coordinates.

    Args:
        diagram: Persistence diagram of shape (n_features, 2).
        resolution: Grid resolution (height, width) for the image.
        sigma: Gaussian bandwidth for kernel smoothing.
        birth_range: Range of birth values. Auto-computed if None.
        persistence_range: Range of persistence values. Auto-computed if None.

    Returns:
        Persistence image as flattened vector of shape (resolution[0] * resolution[1],).
    """
    h, w = resolution
    image = np.zeros((h, w))

    if diagram.shape[0] == 0:
        return image.flatten()

    # Convert to birth-persistence coordinates
    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]

    # Filter out zero-persistence features
    mask = persistences > 1e-8
    births = births[mask]
    persistences = persistences[mask]

    if len(births) == 0:
        return image.flatten()

    # Determine ranges
    if birth_range is None:
        birth_range = (births.min() - sigma, births.max() + sigma)
    if persistence_range is None:
        persistence_range = (0, persistences.max() + sigma)

    # Create grid
    birth_grid = np.linspace(birth_range[0], birth_range[1], w)
    pers_grid = np.linspace(persistence_range[0], persistence_range[1], h)

    # Weight by persistence (more persistent features contribute more)
    weights = persistences / (persistences.max() + 1e-8)

    # Compute persistence image via Gaussian KDE
    for birth_val, pers_val, weight in zip(births, persistences, weights):
        for i, pg in enumerate(pers_grid):
            for j, bg in enumerate(birth_grid):
                dist_sq = (bg - birth_val) ** 2 + (pg - pers_val) ** 2
                image[i, j] += weight * np.exp(-dist_sq / (2 * sigma ** 2))

    return image.flatten()


def compute_wasserstein_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    p: int = 2,
) -> float:
    """Compute p-Wasserstein distance between two persistence diagrams.

    Implements Equation 8 from the paper. Uses the L∞ ground metric
    and computes the optimal matching via the Hungarian algorithm.

    Args:
        diagram1: First persistence diagram of shape (n1, 2).
        diagram2: Second persistence diagram of shape (n2, 2).
        p: Exponent for the Wasserstein distance (default: 2).

    Returns:
        The p-Wasserstein distance between the diagrams.
    """
    try:
        from persim import wasserstein

        return wasserstein(diagram1, diagram2, matching=False)
    except ImportError:
        pass

    # Fallback: simplified matching using scipy
    from scipy.optimize import linear_sum_assignment

    # Augment diagrams with diagonal projections
    n1 = diagram1.shape[0] if diagram1.shape[0] > 0 else 0
    n2 = diagram2.shape[0] if diagram2.shape[0] > 0 else 0

    if n1 == 0 and n2 == 0:
        return 0.0

    # Cost of point to diagonal: half persistence
    def diag_cost(pt):
        return ((pt[1] - pt[0]) / 2.0) ** p

    # Build cost matrix for augmented matching
    size = n1 + n2
    cost = np.full((size, size), np.inf)

    # Point-to-point costs
    for i in range(n1):
        for j in range(n2):
            cost[i, j] = np.max(np.abs(diagram1[i] - diagram2[j])) ** p

    # Point-to-diagonal costs
    for i in range(n1):
        for j in range(n2, size):
            cost[i, j] = diag_cost(diagram1[i])

    for i in range(n1, size):
        for j in range(n2):
            cost[i, j] = diag_cost(diagram2[j])

    # Diagonal-to-diagonal: zero cost
    for i in range(n1, size):
        for j in range(n2, size):
            cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    total = cost[row_ind, col_ind].sum()

    return total ** (1.0 / p)


class TopologyExtractor:
    """Extract topological features from point clouds.

    Wraps the full pipeline: point cloud → persistence diagrams →
    persistence images. Produces a fixed-size topological feature
    vector f_T = Φ(P_T) ∈ R^m as described in Equation 11.

    Args:
        max_homology_dim: Maximum homological dimension K.
        resolution: Persistence image resolution per dimension.
        sigma: Persistence image Gaussian bandwidth.
        max_points: Maximum points to subsample (for efficiency).
        max_edge: Maximum edge length for Rips filtration.
    """

    def __init__(
        self,
        max_homology_dim: int = 1,
        resolution: Tuple[int, int] = (10, 10),
        sigma: float = 0.1,
        max_points: int = 500,
        max_edge: float = 2.0,
    ):
        self.max_homology_dim = max_homology_dim
        self.resolution = tuple(resolution)
        self.sigma = sigma
        self.max_points = max_points
        self.max_edge = max_edge

        # Feature dimension: resolution product × (K+1) dimensions
        self.feature_dim = (
            self.resolution[0] * self.resolution[1] * (max_homology_dim + 1)
        )

    def extract(
        self,
        X: np.ndarray,
        return_diagrams: bool = False,
    ) -> Dict:
        """Extract topological features from a point cloud.

        Args:
            X: Point cloud of shape (n_samples, n_features).
            return_diagrams: If True, also return raw persistence diagrams.

        Returns:
            Dictionary with keys:
                'features': Persistence image feature vector, shape (m,).
                'diagrams': (optional) List of persistence diagrams.
        """
        # Subsample if necessary
        if X.shape[0] > self.max_points:
            indices = np.random.choice(
                X.shape[0], self.max_points, replace=False
            )
            X_sub = X[indices]
        else:
            X_sub = X

        # Compute pairwise distances
        dist_matrix = _compute_pairwise_distances(X_sub)

        # Compute persistence diagrams
        diagrams = _rips_persistence_diagram(
            dist_matrix,
            max_dim=self.max_homology_dim,
            max_edge=self.max_edge,
        )

        # Compute persistence images for each dimension
        pi_vectors = []
        for dim_diagram in diagrams:
            pi = compute_persistence_image(
                dim_diagram,
                resolution=self.resolution,
                sigma=self.sigma,
            )
            pi_vectors.append(pi)

        # Concatenate into single feature vector f_T
        features = np.concatenate(pi_vectors)

        result = {"features": features}
        if return_diagrams:
            result["diagrams"] = diagrams

        return result

    def extract_torch(
        self,
        X: torch.Tensor,
        return_diagrams: bool = False,
    ) -> Dict:
        """Extract topological features from a PyTorch tensor.

        Args:
            X: Point cloud tensor of shape (n_samples, n_features).
            return_diagrams: If True, also return raw persistence diagrams.

        Returns:
            Dictionary with:
                'features': torch.Tensor of shape (m,).
                'diagrams': (optional) List of persistence diagrams.
        """
        X_np = X.detach().cpu().numpy()
        result = self.extract(X_np, return_diagrams=return_diagrams)
        result["features"] = torch.tensor(
            result["features"], dtype=torch.float32
        )
        return result

    @classmethod
    def from_config(cls, config: dict) -> "TopologyExtractor":
        """Create from configuration dictionary."""
        topo_cfg = config["model"]["topology"]
        return cls(
            max_homology_dim=topo_cfg.get("max_homology_dim", 1),
            resolution=tuple(
                topo_cfg.get("persistence_image_resolution", [10, 10])
            ),
            sigma=topo_cfg.get("persistence_image_sigma", 0.1),
            max_points=topo_cfg.get("max_points_subsample", 500),
            max_edge=topo_cfg.get("filtration_max_edge", 2.0),
        )
