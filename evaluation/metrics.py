"""
Evaluation Metrics Module
=========================
Implements all evaluation metrics used in Meta-TTA-TSM experiments:

1. Fisher Divergence (primary metric) — measures quality of score estimation.
2. Maximum Mean Discrepancy (MMD) — distribution-level comparison.
3. Negative Log-Likelihood (NLL) — density estimation quality.
4. Structure Recovery (AUC-ROC, SHD) — for graphical model experiments.

Reference: Section 5.1 (Metrics) of the accompanying paper.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict
from sklearn.metrics import roc_auc_score


def fisher_divergence(
    score_predicted: np.ndarray,
    score_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Fisher divergence between predicted and true scores.

    D_F = E[||s_θ(x) - ∇log p(x)||²]

    This is the primary evaluation metric for score matching quality.

    Args:
        score_predicted: Predicted scores, shape (n, d).
        score_true: True scores (from oracle or ground truth), shape (n, d).
        mask: Optional binary mask, shape (n, d). If provided, only
              evaluates on observed dimensions.

    Returns:
        Fisher divergence (scalar).
    """
    diff = score_predicted - score_true

    if mask is not None:
        diff = diff * mask

    # Mean squared L2 norm of difference
    per_sample = np.sum(diff ** 2, axis=-1)
    return float(np.mean(per_sample))


def fisher_divergence_gaussian(
    score_predicted: np.ndarray,
    data: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Fisher divergence for Gaussian with known parameters.

    For a Gaussian N(μ, Σ), the true score is:
        s(x) = -Σ^{-1}(x - μ) = -P(x - μ)

    where P = Σ^{-1} is the precision matrix.

    Args:
        score_predicted: Predicted scores, shape (n, d).
        data: Data points, shape (n, d).
        mean: True mean, shape (d,).
        precision: True precision matrix, shape (d, d).
        mask: Optional binary mask.

    Returns:
        Fisher divergence.
    """
    # Compute true score: s(x) = -P(x - μ)
    centered = data - mean[np.newaxis, :]
    score_true = -centered @ precision.T

    return fisher_divergence(score_predicted, score_true, mask)


def mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
) -> float:
    """Compute Maximum Mean Discrepancy between two distributions.

    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Args:
        X: Samples from P, shape (n, d).
        Y: Samples from Q, shape (m, d).
        kernel: Kernel type ('rbf').
        bandwidth: Kernel bandwidth. Auto-selected if None.

    Returns:
        MMD value (scalar).
    """
    if bandwidth is None:
        # Median heuristic
        combined = np.vstack([X, Y])
        dists = np.sqrt(
            np.sum(
                (combined[:, np.newaxis, :] - combined[np.newaxis, :, :]) ** 2,
                axis=-1,
            )
        )
        bandwidth = float(np.median(dists[dists > 0]))
        bandwidth = max(bandwidth, 1e-5)

    gamma = 1.0 / (2 * bandwidth ** 2)

    def rbf_kernel(A, B):
        dists = np.sum(
            (A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=-1
        )
        return np.exp(-gamma * dists)

    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    n = X.shape[0]
    m = Y.shape[0]

    # Unbiased estimate
    mmd_sq = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        - 2 * K_XY.sum() / (n * m)
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    )

    return float(max(0, mmd_sq) ** 0.5)


def negative_log_likelihood(
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Compute negative log-likelihood under a Gaussian model.

    NLL = -E[log p(x)] = (d/2)log(2π) + (1/2)log|Σ| + (1/2)E[(x-μ)ᵀΣ⁻¹(x-μ)]

    Args:
        data: Data array, shape (n, d).
        mean: Mean vector, shape (d,).
        covariance: Covariance matrix, shape (d, d).

    Returns:
        Average NLL per sample.
    """
    n, d = data.shape

    # Log-determinant
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("inf")

    precision = np.linalg.inv(covariance)

    centered = data - mean[np.newaxis, :]
    quad_form = np.sum((centered @ precision) * centered, axis=-1)

    nll = 0.5 * (d * np.log(2 * np.pi) + logdet + np.mean(quad_form))

    return float(nll)


def structural_hamming_distance(
    adj_true: np.ndarray,
    adj_pred: np.ndarray,
) -> int:
    """Compute Structural Hamming Distance between adjacency matrices.

    SHD counts the number of edge additions, deletions, and reversals
    needed to transform the predicted graph into the true graph.

    Args:
        adj_true: True adjacency matrix, shape (d, d).
        adj_pred: Predicted adjacency matrix, shape (d, d).

    Returns:
        SHD (integer count of differences).
    """
    # Binarize
    true_binary = (np.abs(adj_true) > 1e-10).astype(int)
    pred_binary = (np.abs(adj_pred) > 1e-10).astype(int)

    # Only count upper triangle (undirected graph)
    diff = np.abs(true_binary - pred_binary)
    shd = int(np.sum(np.triu(diff, k=1)))

    return shd


def structure_recovery_metrics(
    precision_true: np.ndarray,
    precision_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute structure recovery metrics for graphical model evaluation.

    Args:
        precision_true: True precision matrix, shape (d, d).
        precision_pred: Estimated precision matrix, shape (d, d).

    Returns:
        Dictionary with 'auc_roc' and 'shd' metrics.
    """
    d = precision_true.shape[0]

    # Adjacency: non-zero off-diagonal entries
    adj_true = np.abs(precision_true) > 1e-10
    np.fill_diagonal(adj_true, False)

    adj_pred_scores = np.abs(precision_pred)
    np.fill_diagonal(adj_pred_scores, 0)

    # Flatten upper triangle for AUC-ROC
    upper_idx = np.triu_indices(d, k=1)
    y_true = adj_true[upper_idx].astype(int)
    y_scores = adj_pred_scores[upper_idx]

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.5  # All same class

    # SHD (threshold predicted precision for binary adjacency)
    threshold = np.median(adj_pred_scores[adj_pred_scores > 0])
    adj_pred_binary = (adj_pred_scores > threshold).astype(int)
    shd = structural_hamming_distance(adj_true.astype(int), adj_pred_binary)

    return {"auc_roc": float(auc), "shd": shd}
