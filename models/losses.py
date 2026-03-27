"""
Loss Functions Module
=====================
Implements all loss functions used in Meta-TTA-TSM:

1. Implicit Score Matching (ISM) loss — the primary score matching objective.
2. Topological consistency loss — Wasserstein distance on persistence diagrams.
3. Combined adaptation loss — ISM + λ_topo × topological consistency.

Reference: Equations 5, 15, and Section 4.0.4 / B.2 of the accompanying paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from models.score_network import ScoreNetwork


class ISMLoss(nn.Module):
    """Implicit Score Matching (ISM) Loss.

    Implements Hyv\"arinen's implicit score matching objective:

        L_ISM(θ) = E_x [ (1/2) ||s_θ(x)||² + tr(∇_x s_θ(x)) ]

    This provides an equivalent objective to minimizing the Fisher
    divergence without requiring access to the true score.

    The trace of the Jacobian is computed via the Hutchinson estimator
    (random projections) for efficiency in high dimensions.

    Reference: Equation 5 in the paper.
    """

    def __init__(self, use_hutchinson: bool = True, num_slices: int = 1):
        """Initialize ISM loss.

        Args:
            use_hutchinson: Use Hutchinson's trace estimator (stochastic).
                           If False, compute exact Jacobian trace (expensive).
            num_slices: Number of random projections for Hutchinson estimator.
        """
        super().__init__()
        self.use_hutchinson = use_hutchinson
        self.num_slices = num_slices

    def forward(
        self,
        score_network: ScoreNetwork,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ISM loss on a batch of samples.

        Args:
            score_network: The score network s_θ.
            x: Input data of shape (batch_size, d). Missing values
               should be zeroed out.
            mask: Binary mask of shape (batch_size, d).
                  1 = observed, 0 = missing.

        Returns:
            Scalar ISM loss value.
        """
        x = x.requires_grad_(True)
        scores = score_network(x, mask)

        # Term 1: (1/2) ||s_θ(x)||²
        if mask is not None:
            # Only count observed dimensions
            score_norm = 0.5 * (scores * mask).pow(2).sum(dim=-1).mean()
        else:
            score_norm = 0.5 * scores.pow(2).sum(dim=-1).mean()

        # Term 2: tr(∇_x s_θ(x))
        if self.use_hutchinson:
            jacobian_trace = self._hutchinson_trace(scores, x, mask)
        else:
            jacobian_trace = self._exact_trace(scores, x, mask)

        loss = score_norm + jacobian_trace
        return loss

    def _hutchinson_trace(
        self,
        scores: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate Jacobian trace via Hutchinson's estimator.

        Uses random Rademacher vectors v to estimate:
            tr(J) ≈ E_v[v^T J v]

        Args:
            scores: Score network output, shape (batch_size, d).
            x: Input requiring gradients, shape (batch_size, d).
            mask: Optional binary mask.

        Returns:
            Estimated trace, scalar.
        """
        trace_estimate = 0.0

        for _ in range(self.num_slices):
            # Random Rademacher vector
            v = torch.randint(
                0, 2, size=scores.shape, device=scores.device
            ).float() * 2 - 1

            if mask is not None:
                v = v * mask

            # Compute v^T J v via vector-Jacobian product
            vjp = torch.autograd.grad(
                outputs=scores,
                inputs=x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]

            trace_estimate += (v * vjp).sum(dim=-1).mean()

        return trace_estimate / self.num_slices

    def _exact_trace(
        self,
        scores: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute exact Jacobian trace (sum of diagonal entries).

        More expensive but exact. Computes ∂s_θ(x)_j / ∂x_j for each j.

        Args:
            scores: Score network output, shape (batch_size, d).
            x: Input requiring gradients, shape (batch_size, d).
            mask: Optional binary mask.

        Returns:
            Exact trace, scalar.
        """
        d = scores.shape[-1]
        trace = 0.0

        for j in range(d):
            if mask is not None:
                # Skip unobserved dimensions
                if mask[:, j].sum() == 0:
                    continue

            grad_j = torch.autograd.grad(
                outputs=scores[:, j].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
            )[0][:, j]

            if mask is not None:
                trace += (grad_j * mask[:, j]).mean()
            else:
                trace += grad_j.mean()

        return trace


class TopologicalConsistencyLoss(nn.Module):
    """Topological Consistency Loss.

    Measures the Wasserstein distance between the persistence diagram
    of generated samples and a target persistence diagram. This loss
    ensures that adapted score functions produce samples with the
    correct topological structure.

    L_topo(θ; P_test) = d_W(P_gen(θ), P_test)

    Reference: Definition B.2 in the paper.
    """

    def __init__(self, p: int = 2):
        """Initialize topological consistency loss.

        Args:
            p: Exponent for Wasserstein distance.
        """
        super().__init__()
        self.p = p

    def forward(
        self,
        features_current: torch.Tensor,
        features_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute topological consistency loss.

        Since Wasserstein distance on persistence diagrams is not
        differentiable, we use a proxy: L2 distance between persistence
        image feature vectors. This is a common differentiable
        approximation used in practice.

        Args:
            features_current: Persistence image features of current
                            samples/model, shape (m,).
            features_target: Persistence image features of target
                           topology, shape (m,).

        Returns:
            Scalar topological consistency loss.
        """
        return torch.norm(features_current - features_target, p=self.p)


class CombinedAdaptationLoss(nn.Module):
    """Combined Adaptation Loss for Test-Time Adaptation.

    L_adapt(θ) = L_ISM(θ; D_test) + λ_topo · L_topo(θ; P_test)

    Balances score matching fidelity with topological consistency
    during test-time adaptation.

    Reference: Section B.2, Equation in TTA objective.
    """

    def __init__(
        self,
        lambda_topo: float = 0.1,
        use_hutchinson: bool = True,
    ):
        """Initialize combined loss.

        Args:
            lambda_topo: Weight for topological consistency term.
            use_hutchinson: Use Hutchinson estimator for ISM trace.
        """
        super().__init__()
        self.lambda_topo = lambda_topo
        self.ism_loss = ISMLoss(use_hutchinson=use_hutchinson)
        self.topo_loss = TopologicalConsistencyLoss()

    def forward(
        self,
        score_network: ScoreNetwork,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        topo_features_current: torch.Tensor,
        topo_features_target: torch.Tensor,
    ) -> dict:
        """Compute combined adaptation loss.

        Args:
            score_network: Score network to evaluate.
            x: Input data of shape (batch_size, d).
            mask: Binary mask of shape (batch_size, d).
            topo_features_current: Current topology features.
            topo_features_target: Target topology features.

        Returns:
            Dictionary with 'total', 'ism', and 'topo' loss values.
        """
        l_ism = self.ism_loss(score_network, x, mask)
        l_topo = self.topo_loss(topo_features_current, topo_features_target)

        total = l_ism + self.lambda_topo * l_topo

        return {
            "total": total,
            "ism": l_ism.detach(),
            "topo": l_topo.detach(),
        }
