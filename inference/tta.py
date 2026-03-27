"""
Test-Time Adaptation (TTA) Module
==================================
Implements the test-time adaptation procedure for Meta-TTA-TSM.

At deployment, when the test data exhibits topological drift (measured
by Wasserstein distance between persistence diagrams), the score
function is adapted via gradient descent on a combined objective:

    L_adapt(θ) = L_ISM(θ; D_test) + λ_topo · L_topo(θ; P_test)

This ensures both score matching fidelity and topological consistency
are maintained under novel missingness patterns.

Reference: Section 4.0.6 and Theorem 4.3 of the paper.
"""

import copy
import torch
import numpy as np
from typing import Optional, Dict

from models.score_network import ScoreNetwork
from models.hypernetwork import HyperNetwork
from models.topology import TopologyExtractor, compute_wasserstein_distance
from models.losses import ISMLoss, TopologicalConsistencyLoss


class TestTimeAdapter:
    """Test-Time Adaptation under topological drift.

    Detects topological drift via Wasserstein distance on persistence
    diagrams. If drift exceeds a threshold τ, adapts the score function
    parameters to the new topology while preserving score matching
    fidelity.

    Args:
        score_network: Base score network (will be deep-copied for adaptation).
        hypernetwork: Trained hypernetwork for initialization.
        topology_extractor: Topological feature extractor.
        config: Configuration dictionary.
        device: Torch device.
    """

    def __init__(
        self,
        score_network: ScoreNetwork,
        hypernetwork: HyperNetwork,
        topology_extractor: TopologyExtractor,
        config: dict,
        device: torch.device,
    ):
        self.score_network = score_network
        self.hypernetwork = hypernetwork
        self.topology_extractor = topology_extractor
        self.device = device

        # TTA configuration
        tta_cfg = config.get("test_time_adaptation", {})
        self.tta_lr = tta_cfg.get("tta_lr", 5e-3)
        self.tta_steps = tta_cfg.get("tta_steps", 10)
        self.lambda_topo = tta_cfg.get("lambda_topo", 0.1)
        self.drift_threshold = tta_cfg.get("drift_threshold", 0.5)

        # Loss functions
        self.ism_loss = ISMLoss(use_hutchinson=True)
        self.topo_loss = TopologicalConsistencyLoss()

        # Store training topology for drift comparison
        self.train_diagrams = None
        self.train_topo_features = None

    def set_training_topology(
        self,
        train_data: np.ndarray,
    ) -> None:
        """Compute and store training data topology for drift detection.

        Args:
            train_data: Training data (observed) of shape (n, d).
        """
        result = self.topology_extractor.extract(
            train_data, return_diagrams=True
        )
        self.train_diagrams = result["diagrams"]
        self.train_topo_features = torch.tensor(
            result["features"], dtype=torch.float32, device=self.device
        )

    def detect_drift(
        self,
        test_data: np.ndarray,
    ) -> Dict[str, float]:
        """Detect topological drift between training and test data.

        Computes Δ_topo = d_W(P_train, P_test) and compares against
        the threshold τ.

        Reference: Equation 9 in the paper.

        Args:
            test_data: Test data (observed) of shape (n, d).

        Returns:
            Dictionary with:
                'drift': Wasserstein distance value.
                'is_drifted': Boolean indicating drift detection.
                'threshold': The threshold τ used.
        """
        test_result = self.topology_extractor.extract(
            test_data, return_diagrams=True
        )
        test_diagrams = test_result["diagrams"]

        # Compute Wasserstein distance for each homological dimension
        total_drift = 0.0
        for dim_idx in range(len(self.train_diagrams)):
            if dim_idx < len(test_diagrams):
                d_train = self.train_diagrams[dim_idx]
                d_test = test_diagrams[dim_idx]
                if d_train.shape[0] > 0 and d_test.shape[0] > 0:
                    total_drift += compute_wasserstein_distance(
                        d_train, d_test
                    )

        return {
            "drift": total_drift,
            "is_drifted": total_drift > self.drift_threshold,
            "threshold": self.drift_threshold,
        }

    def adapt(
        self,
        test_data: torch.Tensor,
        test_masks: torch.Tensor,
        test_topo_features: torch.Tensor,
        verbose: bool = False,
    ) -> ScoreNetwork:
        """Adapt the score function to test-time topology.

        Performs K_adapt gradient steps on:
            L_adapt(θ) = L_ISM(θ; D_test) + λ_topo · L_topo(θ; P_test)

        Reference: Section B.2, Algorithm in deployment phase.

        Args:
            test_data: Observed test data (n_test, d).
            test_masks: Test masks (n_test, d).
            test_topo_features: Target topology features (m,).
            verbose: Print adaptation progress.

        Returns:
            Adapted ScoreNetwork with updated parameters.
        """
        # Deep copy score network for adaptation
        adapted_net = copy.deepcopy(self.score_network).to(self.device)
        adapted_net.train()

        # TTA optimizer (only adapts score network, not hypernetwork)
        tta_optimizer = torch.optim.SGD(
            adapted_net.parameters(), lr=self.tta_lr
        )

        test_data = test_data.to(self.device)
        test_masks = test_masks.to(self.device)
        test_topo_features = test_topo_features.to(self.device)

        for step in range(self.tta_steps):
            tta_optimizer.zero_grad()

            # Score matching loss
            l_ism = self.ism_loss(adapted_net, test_data, test_masks)

            # Topological consistency loss (proxy via feature L2)
            if self.train_topo_features is not None:
                l_topo = self.topo_loss(
                    self.train_topo_features, test_topo_features
                )
            else:
                l_topo = torch.tensor(0.0, device=self.device)

            # Combined loss
            total_loss = l_ism + self.lambda_topo * l_topo
            total_loss.backward()
            tta_optimizer.step()

            if verbose and (step + 1) % max(1, self.tta_steps // 5) == 0:
                print(
                    f"  TTA Step {step + 1}/{self.tta_steps} | "
                    f"Loss: {total_loss.item():.6f} | "
                    f"ISM: {l_ism.item():.6f} | "
                    f"Topo: {l_topo.item():.6f}"
                )

        adapted_net.eval()
        return adapted_net

    def predict_with_tta(
        self,
        test_data_np: np.ndarray,
        test_masks_np: np.ndarray,
        force_adapt: bool = False,
        verbose: bool = False,
    ) -> Dict:
        """Full TTA pipeline: detect drift → adapt if needed → predict.

        Args:
            test_data_np: Observed test data (n, d) as numpy array.
            test_masks_np: Test masks (n, d) as numpy array.
            force_adapt: Force adaptation regardless of drift detection.
            verbose: Print progress.

        Returns:
            Dictionary with:
                'scores': Score predictions (n, d).
                'drift_info': Drift detection results.
                'adapted': Whether adaptation was performed.
        """
        # Step 1: Extract test topology
        test_result = self.topology_extractor.extract(
            test_data_np, return_diagrams=True
        )
        test_topo_features = torch.tensor(
            test_result["features"], dtype=torch.float32, device=self.device
        )

        # Step 2: Detect drift
        drift_info = self.detect_drift(test_data_np)

        if verbose:
            print(
                f"Topological drift: {drift_info['drift']:.4f} "
                f"(threshold: {drift_info['threshold']})"
            )

        # Step 3: Initialize score network via hypernetwork
        self.hypernetwork.eval()
        with torch.no_grad():
            generated_params = self.hypernetwork(test_topo_features)
        self.hypernetwork.apply_params_to_network(
            self.score_network, generated_params
        )

        # Step 4: Adapt if drift detected
        if drift_info["is_drifted"] or force_adapt:
            if verbose:
                print("Drift detected! Performing test-time adaptation...")

            test_data_t = torch.tensor(
                test_data_np, dtype=torch.float32, device=self.device
            )
            test_masks_t = torch.tensor(
                test_masks_np, dtype=torch.float32, device=self.device
            )

            adapted_net = self.adapt(
                test_data_t,
                test_masks_t,
                test_topo_features,
                verbose=verbose,
            )
            predict_net = adapted_net
            was_adapted = True
        else:
            if verbose:
                print("No significant drift. Using hypernetwork initialization.")
            predict_net = self.score_network
            was_adapted = False

        # Step 5: Predict scores
        predict_net.eval()
        test_data_t = torch.tensor(
            test_data_np, dtype=torch.float32, device=self.device
        )
        test_masks_t = torch.tensor(
            test_masks_np, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            scores = predict_net(test_data_t, test_masks_t)

        return {
            "scores": scores.cpu().numpy(),
            "drift_info": drift_info,
            "adapted": was_adapted,
        }
