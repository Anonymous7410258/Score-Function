"""
Unit Tests for Meta-TTA-TSM
============================
Tests core components: score network, hypernetwork, topology extractor,
loss functions, data generators, and missingness masks.

Run:
    python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import torch
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScoreNetwork:
    """Tests for the score network module."""

    def test_forward_shape(self):
        """Score network output should match input dimensionality."""
        from models.score_network import ScoreNetwork

        net = ScoreNetwork(input_dim=20, hidden_dims=[64, 32])
        x = torch.randn(16, 20)
        out = net(x)
        assert out.shape == (16, 20), f"Expected (16, 20), got {out.shape}"

    def test_masked_forward(self):
        """Scores for missing dimensions should be zeroed out."""
        from models.score_network import ScoreNetwork

        net = ScoreNetwork(input_dim=10, hidden_dims=[32])
        x = torch.randn(8, 10)
        mask = torch.ones(8, 10)
        mask[:, 5:] = 0  # Last 5 dims missing

        out = net(x, mask)
        assert torch.all(out[:, 5:] == 0), "Missing dims should be zero"

    def test_residual_mode(self):
        """Residual and non-residual modes should both work."""
        from models.score_network import ScoreNetwork

        for residual in [True, False]:
            net = ScoreNetwork(
                input_dim=10, hidden_dims=[32, 16], use_residual=residual
            )
            x = torch.randn(4, 10)
            out = net(x)
            assert out.shape == (4, 10)

    def test_from_config(self):
        """Score network should be constructible from config dict."""
        from models.score_network import ScoreNetwork

        config = {
            "model": {
                "score_network": {
                    "input_dim": 30,
                    "hidden_dims": [64, 64],
                    "activation": "elu",
                    "use_residual": True,
                    "dropout": 0.0,
                }
            }
        }
        net = ScoreNetwork.from_config(config)
        assert net.input_dim == 30

    def test_param_count(self):
        """Parameter count should be positive and deterministic."""
        from models.score_network import ScoreNetwork

        net = ScoreNetwork(input_dim=10, hidden_dims=[32])
        assert net.get_num_params() > 0


class TestHyperNetwork:
    """Tests for the hypernetwork module."""

    def test_parameter_generation(self):
        """Hypernetwork should generate valid parameters for score network."""
        from models.score_network import ScoreNetwork
        from models.hypernetwork import HyperNetwork

        score_net = ScoreNetwork(input_dim=10, hidden_dims=[32])
        hyper_net = HyperNetwork(
            topo_feature_dim=50,
            target_network=score_net,
            hidden_dims=[64],
            use_spectral_norm=False,
        )

        topo_features = torch.randn(50)
        params = hyper_net(topo_features)

        assert isinstance(params, dict), "Should return dictionary"
        assert len(params) > 0, "Should generate at least one parameter"

        # Check shapes match score network
        for name, param in score_net.named_parameters():
            assert name in params, f"Missing parameter: {name}"
            assert (
                params[name].shape == param.shape
            ), f"Shape mismatch for {name}"

    def test_apply_params(self):
        """Generated params should be applicable to score network."""
        from models.score_network import ScoreNetwork
        from models.hypernetwork import HyperNetwork

        score_net = ScoreNetwork(input_dim=10, hidden_dims=[32])
        hyper_net = HyperNetwork(
            topo_feature_dim=50,
            target_network=score_net,
            hidden_dims=[64],
            use_spectral_norm=False,
        )

        topo_features = torch.randn(50)
        params = hyper_net(topo_features)
        hyper_net.apply_params_to_network(score_net, params)

        # Verify network still works
        x = torch.randn(4, 10)
        out = score_net(x)
        assert out.shape == (4, 10)


class TestTopology:
    """Tests for the topology extraction module."""

    def test_persistence_image(self):
        """Persistence image should produce correct shape."""
        from models.topology import compute_persistence_image

        # Simple persistence diagram
        diagram = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 1.5]])
        pi = compute_persistence_image(diagram, resolution=(5, 5), sigma=0.1)
        assert pi.shape == (25,), f"Expected (25,), got {pi.shape}"

    def test_empty_diagram(self):
        """Empty persistence diagram should return zero vector."""
        from models.topology import compute_persistence_image

        diagram = np.zeros((0, 2))
        pi = compute_persistence_image(diagram, resolution=(5, 5))
        assert pi.shape == (25,)
        assert np.allclose(pi, 0.0)

    def test_topology_extractor(self):
        """TopologyExtractor should produce valid feature vectors."""
        from models.topology import TopologyExtractor

        extractor = TopologyExtractor(
            max_homology_dim=1,
            resolution=(5, 5),
            sigma=0.1,
            max_points=50,
        )

        # Simple point cloud
        X = np.random.randn(100, 3)
        result = extractor.extract(X, return_diagrams=True)

        assert "features" in result
        assert "diagrams" in result
        assert result["features"].shape == (extractor.feature_dim,)

    def test_topology_extractor_torch(self):
        """TopologyExtractor should work with PyTorch tensors."""
        from models.topology import TopologyExtractor

        extractor = TopologyExtractor(
            max_homology_dim=0,
            resolution=(3, 3),
            max_points=30,
        )

        X = torch.randn(50, 5)
        result = extractor.extract_torch(X)
        assert isinstance(result["features"], torch.Tensor)

    def test_wasserstein_distance(self):
        """Wasserstein distance should be non-negative and symmetric."""
        from models.topology import compute_wasserstein_distance

        d1 = np.array([[0.0, 1.0], [0.5, 2.0]])
        d2 = np.array([[0.0, 1.5], [0.3, 1.8]])

        dist = compute_wasserstein_distance(d1, d2)
        assert dist >= 0, "Distance should be non-negative"

        dist_rev = compute_wasserstein_distance(d2, d1)
        assert abs(dist - dist_rev) < 1e-6, "Distance should be symmetric"


class TestLosses:
    """Tests for loss function module."""

    def test_ism_loss_computes(self):
        """ISM loss should compute without error."""
        from models.score_network import ScoreNetwork
        from models.losses import ISMLoss

        net = ScoreNetwork(input_dim=5, hidden_dims=[16])
        loss_fn = ISMLoss(use_hutchinson=True)

        x = torch.randn(8, 5)
        loss = loss_fn(net, x)

        assert loss.shape == (), "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_ism_loss_with_mask(self):
        """ISM loss should work with masks."""
        from models.score_network import ScoreNetwork
        from models.losses import ISMLoss

        net = ScoreNetwork(input_dim=5, hidden_dims=[16])
        loss_fn = ISMLoss(use_hutchinson=True)

        x = torch.randn(8, 5)
        mask = torch.ones(8, 5)
        mask[:, 3:] = 0

        loss = loss_fn(net, x, mask)
        assert torch.isfinite(loss)

    def test_topological_consistency_loss(self):
        """Topological consistency loss should be non-negative."""
        from models.losses import TopologicalConsistencyLoss

        loss_fn = TopologicalConsistencyLoss()
        f1 = torch.randn(50)
        f2 = torch.randn(50)

        loss = loss_fn(f1, f2)
        assert loss >= 0, "Loss should be non-negative"

    def test_combined_loss(self):
        """Combined loss should return all three components."""
        from models.score_network import ScoreNetwork
        from models.losses import CombinedAdaptationLoss

        net = ScoreNetwork(input_dim=5, hidden_dims=[16])
        loss_fn = CombinedAdaptationLoss(lambda_topo=0.1)

        x = torch.randn(4, 5)
        mask = torch.ones(4, 5)
        f1 = torch.randn(20)
        f2 = torch.randn(20)

        result = loss_fn(net, x, mask, f1, f2)
        assert "total" in result
        assert "ism" in result
        assert "topo" in result


class TestDatasets:
    """Tests for dataset generation."""

    def test_gaussian_data(self):
        """Gaussian data should have correct shape."""
        from data.datasets import generate_gaussian_data

        data = generate_gaussian_data(n_samples=100, dim=10, seed=42)
        assert data.shape == (100, 10)
        assert data.dtype == np.float32

    def test_truncated_gaussian(self):
        """Truncated Gaussian data should be positive."""
        from data.datasets import generate_gaussian_data

        data = generate_gaussian_data(
            n_samples=100, dim=5, truncated=True, seed=42
        )
        assert np.all(data > 0), "Truncated data should be positive"

    def test_ica_data(self):
        """ICA data should have correct shape."""
        from data.datasets import generate_ica_data

        data = generate_ica_data(n_samples=100, dim=10, seed=42)
        assert data.shape == (100, 10)

    def test_ggm_data(self):
        """GGM data should return data and adjacency."""
        from data.datasets import generate_ggm_data

        data, adj = generate_ggm_data(n_samples=100, dim=10, seed=42)
        assert data.shape == (100, 10)
        assert adj.shape == (10, 10)
        assert np.all(np.diag(adj) == 0), "Diagonal should be zero"


class TestMissingness:
    """Tests for missingness mask generation."""

    def test_mcar_mask_shape(self):
        """MCAR mask should have same shape as data."""
        from data.missingness import generate_mcar_mask

        mask = generate_mcar_mask((100, 10), missing_rate=0.3, seed=42)
        assert mask.shape == (100, 10)
        assert mask.dtype == np.float32

    def test_mcar_rate(self):
        """Actual missingness rate should approximate target."""
        from data.missingness import generate_mcar_mask

        mask = generate_mcar_mask((10000, 50), missing_rate=0.4, seed=42)
        actual_rate = 1.0 - mask.mean()
        assert abs(actual_rate - 0.4) < 0.02, (
            f"Rate {actual_rate:.3f} too far from 0.4"
        )

    def test_mcar_at_least_one_observed(self):
        """Each sample should have at least one observed dimension."""
        from data.missingness import generate_mcar_mask

        mask = generate_mcar_mask((100, 10), missing_rate=0.9, seed=42)
        assert np.all(mask.sum(axis=1) >= 1)

    def test_invalid_rate(self):
        """Should raise error for invalid missingness rate."""
        from data.missingness import generate_mcar_mask

        with pytest.raises(ValueError):
            generate_mcar_mask((10, 5), missing_rate=1.0)

    def test_task_sampler(self):
        """MissingnessTaskSampler should produce valid tasks."""
        from data.missingness import MissingnessTaskSampler

        data = np.random.randn(500, 10).astype(np.float32)
        sampler = MissingnessTaskSampler(data, samples_per_task=50, seed=42)

        task = sampler.sample_task()
        assert "observed_data" in task
        assert "masks" in task
        assert "full_data" in task
        assert task["observed_data"].shape == (50, 10)


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_fisher_divergence(self):
        """Fisher divergence should be non-negative."""
        from evaluation.metrics import fisher_divergence

        s_pred = np.random.randn(100, 10)
        s_true = np.random.randn(100, 10)
        fd = fisher_divergence(s_pred, s_true)
        assert fd >= 0

    def test_fisher_divergence_self(self):
        """Fisher divergence of identical scores should be zero."""
        from evaluation.metrics import fisher_divergence

        s = np.random.randn(100, 10)
        fd = fisher_divergence(s, s)
        assert abs(fd) < 1e-10

    def test_mmd(self):
        """MMD should be non-negative."""
        from evaluation.metrics import mmd

        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5) + 1.0
        d = mmd(X, Y)
        assert d >= 0

    def test_shd(self):
        """SHD should be non-negative integer."""
        from evaluation.metrics import structural_hamming_distance

        A = np.eye(5)
        B = np.eye(5)
        B[0, 1] = 1
        B[1, 0] = 1
        shd = structural_hamming_distance(A, B)
        assert isinstance(shd, int)
        assert shd >= 0


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_standardize(self):
        """Standardized data should have ~zero mean and ~unit var."""
        from utils.preprocessing import standardize_data

        data = np.random.randn(1000, 5) * 3 + 2
        std_data, stats = standardize_data(data)
        assert abs(std_data.mean()) < 0.1
        assert abs(std_data.std() - 1.0) < 0.1

    def test_normalize(self):
        """Normalized data should be in [0, 1] range."""
        from utils.preprocessing import normalize_data

        data = np.random.randn(100, 5)
        norm_data, stats = normalize_data(data)
        assert norm_data.min() >= -1e-6
        assert norm_data.max() <= 1.0 + 1e-6


class TestReproducibility:
    """Tests for reproducibility utilities."""

    def test_set_seed(self):
        """Setting same seed should produce same random numbers."""
        from utils.reproducibility import set_seed

        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b), "Same seed should produce same output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
