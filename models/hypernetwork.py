"""
Hypernetwork Module
===================
Implements the hypernetwork H_ϕ: R^m → Θ that maps topological features
(persistence images) to the parameters of a score network.

The hypernetwork enables topology-conditioned score functions: given
topological features f_T from the observed data manifold, it produces
score network parameters θ_T^(0) = H_ϕ(f_T) tailored to that topology.

Reference: Section 4.0.3, Equation 12-14 of the accompanying paper.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List, Optional, Dict

from models.score_network import ScoreNetwork


class HyperNetwork(nn.Module):
    """Hypernetwork H_ϕ: R^m → Θ for topology-conditioned score functions.

    Maps persistence image features to score network parameters using a
    chunked generation strategy. Instead of generating all parameters at
    once (which would be prohibitively large), it generates parameters
    for each layer of the score network separately.

    Args:
        topo_feature_dim: Dimension of topological feature vector f_T.
        target_network: A ScoreNetwork instance whose parameter structure
                        defines the output specification.
        hidden_dims: Hidden layer sizes for the hypernetwork.
        activation: Activation function name.
        use_spectral_norm: Apply spectral normalization to enforce
                          Lipschitz constraint (Assumption 5.3 in paper).
    """

    def __init__(
        self,
        topo_feature_dim: int,
        target_network: ScoreNetwork,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.topo_feature_dim = topo_feature_dim
        self.use_spectral_norm = use_spectral_norm

        activations = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "silu": nn.SiLU,
        }
        act_fn = activations.get(activation, nn.ReLU)

        # Build shared feature encoder for topological features
        encoder_layers = []
        prev_dim = topo_feature_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            encoder_layers.extend([linear, act_fn(), nn.LayerNorm(h_dim)])
            prev_dim = h_dim

        self.shared_encoder = nn.Sequential(*encoder_layers)
        self.encoder_out_dim = prev_dim

        # Create a parameter head for each target network parameter
        # This is the "chunked generation" approach
        self.param_heads = nn.ModuleDict()
        self.param_shapes = {}

        for name, param in target_network.named_parameters():
            param_size = param.numel()
            self.param_shapes[name] = param.shape

            # Each head: Linear → output of size matching target param
            head = nn.Linear(self.encoder_out_dim, param_size)
            if use_spectral_norm:
                head = spectral_norm(head)

            # Initialize heads to produce near-zero parameters
            # so initial score network output is close to zero
            if hasattr(head, "weight"):
                nn.init.normal_(head.weight, std=0.01)

            # Use sanitized key for ModuleDict (replace dots with dashes)
            safe_name = name.replace(".", "-")
            self.param_heads[safe_name] = head

        # Store name mapping
        self._name_map = {
            name.replace(".", "-"): name
            for name in self.param_shapes.keys()
        }

    def forward(self, topo_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate score network parameters from topological features.

        Args:
            topo_features: Topological feature vector f_T of shape (m,)
                          or (batch_size, m). If batched, generates
                          parameters for each item (used in meta-batches).

        Returns:
            Dictionary mapping parameter names to generated parameter
            tensors with correct shapes for the target score network.
        """
        # Encode topological features
        if topo_features.dim() == 1:
            topo_features = topo_features.unsqueeze(0)

        encoded = self.shared_encoder(topo_features)  # (B, encoder_out_dim)

        # Generate parameters via per-layer heads
        generated_params = {}
        for safe_name, head in self.param_heads.items():
            original_name = self._name_map[safe_name]
            target_shape = self.param_shapes[original_name]

            # Generate flat parameter vector and reshape
            flat_params = head(encoded)  # (B, param_size)

            if encoded.shape[0] == 1:
                # Single task: squeeze batch dimension
                generated_params[original_name] = flat_params.squeeze(0).view(
                    target_shape
                )
            else:
                # Batched: keep batch dimension
                generated_params[original_name] = flat_params.view(
                    encoded.shape[0], *target_shape
                )

        return generated_params

    def apply_params_to_network(
        self,
        score_network: ScoreNetwork,
        generated_params: Dict[str, torch.Tensor],
    ) -> None:
        """Apply generated parameters to a score network in-place.

        This uses the functional approach: it copies the generated
        parameters into the score network's parameter buffers.

        Args:
            score_network: Target ScoreNetwork to update.
            generated_params: Dictionary of generated parameters.
        """
        for name, param in score_network.named_parameters():
            if name in generated_params:
                param.data.copy_(generated_params[name])

    @classmethod
    def from_config(
        cls,
        config: dict,
        target_network: ScoreNetwork,
    ) -> "HyperNetwork":
        """Create a HyperNetwork from configuration.

        Args:
            config: Configuration dictionary.
            target_network: Score network whose parameters to generate.

        Returns:
            Configured HyperNetwork instance.
        """
        hn_cfg = config["model"]["hypernetwork"]
        return cls(
            topo_feature_dim=hn_cfg.get("topo_feature_dim", 100),
            target_network=target_network,
            hidden_dims=hn_cfg.get("hidden_dims", [512, 512]),
            activation=hn_cfg.get("activation", "relu"),
            use_spectral_norm=hn_cfg.get("use_spectral_norm", True),
        )
