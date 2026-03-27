"""
Score Network Module
====================
Implements the parameterized score function s_θ: R^d → R^d that approximates
the gradient of the log-density (score) of the data distribution.

The architecture is a multi-layer perceptron with optional residual connections,
designed to handle partial observations via masking.

Reference: Section 4.0.3 of the accompanying paper.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and activation.

    Applies: output = activation(W2 · activation(W1 · x + b1) + b2) + x
    When input/output dims differ, a projection shortcut is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projection shortcut if dimensions change
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal for stable training."""
        for module in [self.linear1, self.linear2]:
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        return self.activation(out + residual)


class ScoreNetwork(nn.Module):
    """Parameterized score function s_θ: R^d → R^d.

    Maps (possibly masked) input data to score estimates. The network
    supports masked inputs where missing dimensions are set to zero.

    Architecture:
        Input → [ResidualBlock | Linear+Activation] × L → Linear → Output

    Args:
        input_dim: Data dimensionality d.
        hidden_dims: List of hidden layer sizes.
        activation: Activation function name ('elu', 'relu', 'silu').
        use_residual: Whether to use residual connections.
        dropout: Dropout probability.
    """

    ACTIVATIONS = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "elu",
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(self.ACTIVATIONS.keys())}"
            )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        act_fn = self.ACTIVATIONS[activation]

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            if use_residual:
                layers.append(
                    ResidualBlock(prev_dim, hidden_dim, act_fn(), dropout)
                )
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer: maps to score space R^d
        self.output_layer = nn.Linear(prev_dim, input_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute score estimate s_θ(x).

        Args:
            x: Input tensor of shape (batch_size, d). Missing values
               should be set to zero.
            mask: Optional binary mask of shape (batch_size, d) where
                  1 = observed, 0 = missing. If provided, output scores
                  for missing dimensions are zeroed out.

        Returns:
            Score estimate of shape (batch_size, d).
        """
        features = self.feature_extractor(x)
        scores = self.output_layer(features)

        # Zero out scores for missing dimensions
        if mask is not None:
            scores = scores * mask

        return scores

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict) -> "ScoreNetwork":
        """Create a ScoreNetwork from a configuration dictionary.

        Args:
            config: Dictionary with keys matching constructor arguments.
                    Expected under config['model']['score_network'].

        Returns:
            Configured ScoreNetwork instance.
        """
        sn_cfg = config["model"]["score_network"]
        return cls(
            input_dim=sn_cfg["input_dim"],
            hidden_dims=sn_cfg.get("hidden_dims", [256, 256, 128]),
            activation=sn_cfg.get("activation", "elu"),
            use_residual=sn_cfg.get("use_residual", True),
            dropout=sn_cfg.get("dropout", 0.0),
        )
