import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A residual MLP block with skip connection added before normalization.

    Architecture: x -> Linear -> Activation -> (x + residual) -> LayerNorm -> Dropout

    This design helps prevent gradient decoherence in shared representations
    by providing a direct path for gradients to flow through the network.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float,
        activation: type[nn.Module] = nn.PReLU,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation()
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

        # Projection for residual connection if dimensions don't match
        self.projection = (
            nn.Linear(in_features, out_features) if in_features != out_features else None
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call method to enable direct calls to the block."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        # Compute residual path
        residual: torch.Tensor = x if self.projection is None else self.projection(x)

        # Main path: Linear -> Activation
        out: torch.Tensor = self.linear(x)
        out = self.activation(out)

        # Add residual connection before normalization
        out = out + residual

        # Normalize and optionally dropout
        out = self.norm(out)
        if self.dropout is not None:
            out = self.dropout(out)

        return out
