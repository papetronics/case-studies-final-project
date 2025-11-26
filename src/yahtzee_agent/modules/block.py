import torch
from torch import nn

from utilities.sequential_block import SequentialBlock


class Block(nn.Module):
    """A basic MLP block with Linear, Activation, LayerNorm, and optional Dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float,
        activation: type[nn.Module],
        use_layer_norm: bool,
    ):
        super().__init__()
        layers = [
            nn.Linear(in_features, out_features),
            activation(),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_features))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

        self.network: SequentialBlock = SequentialBlock(*layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call method to enable direct calls to the block."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        output: torch.Tensor = self.network(x)
        return output
