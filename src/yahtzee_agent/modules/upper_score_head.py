from typing import cast

import torch
from torch import nn

from utilities.sequential_block import SequentialBlock

from .block import Block


class UpperScoreHead(nn.Module):
    """Head for upper score estimation."""

    def __init__(
        self,
        hidden_size: int,
        activation: type[nn.Module],
        use_layer_norm: bool,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            Block(hidden_size, hidden_size, 0.0, activation, use_layer_norm),
            nn.Linear(hidden_size, 1),
        ]
        self.network = SequentialBlock(*layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call method to enable direct calls to the head."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the final score of the upper section."""
        return cast("torch.Tensor", self.network(x))
