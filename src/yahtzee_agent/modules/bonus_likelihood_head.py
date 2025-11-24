from typing import cast

import torch
from torch import nn

from utilities.sequential_block import SequentialBlock

from .block import Block


class BonusLikelihoodHead(nn.Module):
    """Head for bonus likelihood estimation."""

    def __init__(
        self,
        hidden_size: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        layers: list[nn.Module] = [
            Block(hidden_size, hidden_size, 0.0, activation),
            nn.Linear(hidden_size, 1),
            nn.ELU(),
        ]
        self.network = SequentialBlock(*layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call method to enable direct calls to the head."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value head."""
        return cast("torch.Tensor", self.network(x))
