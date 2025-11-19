from typing import cast

import torch
from torch import nn

from utilities.sequential_block import SequentialBlock

from .block import Block


class RollingHead(nn.Module):
    """Head for rolling action probabilities."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dropout_rate: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        layers: list[nn.Module] = [
            Block(hidden_size, hidden_size, dropout_rate, activation),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        layers.extend([nn.Linear(hidden_size, output_size), nn.Sigmoid()])
        self.network = SequentialBlock(*layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call method to enable direct calls to the head."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the rolling head."""
        return cast("torch.Tensor", self.network(x))
