from typing import cast

import torch
from torch import nn

from utilities.sequential_block import SequentialBlock

from .block import Block
from .masked_softmax import MaskedSoftmax


class ScoringHead(nn.Module):
    """Head for scoring category probabilities."""

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
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = SequentialBlock(*layers)
        self.masked_softmax = MaskedSoftmax()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Call method to enable direct calls to the head."""
        return self.forward(x, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the scoring head with masked softmax."""
        logits = self.network(x)
        return cast("torch.Tensor", self.masked_softmax(logits, mask))
