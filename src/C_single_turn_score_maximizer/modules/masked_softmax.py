from typing import cast

import torch
from torch import nn


class MaskedSoftmax(nn.Module):
    """Softmax layer that applies a mask before softmax."""

    def __init__(self, mask_value: float = -float("inf")):
        super().__init__()
        self.mask_value = mask_value
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass applying masked softmax."""
        x = x.masked_fill(mask == 0, self.mask_value)
        return cast("torch.Tensor", self.softmax(x))
