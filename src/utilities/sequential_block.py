from typing import cast

import torch
from torch import nn


class SequentialBlock(nn.Module):  # noqa: D101
    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        self.network = nn.Sequential(*layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return cast("torch.Tensor", self.network(x))
