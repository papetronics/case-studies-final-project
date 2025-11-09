from typing import cast

from torch import Tensor, nn


class SequentialBlock(nn.Sequential):
    """A sequential block that enforces Tensor input and output types."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the sequential block."""
        return cast("Tensor", super().forward(x))

    def __call__(self, x: Tensor) -> Tensor:
        """Call method to enable direct calls to the block."""
        # base nn.Module.__call__ is typed as -> Any; we narrow it
        return cast("Tensor", super().__call__(x))
