from enum import Enum
from typing import Literal

from torch import nn


class ActivationFunction(Enum):
    """Supported activation functions for the neural network."""

    ReLU = nn.ReLU
    GELU = nn.GELU
    CELU = nn.CELU
    PReLU = nn.PReLU
    ELU = nn.ELU
    Tanh = nn.Tanh
    LeakyReLU = nn.LeakyReLU
    Softplus = nn.Softplus
    Softsign = nn.Softsign
    Mish = nn.Mish
    Swish = nn.SiLU
    SeLU = nn.SELU


ActivationFunctionName = Literal[
    "ReLU",
    "GELU",
    "CELU",
    "PReLU",
    "ELU",
    "Tanh",
    "LeakyReLU",
    "Softplus",
    "Softsign",
    "Mish",
    "Swish",
    "SeLU",
]
