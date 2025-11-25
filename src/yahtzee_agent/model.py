from collections.abc import Generator
from enum import Enum

import numpy as np
import torch
from torch import nn

from environments.full_yahtzee_env import ActionType, Observation
from src.yahtzee_agent.modules.upper_score_head import UpperScoreHead
from utilities.activation_functions import ActivationFunction, ActivationFunctionName
from utilities.scoring_helper import NUMBER_OF_DICE
from utilities.sequential_block import SequentialBlock

from .features import PhiFeature
from .modules import Block, RollingHead, ScoringHead, ValueHead


class RollingActionRepresentation(str, Enum):
    """Representation for rolling actions."""

    BERNOULLI = "bernoulli"  # 5 independent binary decisions (one per die)
    CATEGORICAL = "categorical"  # Single choice from 32 possible masks


def all_dice_masks() -> Generator[ActionType, None, None]:
    """Generate all possible dice hold masks (5 dice, each can be held or not)."""
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                for l in [0, 1]:  # noqa: E741
                    for m in [0, 1]:
                        yield (i, j, k, l, m)


DICE_MASKS = list(all_dice_masks())
assert len(DICE_MASKS) == 32  # 2^5 possible maskss  # noqa: PLR2004


def get_input_dimensions(features: list[PhiFeature]) -> int:
    """Calculate the input dimension for the model based on features.

    Model inputs are determined by the sum of all feature dimensions.
    Features should be added in the same order they're concatenated in phi().
    """
    return sum(f.size for f in features)


def phi(
    observation: Observation,
    features: list[PhiFeature],
    device: torch.device,
) -> torch.Tensor:
    """Convert observation dictionary to input tensor for the model.

    Simply computes all features in order and concatenates them.
    """
    # Compute all features in order
    feature_vectors = [feature.compute(observation) for feature in features]

    # Concatenate all feature vectors
    input_vector = np.concatenate(feature_vectors)
    return torch.FloatTensor(input_vector).to(device)


def sample_action(
    rolling_logits: torch.Tensor,
    scoring_probs: torch.Tensor,
    value_est: torch.Tensor,
    rolling_action_representation: RollingActionRepresentation,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Sample an action given logits (rolling probs, scoring probs, and value estimate)."""
    rolling_dist: torch.distributions.Distribution
    if rolling_action_representation == RollingActionRepresentation.BERNOULLI:
        rolling_dist = torch.distributions.Bernoulli(logits=rolling_logits)
        rolling_tensor = rolling_dist.sample()
        rolling_log_prob = rolling_dist.log_prob(rolling_tensor).sum(dim=-1)
    else:  # CATEGORICAL
        rolling_dist = torch.distributions.Categorical(logits=rolling_logits)
        rolling_tensor = rolling_dist.sample()
        rolling_log_prob = rolling_dist.log_prob(rolling_tensor).sum()

    scoring_dist = torch.distributions.Categorical(scoring_probs)
    scoring_tensor = scoring_dist.sample()
    scoring_log_prob = scoring_dist.log_prob(scoring_tensor).sum()

    return (rolling_tensor, scoring_tensor), (rolling_log_prob, scoring_log_prob), value_est


def select_action(
    rolling_logits: torch.Tensor,
    scoring_probs: torch.Tensor,
    rolling_action_representation: RollingActionRepresentation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select deterministic action using argmax/threshold (for validation/testing)."""
    if rolling_action_representation == RollingActionRepresentation.BERNOULLI:
        # Threshold at 0.5: keep dice with prob > 0.5
        sigmoided = torch.nn.functional.sigmoid(rolling_logits)
        rolling_tensor = (sigmoided > 0.5).float()  # noqa: PLR2004
    else:  # CATEGORICAL
        rolling_tensor = rolling_logits.argmax(dim=-1)

    scoring_tensor = scoring_probs.argmax(dim=-1)

    return (rolling_tensor, scoring_tensor)


def convert_rolling_action_to_hold_mask(
    rolling_action: torch.Tensor,
    rolling_action_representation: RollingActionRepresentation,
) -> np.ndarray:
    """Convert a rolling action tensor to a numpy hold mask.

    For Categorical: looks up the mask in DICE_MASKS
    For Bernoulli: directly converts to boolean numpy array
    """
    if rolling_action_representation == RollingActionRepresentation.CATEGORICAL:
        return np.array(DICE_MASKS[int(rolling_action.item())], dtype=bool)
    else:  # BERNOULLI
        return rolling_action.cpu().numpy().astype(bool)


class CouldNotFindCategoryMaskFeatureError(Exception):
    """Exception raised when the 'available_categories' feature is not found in the model features."""

    def __init__(self) -> None:
        super().__init__("Could not find 'available_categories' feature in the model features.")


class YahtzeeAgent(nn.Module):
    """Neural network model for maximizing score in a single turn of Yahtzee."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_size: int,
        num_hidden: int,
        dropout_rate: float,
        activation_function: ActivationFunctionName,
        features: list[PhiFeature],
        rolling_action_representation: RollingActionRepresentation | str,
        he_kaiming_initialization: bool,
    ):
        super().__init__()

        activation = ActivationFunction[activation_function].value

        self.dropout_rate = dropout_rate
        # Convert string to enum if needed
        self.rolling_action_representation = (
            RollingActionRepresentation(rolling_action_representation)
            if isinstance(rolling_action_representation, str)
            else rolling_action_representation
        )

        self.features: list[PhiFeature] = features

        # Find the index of the available_categories feature to extract the mask
        self.available_categories_idx: int | None = None
        current_idx = 0
        for feature in features:
            if feature.name == "available_categories":
                self.available_categories_idx = current_idx
                break
            current_idx += feature.size

        input_size = get_input_dimensions(self.features)

        ## Model outputs:
        #   - Rolling action:
        #       - Bernoulli: [5] - Probability of re-rolling each die
        #       - Categorical: [32] - Probability of selecting a particular dice mask
        #   - Scoring probabilities [13]: Probability of selecting each scoring category
        if self.rolling_action_representation == RollingActionRepresentation.BERNOULLI:
            dice_output_size = NUMBER_OF_DICE  # 5 independent binary decisions
        else:  # CATEGORICAL
            dice_output_size = len(DICE_MASKS)  # 32 possible masks
        scoring_output_size = 13

        layers = [Block(input_size, hidden_size, dropout_rate, activation)]
        for _ in range(num_hidden - 2):
            layers.append(Block(hidden_size, hidden_size, dropout_rate, activation))  # noqa: PERF401

        self.network = SequentialBlock(*layers)

        self.action_spine = Block(hidden_size, hidden_size, dropout_rate, activation)

        self.rolling_head = RollingHead(hidden_size, dice_output_size, dropout_rate, activation).to(
            self.device
        )
        self.scoring_head = ScoringHead(
            hidden_size, scoring_output_size, dropout_rate, activation
        ).to(self.device)
        self.value_head = ValueHead(hidden_size, activation).to(self.device)

        self.bonus_likelihood_head = UpperScoreHead(hidden_size, activation).to(self.device)

        # Initialize weights for better behavior at high learning rates
        if he_kaiming_initialization:
            self._initialize_weights()

    @staticmethod
    def _init_kaiming_linear(module: nn.Module) -> None:
        """Initialize linear layers with Kaiming normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _init_final_linear(module: nn.Module) -> None:
        """Initialize final layer with small orthogonal gain."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _initialize_weights(self) -> None:
        """Initialize network weights for high learning rate stability.

        Uses Kaiming fan-in initialization (good for ReLU/SiLU) for hidden layers
        and orthogonal initialization with small gains (0.01) for output layers.
        """
        # Initialize all hidden layers with Kaiming
        self.network.apply(self._init_kaiming_linear)
        self.action_spine.apply(self._init_kaiming_linear)

        # Initialize head hidden layers (all but last)
        for head in [
            self.rolling_head,
            self.scoring_head,
            self.value_head,
            self.bonus_likelihood_head,
        ]:
            modules_list = list(head.modules())
            for module in modules_list[:-1]:
                self._init_kaiming_linear(module)

        # Initialize final layers with small orthogonal gain
        # Find the last Linear layer in each head and apply final initialization
        for head in [
            self.rolling_head,
            self.scoring_head,
            self.value_head,
            self.bonus_likelihood_head,
        ]:
            for module in reversed(list(head.modules())):
                if isinstance(module, nn.Linear):
                    self._init_final_linear(module)
                    break

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call method to enable direct calls to the model."""
        return self.forward(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        spine = self.network(x)

        rolling_output = self.rolling_head(spine)

        # Extract available_categories mask from input
        if self.available_categories_idx is not None:
            # Extract the 13-element mask from the input features
            mask = x[:, self.available_categories_idx : self.available_categories_idx + 13]
        else:
            raise CouldNotFindCategoryMaskFeatureError()

        scoring_output = self.scoring_head(spine, mask)
        value_output = self.value_head(spine)
        bonus_likelihood_output = self.bonus_likelihood_head(spine)

        return (
            rolling_output.squeeze(0),
            scoring_output.squeeze(0),
            value_output.squeeze(0),
            bonus_likelihood_output.squeeze(0),
        )
