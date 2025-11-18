from enum import IntEnum

import numpy as np
import torch
from torch import nn

from environments.full_yahtzee_env import FINAL_ROLL, Observation
from utilities.activation_functions import ActivationFunction, ActivationFunctionName
from utilities.scoring_helper import (
    NUMBER_OF_CATEGORIES,
    NUMBER_OF_DICE,
    NUMBER_OF_DICE_SIDES,
    YAHTZEE_SCORE,
    ScoreCategory,
    get_all_scores,
)
from utilities.sequential_block import SequentialBlock

from .modules import Block, RollingHead, ScoringHead, ValueHead


class BonusFlags(IntEnum):
    """Flags for different bonus information to include in the model input."""

    ## The raw upper score 0-105
    TOTAL_UPPER_SCORE = 0

    ## The raw upper score normalized to [0, 1]
    NORMALIZED_TOTAL_UPPER_SCORE = 1

    ## Points away from bonus, i.e. max(0, 63-upper)
    POINTS_AWAY_FROM_BONUS = 2

    ## Percent progress towards bonus (0.0 to 1.0)
    PERCENT_PROGRESS_TOWARDS_BONUS = 3

    ## Golf Scoring (sum each scored category as +/- from getting 3 dice of that category)
    GOLF_SCORING = 4

    ## Golf Scoring normalized to [-1, 1]
    NORMALIZED_GOLF_SCORING = 5

    ## Per-category score needed to achieve bonus
    AVERAGE_SCORE_NEEDED_PER_OPEN_CATEGORY = 6


UPPER_SCORE_THRESHOLD = 63
MAX_UPPER_SCORE = 5 * (1 + 2 + 3 + 4 + 5 + 6)  # 5 dice, each can contribute 1-6
GOLF_TARGET = (np.arange(6) + 1) * 3  # Target score for golf scoring per category


def get_input_dimensions(bonus_flags: set[BonusFlags]) -> int:
    """Calculate the input dimension for the model based on bonus flags.

    Model inputs:
      - Dice [30]: One-hot encoding of 5 dice (6 sides each) = 5 * 6 = 30
      - Rolls Used [3]: One-hot encoding of rolls used (0, 1, 2) = 3
      - Available Categories [13]: One-hot encoding of available scoring categories = 13
      - Current Phase [1]: Current phase of the game (0: rolling, 1: scoring) = 1
      - Dice Counts [6]: Counts of each die face (1-6) = 6
      - Bonus Information [varies]: Various bonus-related inputs = len(bonus_flags)
      - Has Earned Yahtzee [1]: Whether the player has already scored a Yahtzee = 1
      - Potential Scores [13]: Potential score in each category = 13
      - Joker Indicator [1]: Whether joker rules are active = 1
    """
    return int(
        (NUMBER_OF_DICE * NUMBER_OF_DICE_SIDES)  # Dice one-hot
        + (FINAL_ROLL + 1)  # Rolls used one-hot
        + NUMBER_OF_CATEGORIES  # Available categories
        + 1  # Current phase
        + NUMBER_OF_DICE_SIDES  # Dice counts
        + len(bonus_flags)  # Bonus information
        + 1  # Has earned Yahtzee
        + 1  # Percent of game remaining
        + NUMBER_OF_CATEGORIES  # potential score in category
        + 1  # joker indicator
    )


NORMAL_SCORE_MAX = np.array(
    [
        5,  # Ones
        10,  # Twos
        15,  # Threes
        20,  # Fours
        25,  # Fives
        30,  # Sixes
        30,  # Three of a Kind
        30,  # Four of a Kind
        25,  # Full House
        30,  # Small Straight
        40,  # Large Straight
        50,  # Yahtzee
        30,  # Chance
    ]
)


def phi(
    observation: Observation, bonus_flags: set[BonusFlags], device: torch.device
) -> torch.Tensor:
    """Convert observation dictionary to input tensor for the model."""
    dice = observation["dice"]  # numpy array showing the actual dice, e.g. [1, 3, 5, 6, 2]
    dice_counts = np.bincount(dice, minlength=7)[1:]  # counts of dice faces from 1 to 6
    rolls_used = observation["rolls_used"]  # integer: 0, 1, or 2
    available_categories = observation[
        "score_sheet_available_mask"
    ]  # mask for available scoring categories (13,)
    phase = observation.get("phase", 0)  # Current phase of the game (0: rolling, 1: scoring)

    bonus_information = []

    total_upper_score = observation["score_sheet"][:6].sum()

    golf_score = np.sum(
        (observation["score_sheet"][:6] - GOLF_TARGET) * (1 - available_categories[:6])
    )

    score_values, joker = get_all_scores(
        dice,
        available_categories,
        observation["score_sheet"][ScoreCategory.YAHTZEE] == YAHTZEE_SCORE,
    )

    # Normalize score values to [0, 1], capped at 1
    normalized_score_values = np.minimum(score_values / NORMAL_SCORE_MAX, 1.0)

    normalized_golf_score = (
        golf_score / UPPER_SCORE_THRESHOLD
        if golf_score < 0
        else golf_score / (MAX_UPPER_SCORE - UPPER_SCORE_THRESHOLD)
    )

    if BonusFlags.TOTAL_UPPER_SCORE in bonus_flags:
        bonus_information.append(total_upper_score)

    if BonusFlags.NORMALIZED_TOTAL_UPPER_SCORE in bonus_flags:
        bonus_information.append(total_upper_score / MAX_UPPER_SCORE)

    if BonusFlags.POINTS_AWAY_FROM_BONUS in bonus_flags:
        points_away = max(0, UPPER_SCORE_THRESHOLD - total_upper_score)
        bonus_information.append(points_away)

    if BonusFlags.PERCENT_PROGRESS_TOWARDS_BONUS in bonus_flags:
        percent_progress = min(1.0, total_upper_score / UPPER_SCORE_THRESHOLD)
        bonus_information.append(percent_progress)

    if BonusFlags.GOLF_SCORING in bonus_flags:
        bonus_information.append(normalized_golf_score)

    if BonusFlags.AVERAGE_SCORE_NEEDED_PER_OPEN_CATEGORY in bonus_flags:
        num_open_categories = np.sum(available_categories[:6])  # Only upper categories
        if num_open_categories > 0:
            points_needed = max(0, UPPER_SCORE_THRESHOLD - total_upper_score)
            average_needed = points_needed / num_open_categories
        else:
            average_needed = 0.0
        bonus_information.append(average_needed)

    dice_onehot = np.eye(6)[dice - 1].flatten()
    rolls_onehot = np.eye(3)[rolls_used]

    has_earned_yahtzee = observation["score_sheet"][ScoreCategory.YAHTZEE] == YAHTZEE_SCORE

    # print(available_categories)

    percent_of_game_remaining = 1.0 - (np.sum(available_categories) / NUMBER_OF_CATEGORIES)

    input_vector = np.concatenate(
        [
            dice_onehot,
            dice_counts,
            rolls_onehot,
            normalized_score_values,  # Normalized score values in [0,1]
            [joker],
            np.array(bonus_information),
            [percent_of_game_remaining],
            [phase],
            [has_earned_yahtzee],
            available_categories,
        ]
    )
    return torch.FloatTensor(input_vector).to(device)


def sample_action(
    rolling_probs: torch.Tensor,
    scoring_probs: torch.Tensor,
    value_est: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Sample an action given logits (rolling probs, scoring probs, and value estimate)."""
    rolling_dist = torch.distributions.Bernoulli(rolling_probs)
    rolling_tensor = rolling_dist.sample()
    rolling_log_prob = rolling_dist.log_prob(rolling_tensor).sum()

    scoring_dist = torch.distributions.Categorical(scoring_probs)
    scoring_tensor = scoring_dist.sample()
    scoring_log_prob = scoring_dist.log_prob(scoring_tensor).sum()

    return (rolling_tensor, scoring_tensor), (rolling_log_prob, scoring_log_prob), value_est


def select_action(
    rolling_probs: torch.Tensor, scoring_probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select deterministic action using argmax/threshold (for validation/testing).

    Uses >0.5 threshold for rolling decisions and argmax for scoring decisions.
    """
    # Rolling: threshold at 0.5 (keep dice with prob > 0.5)
    rolling_tensor = (rolling_probs > 0.5).float()  # noqa: PLR2004

    # Scoring: argmax to select highest probability category
    scoring_tensor = scoring_probs.argmax(dim=-1)

    return (rolling_tensor, scoring_tensor)


class YahtzeeAgent(nn.Module):
    """Neural network model for maximizing score in a single turn of Yahtzee."""

    def __init__(
        self,
        hidden_size: int,
        num_hidden: int,
        dropout_rate: float,
        activation_function: ActivationFunctionName,
    ):
        super().__init__()

        activation = ActivationFunction[activation_function].value

        self.dropout_rate = dropout_rate

        self.bonus_flags: set[BonusFlags] = {BonusFlags.PERCENT_PROGRESS_TOWARDS_BONUS}

        input_size = get_input_dimensions(self.bonus_flags)

        ## Model outputs:
        #   - Action Probabilities [5]: Probability of re-rolling each of the 5 dice
        #   - Scoring probabilities [13]: Probability of selecting each scoring category
        dice_output_size = 5
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

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call method to enable direct calls to the model."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        spine = self.network(x)

        rolling_output = self.rolling_head(spine)
        # Select last 13 inputs as mask for scoring
        scoring_output = self.scoring_head(spine, x[:, -13:])
        value_output = self.value_head(spine)

        return rolling_output.squeeze(0), scoring_output.squeeze(0), value_output.squeeze(0)
