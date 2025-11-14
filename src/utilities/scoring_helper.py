from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

FULL_HOUSE_SCORE: int = 25
SMALL_STRAIGHT_SCORE: int = 30
LARGE_STRAIGHT_SCORE: int = 40
YAHTZEE_SCORE: int = 50
YAHTZEE_BONUS_SCORE: int = 100

NUMBER_OF_DICE: int = 5
NUMBER_OF_DICE_SIDES: int = 6

MINIMUM_UPPER_SCORE_FOR_BONUS: int = 63
BONUS_POINTS: int = 35

DICE_COUNT_PAIR: int = 2
DICE_COUNT_THREE_OF_A_KIND: int = 3
DICE_COUNT_FOUR_OF_A_KIND: int = 4
DICE_COUNT_YAHTZEE: int = 5

NUMBER_OF_CATEGORIES: int = 13


@dataclass
class ScoreCategory:
    """Yahtzee scoring categories with their numeric identifiers and labels."""

    ONES: int = 0
    TWOS: int = 1
    THREES: int = 2
    FOURS: int = 3
    FIVES: int = 4
    SIXES: int = 5
    THREE_OF_A_KIND: int = 6
    FOUR_OF_A_KIND: int = 7
    FULL_HOUSE: int = 8
    SMALL_STRAIGHT: int = 9
    LARGE_STRAIGHT: int = 10
    YAHTZEE: int = 11
    CHANCE: int = 12

    LABELS: ClassVar[list[str]] = [
        "Ones",
        "Twos",
        "Threes",
        "Fours",
        "Fives",
        "Sixes",
        "Three of a Kind",
        "Four of a Kind",
        "Full House",
        "Small Straight",
        "Large Straight",
        "Yahtzee",
        "Chance",
    ]


def get_all_scores(
    dice: NDArray[np.int_], open_scores: NDArray[np.int_], already_has_yahtzee: bool = False
) -> tuple[NDArray[np.int_], bool]:
    """Given a set of dice, compute the raw score for all categories and soft targets for max scoring.

    Args:
        dice: Array of shape (5,) containing dice values
        open_scores: Array of shape (13,) containing binary flags for open categories
        already_has_yahtzee: Whether the player has already scored a Yahtzee in this game

    Returns
    -------
        Tuple of (masked_scores, joker_rules_active) where the first has shape (13,) and the second is a boolean
    """
    # Count occurrences of each die face (1-6)
    counts = np.bincount(dice, minlength=7)[1:]  # Ignore index 0
    dice_sum = np.sum(dice)

    has_three = np.any(counts >= DICE_COUNT_THREE_OF_A_KIND)
    has_four = np.any(counts >= DICE_COUNT_FOUR_OF_A_KIND)
    has_pair = np.any(counts == DICE_COUNT_PAIR)
    has_yahtzee = np.any(counts == DICE_COUNT_YAHTZEE)
    has_small = any(np.all(counts[i : i + 4] >= 1) for i in range(3))
    has_large = np.all(counts[0:5] >= 1) or np.all(counts[1:6] >= 1)

    joker_rules_active: bool = bool(has_yahtzee and already_has_yahtzee)

    # Score possibilities: what you would get if the category is satisfied
    score_possibilities: NDArray[np.int_] = np.array(
        [
            counts[0] * 1,  # Ones
            counts[1] * 2,  # Twos
            counts[2] * 3,  # Threes
            counts[3] * 4,  # Fours
            counts[4] * 5,  # Fives
            counts[5] * 6,  # Sixes
            dice_sum,  # Three of a Kind
            dice_sum,  # Four of a Kind
            FULL_HOUSE_SCORE,  # Full House
            SMALL_STRAIGHT_SCORE,  # Small Straight
            LARGE_STRAIGHT_SCORE,  # Large Straight
            YAHTZEE_SCORE,  # Yahtzee
            dice_sum,  # Chance
        ]
    )

    # Indicator: 1 if category is satisfied, 0 otherwise
    indicators: NDArray[np.int_] = np.array(
        [
            1,  # Ones (always valid)
            1,  # Twos (always valid)
            1,  # Threes (always valid)
            1,  # Fours (always valid)
            1,  # Fives (always valid)
            1,  # Sixes (always valid)
            int(has_three),  # Three of a Kind
            int(has_four),  # Four of a Kind
            int((has_three and has_pair) or joker_rules_active),  # Full House
            int(has_small or joker_rules_active),  # Small Straight (1-4, 2-5, 3-6)
            int(has_large or joker_rules_active),  # Large Straight (1-5 or 2-6)
            int(has_yahtzee),  # Yahtzee
            1,  # Chance (always valid)
        ]
    )

    return score_possibilities * indicators * open_scores, joker_rules_active


def has_bonus(scores: np.ndarray) -> bool:
    """Check if the upper section bonus has been achieved."""
    return bool(np.sum(scores[:NUMBER_OF_DICE_SIDES]) >= MINIMUM_UPPER_SCORE_FOR_BONUS)


def get_all_scores_with_target(
    dice: NDArray[np.int_], open_scores: NDArray[np.int_], already_has_yahtzee: bool = False
) -> tuple[NDArray[np.int_], NDArray[np.float32], bool]:
    """Given a set of dice, compute the raw score for all categories and soft targets for max scoring.

    Args:
        dice: Array of shape (5,) containing dice values
        open_scores: Array of shape (13,) containing binary flags for open categories
        already_has_yahtzee: Whether the player has already scored a Yahtzee in this game

    Returns
    -------
        Tuple of (masked_scores, max_scoring_target, joker_rules_active) where the first two have shape (13,) and the third is a boolean
    """
    masked_scores, joker_rules_active = get_all_scores(dice, open_scores, already_has_yahtzee)

    max_score = np.max(masked_scores)
    # Create soft targets: 1.0 for all categories that achieve max score, 0.0 otherwise
    max_scoring_target = np.zeros(NUMBER_OF_CATEGORIES, dtype=np.float32)
    if max_score > 0:  # Avoid division by zero when all scores are 0
        max_scoring_target = (masked_scores == max_score).astype(np.float32)
        # Normalize so probabilities sum to 1
        max_scoring_target = max_scoring_target / np.sum(max_scoring_target)
    else:
        # When all scores are 0, all open categories are equally valid
        max_scoring_target = open_scores.astype(np.float32)
        max_scoring_target /= np.sum(max_scoring_target, dtype=np.float32)

    return masked_scores, max_scoring_target, bool(joker_rules_active)
