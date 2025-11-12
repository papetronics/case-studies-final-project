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
    # Count occurrences of each die face (1-6)
    counts = np.bincount(dice, minlength=7)[1:]  # Ignore index 0
    is_yahtzee = np.any(counts == DICE_COUNT_YAHTZEE)
    joker_rules_active = is_yahtzee and already_has_yahtzee
    upper_scores = counts * np.arange(1, 7)

    # Calculate scores for lower section
    three_of_a_kind = np.sum(dice) if np.any(counts >= DICE_COUNT_THREE_OF_A_KIND) else 0
    four_of_a_kind = np.sum(dice) if np.any(counts >= DICE_COUNT_FOUR_OF_A_KIND) else 0
    full_house = (
        FULL_HOUSE_SCORE
        if (
            (np.any(counts == DICE_COUNT_THREE_OF_A_KIND) and np.any(counts == DICE_COUNT_PAIR))
            or joker_rules_active
        )
        else 0
    )
    small_straight = (
        SMALL_STRAIGHT_SCORE if (has_small_straight(counts) or joker_rules_active) else 0
    )
    large_straight = (
        LARGE_STRAIGHT_SCORE if (has_large_straight(counts) or joker_rules_active) else 0
    )
    yahtzee = YAHTZEE_SCORE if np.any(counts == DICE_COUNT_YAHTZEE) else 0
    chance = np.sum(dice)

    lower_scores = np.array(
        [
            three_of_a_kind,
            four_of_a_kind,
            full_house,
            small_straight,
            large_straight,
            yahtzee,
            chance,
        ]
    )

    all_scores: NDArray[np.int_] = np.concatenate([upper_scores, lower_scores])

    # Create soft targets for max scoring - handle multiple max values
    masked_scores = all_scores * open_scores
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


def has_small_straight(counts: np.ndarray) -> bool:
    """Check for small straight (4 consecutive numbers)."""
    straights = [
        counts[0:4],  # 1-4
        counts[1:5],  # 2-5
        counts[2:6],  # 3-6
    ]
    return any(np.all(straight >= 1) for straight in straights)


def has_large_straight(counts: np.ndarray) -> bool:
    """Check for large straight (5 consecutive numbers)."""
    return bool(np.all(counts[0:5] >= 1) or np.all(counts[1:6] >= 1))


def has_bonus(scores: np.ndarray) -> bool:
    """Check if the upper section bonus has been achieved."""
    upper_total = np.sum(scores[:NUMBER_OF_DICE_SIDES])
    return bool(upper_total >= MINIMUM_UPPER_SCORE_FOR_BONUS)
