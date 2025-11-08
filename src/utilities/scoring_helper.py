from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreCategory:
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

    LABELS = [
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


def get_all_scores(dice: np.ndarray, open_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a set of dice, compute the raw score for all categories and soft targets for max scoring."""
    # Count occurrences of each die face (1-6)
    counts = np.bincount(dice, minlength=7)[1:]  # Ignore index 0
    upper_scores = counts * np.arange(1, 7)

    # Calculate scores for lower section
    three_of_a_kind = np.sum(dice) if np.any(counts >= 3) else 0
    four_of_a_kind = np.sum(dice) if np.any(counts >= 4) else 0
    full_house = 25 if (np.any(counts == 3) and np.any(counts == 2)) else 0
    small_straight = 30 if (has_small_straight(counts)) else 0
    large_straight = 40 if (has_large_straight(counts)) else 0
    yahtzee = 50 if np.any(counts == 5) else 0
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

    all_scores = np.concatenate([upper_scores, lower_scores])

    # Create soft targets for max scoring - handle multiple max values
    masked_scores = all_scores * open_scores
    max_score = np.max(masked_scores)

    # Create soft targets: 1.0 for all categories that achieve max score, 0.0 otherwise
    max_scoring_target = np.zeros(13, dtype=np.float32)
    if max_score > 0:  # Avoid division by zero when all scores are 0
        max_scoring_target = (masked_scores == max_score).astype(np.float32)
        # Normalize so probabilities sum to 1
        max_scoring_target = max_scoring_target / np.sum(max_scoring_target)
    else:
        # When all scores are 0, all open categories are equally valid
        max_scoring_target = open_scores.astype(np.float32)
        max_scoring_target = max_scoring_target / np.sum(max_scoring_target)

    return masked_scores, max_scoring_target


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
