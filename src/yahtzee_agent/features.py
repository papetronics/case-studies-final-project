"""Feature definitions for the phi() state representation function."""
# ruff: noqa: D102

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from environments.full_yahtzee_env import FINAL_ROLL, Observation
from utilities.scoring_helper import (
    NUMBER_OF_CATEGORIES,
    NUMBER_OF_DICE,
    NUMBER_OF_DICE_SIDES,
    YAHTZEE_SCORE,
    ScoreCategory,
    get_all_scores,
)


class PhiFeature(ABC):
    """Base class for phi() features."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Extract and encode the feature from the observation."""
        pass


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


class PotentialScoringOpportunitiesFeature(PhiFeature):
    """Potential scores available for each category plus joker indicator."""

    @property
    def name(self) -> str:
        return "potential_scoring_opportunities"

    @property
    def size(self) -> int:
        return int(NUMBER_OF_CATEGORIES + 1)

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute potential scores and joker status from current dice and scoresheet."""
        dice = observation["dice"]
        available_categories = observation["score_sheet_available_mask"]
        has_scored_yahtzee = observation["score_sheet"][ScoreCategory.YAHTZEE] == YAHTZEE_SCORE

        score_values, joker = get_all_scores(
            dice,
            available_categories,
            has_scored_yahtzee,
        )

        return np.concatenate([(score_values / 50.0), [float(joker)]])


class GameProgressFeature(PhiFeature):
    """Percentage of game remaining based on available scoring categories."""

    @property
    def name(self) -> str:
        return "game_progress"

    @property
    def size(self) -> int:
        return 1

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute percent of game remaining from available categories."""
        available_categories = observation["score_sheet_available_mask"]
        percent_of_game_remaining = 1.0 - (np.sum(available_categories) / NUMBER_OF_CATEGORIES)
        return np.array([percent_of_game_remaining])


# Base observation features (always included)


class DiceOneHotFeature(PhiFeature):
    """One-hot encoding of the 5 dice values."""

    @property
    def name(self) -> str:
        return "dice_onehot"

    @property
    def size(self) -> int:
        return int(NUMBER_OF_DICE * NUMBER_OF_DICE_SIDES)

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute one-hot encoding of dice."""
        dice = observation["dice"]  # numpy array showing actual dice, e.g. [1, 3, 5, 6, 2]
        dice_onehot: NDArray[np.floating] = (
            np.eye(NUMBER_OF_DICE_SIDES)[dice - 1].flatten().astype(np.float64)
        )
        return dice_onehot


class DiceCountsFeature(PhiFeature):
    """Counts of each die face (1-6)."""

    @property
    def name(self) -> str:
        return "dice_counts"

    @property
    def size(self) -> int:
        return int(NUMBER_OF_DICE_SIDES)

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute counts of each die face."""
        dice = observation["dice"]
        dice_counts = np.bincount(dice, minlength=7)[1:]  # counts of dice faces from 1 to 6
        return dice_counts.astype(np.float64)


class RollsUsedFeature(PhiFeature):
    """One-hot encoding of rolls used (0, 1, or 2)."""

    @property
    def name(self) -> str:
        return "rolls_used"

    @property
    def size(self) -> int:
        return int(FINAL_ROLL + 1)

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute one-hot encoding of rolls used."""
        rolls_used = observation["rolls_used"]  # integer: 0, 1, or 2
        rolls_onehot: NDArray[np.floating] = np.eye(FINAL_ROLL + 1)[rolls_used].astype(np.float64)
        return rolls_onehot


class PhaseFeature(PhiFeature):
    """Current phase of the game (0: rolling, 1: scoring)."""

    @property
    def name(self) -> str:
        return "phase"

    @property
    def size(self) -> int:
        return 1

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute current phase."""
        phase = observation.get("phase", 0)  # Current phase (0: rolling, 1: scoring)
        return np.array([phase], dtype=np.float64)


class HasEarnedYahtzeeFeature(PhiFeature):
    """Whether the player has already scored a Yahtzee (50 points)."""

    @property
    def name(self) -> str:
        return "has_earned_yahtzee"

    @property
    def size(self) -> int:
        return 1

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute whether Yahtzee has been scored."""
        has_earned_yahtzee = observation["score_sheet"][ScoreCategory.YAHTZEE] == YAHTZEE_SCORE
        return np.array([float(has_earned_yahtzee)], dtype=np.float64)


class AvailableCategoriesFeature(PhiFeature):
    """Binary mask of which scoring categories are still available."""

    @property
    def name(self) -> str:
        return "available_categories"

    @property
    def size(self) -> int:
        return int(NUMBER_OF_CATEGORIES)

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute available categories mask."""
        available_categories: NDArray[np.floating] = observation[
            "score_sheet_available_mask"
        ].astype(np.float64)
        return available_categories


# Bonus-related features

UPPER_SCORE_THRESHOLD = 63


class PercentProgressTowardsBonusFeature(PhiFeature):
    """Percent progress towards upper section bonus (0.0 to 1.0)."""

    @property
    def name(self) -> str:
        return "percent_progress_towards_bonus"

    @property
    def size(self) -> int:
        return 1

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Upper points remaining."""
        total_upper_score = observation["score_sheet"][:6].sum()
        percent_progress = 1.0 - min(1.0, total_upper_score / UPPER_SCORE_THRESHOLD)
        return np.array([percent_progress], dtype=np.float64)


class UpperSectionGolfScoresFeature(PhiFeature):
    """Unnormalized "golf" scores for upper section categories (1-6)."""

    @property
    def name(self) -> str:
        return "upper_section_golf_scores"

    @property
    def size(self) -> int:
        return 6

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute normalized golf scores for upper section."""
        score_sheet = observation["score_sheet"]
        target_score = 3 * np.linspace(1, 6, 6)  # Target score is 3 times the die face value
        golf_score: NDArray[np.floating] = ((score_sheet[:6] - target_score) / 30).astype(
            np.float64
        )
        return golf_score


FEATURE_REGISTRY: dict[str, type[PhiFeature]] = {
    # Advanced features
    "potential_scoring_opportunities": PotentialScoringOpportunitiesFeature,
    "game_progress": GameProgressFeature,
    # Base observation features
    "dice_onehot": DiceOneHotFeature,
    "dice_counts": DiceCountsFeature,
    "rolls_used": RollsUsedFeature,
    "phase": PhaseFeature,
    "has_earned_yahtzee": HasEarnedYahtzeeFeature,
    "available_categories": AvailableCategoriesFeature,
    # Bonus-related features
    "percent_progress_towards_bonus": PercentProgressTowardsBonusFeature,
    "upper_section_golf_scores": UpperSectionGolfScoresFeature,
}


class UnknownFeatureError(ValueError):
    """Exception raised when an unknown feature name is requested."""

    def __init__(self, feature_name: str, available_features: list[str]) -> None:
        available = ", ".join(available_features)
        super().__init__(f"Unknown feature '{feature_name}'. Available features: {available}")


def create_features(feature_names: list[str]) -> list[PhiFeature]:
    """Create feature instances from names."""
    features = []
    for name in feature_names:
        if name not in FEATURE_REGISTRY:
            raise UnknownFeatureError(name, list(FEATURE_REGISTRY.keys()))
        feature_class = FEATURE_REGISTRY[name]
        features.append(feature_class())
    return features
