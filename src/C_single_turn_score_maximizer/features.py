"""Feature definitions for the phi() state representation function."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from environments.full_yahtzee_env import Observation
from utilities.scoring_helper import (
    NUMBER_OF_CATEGORIES,
    YAHTZEE_SCORE,
    ScoreCategory,
    get_all_scores,
)


class PhiFeature(ABC):
    """Base class for phi() features."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this feature (used in config)."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of elements this feature contributes to the input vector."""
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
    """Feature that encodes the potential scores available for each category.

    This feature provides information about what scores are achievable with the current
    dice roll across all scoring categories, plus a joker indicator for when yahtzee
    joker rules are active.

    Output dimensions:
        - Potential scores [13]: The score value achievable in each category (0 if unavailable)
        - Joker indicator [1]: 1.0 if joker rules are active, 0.0 otherwise
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this feature."""
        return "potential_scoring_opportunities"

    @property
    def size(self) -> int:
        """13 categories + 1 joker indicator = 14 dimensions."""
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

        return np.concatenate([(score_values / NORMAL_SCORE_MAX), [float(joker)]])


class GameProgressFeature(PhiFeature):
    """Feature that encodes how far through the game we are.

    This feature provides information about game progression by calculating what
    percentage of the game remains based on how many scoring categories are still available.

    Output dimensions:
        - Percent of game remaining [1]: Value from 0.0 (game complete) to 1.0 (game start)
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this feature."""
        return "game_progress"

    @property
    def size(self) -> int:
        """1 dimension for percent of game remaining."""
        return 1

    def compute(self, observation: Observation) -> NDArray[np.floating]:
        """Compute percent of game remaining from available categories."""
        available_categories = observation["score_sheet_available_mask"]
        percent_of_game_remaining = 1.0 - (np.sum(available_categories) / NUMBER_OF_CATEGORIES)
        return np.array([percent_of_game_remaining])


FEATURE_REGISTRY: dict[str, type[PhiFeature]] = {
    "potential_scoring_opportunities": PotentialScoringOpportunitiesFeature,
    "game_progress": GameProgressFeature,
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
