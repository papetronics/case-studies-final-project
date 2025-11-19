"""Feature definitions for the phi() state representation function."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from environments.full_yahtzee_env import Observation
from utilities.scoring_helper import YAHTZEE_SCORE, ScoreCategory, get_all_scores


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
        return 14

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

        return np.concatenate([score_values.astype(np.float64), [float(joker)]])


FEATURE_REGISTRY: dict[str, type[PhiFeature]] = {
    "potential_scoring_opportunities": PotentialScoringOpportunitiesFeature,
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
