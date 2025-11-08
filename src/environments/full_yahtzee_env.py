from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, TypedDict, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from utilities.scoring_helper import (
    BONUS_POINTS,
    MINIMUM_UPPER_SCORE_FOR_BONUS,
    NUMBER_OF_DICE,
    ScoreCategory,
    get_all_scores,
)

register(
    id="FullYahtzee-v1",
    entry_point="full_yahtzee_env:YahtzeeEnv",
    max_episode_steps=39,  # 3 rounds: roll, roll, score X 13 categories = 39 steps
)


FIRST_ROLL = 0
SECOND_ROLL = 1
FINAL_ROLL = 2

SCORE_AVAILABLE = 1
SCORE_FILLED = 0


class Phase(IntEnum):
    """Phases of the Yahtzee game."""

    ROLLING = 0
    SCORING = 1


class CategoryAlreadyFilledError(Exception):
    """Exception raised when trying to score in an already filled category."""

    def __init__(self, category: int) -> None:
        self.category = category

    def __str__(self) -> str:
        """Generate error message."""
        return f"The selected scoring category [{ScoreCategory.LABELS[self.category]}] has already been filled."


class AllRollsUsedError(Exception):
    """Exception raised when trying to roll dice after all rolls have been used."""

    def __str__(self) -> str:
        """Generate error message."""
        return "All rolls have been used for this turn; cannot roll again."


class ScoringAction(TypedDict):
    """Scoring action type for the Yahtzee environment."""

    score_category: int


class RollingAction(TypedDict):
    """Rolling action type for the Yahtzee environment."""

    hold_mask: np.ndarray


Action = ScoringAction | RollingAction


class Observation(TypedDict):
    """Observation type for the Yahtzee environment."""

    dice: np.ndarray
    rolls_used: Literal[0, 1, 2]
    score_sheet: np.ndarray
    score_sheet_available_mask: np.ndarray
    phase: Phase


@dataclass
class DiceState:
    """Dataclass to manage the state of the dice and scoring in Yahtzee."""

    def __post_init__(self) -> None:
        """Initialize the dice state."""
        self.dice: np.ndarray = np.zeros(NUMBER_OF_DICE, dtype=np.int32)
        self.rolls_used: Literal[0, 1, 2] = 0  # Track rolls used instead of remaining (0, 1, 2)

        self.score_sheet: np.ndarray = np.zeros(13, dtype=np.int32)
        # Initialize the scoresheet to randomly pick 0->12 categories being full,
        # then randomly pick which categories are full
        # 1 = available, 0 = filled
        self.score_sheet_available_mask: np.ndarray = np.ones(13, dtype=np.int32) * SCORE_AVAILABLE

        self.__state: Observation = {
            "dice": self.dice,
            "rolls_used": self.rolls_used,
            "phase": Phase.ROLLING,
            "score_sheet": self.score_sheet,
            "score_sheet_available_mask": self.score_sheet_available_mask,  # 1 if available, 0 if filled
        }

    def observation(self) -> Observation:
        """Convert to observation format."""
        # Update the state dict with current values
        self.__state["rolls_used"] = self.rolls_used
        self.__state["phase"] = Phase.ROLLING if self.rolls_used < FINAL_ROLL else Phase.SCORING
        return self.__state

    def reset(self) -> None:
        """Reset the dice state with separate initialization."""
        self.rolls_used = 0
        # Directly initialize dice with random values (no artificial roll_dice call)
        self.dice[:] = np.random.randint(1, 7, size=NUMBER_OF_DICE)
        self.dice.sort()  # Sort in-place

        self.score_sheet_available_mask[:] = SCORE_AVAILABLE
        self.score_sheet[:] = 0

    def roll_dice(self, roll_mask: np.ndarray) -> None:
        """Roll the dice, holding those indicated by the hold_mask."""
        if self.rolls_used >= FINAL_ROLL:
            raise AllRollsUsedError()
        else:
            self.rolls_used = cast("Literal[0, 1, 2]", self.rolls_used + 1)

        # Only roll dice that aren't held (where roll_mask is 1)
        # Convert to boolean array for proper indexing
        num_to_roll = np.sum(roll_mask)

        if num_to_roll > 0:
            # Generate only the random numbers we need
            new_rolls = np.random.randint(1, 7, size=num_to_roll)
            # Use boolean indexing to update only non-held dice
            self.dice[roll_mask] = new_rolls

        self.dice.sort()  # Sort in-place after rolling

    def score_dice(self, category: int) -> int:
        """Score the dice for the given category and update the scoresheet."""
        current_upper_score = np.sum(self.score_sheet[0:6])

        score, _ = get_all_scores(self.dice, self.score_sheet_available_mask)
        if self.score_sheet_available_mask[category] != 1:
            raise CategoryAlreadyFilledError(category)
        self.score_sheet_available_mask[category] = 0  # Mark as filled
        self.score_sheet[category] = score[category]

        # new upper score after scoring
        new_upper_score = np.sum(self.score_sheet[0:6])

        # Check for upper section bonus
        bonus: int = 0
        if (
            current_upper_score < MINIMUM_UPPER_SCORE_FOR_BONUS
            and new_upper_score >= MINIMUM_UPPER_SCORE_FOR_BONUS
        ):
            bonus = BONUS_POINTS  # Bonus category is index 12

        # Reset dice for next turn
        self.rolls_used = 0
        self.phase = 0
        self.dice[:] = np.random.randint(1, 7, size=NUMBER_OF_DICE)
        self.dice.sort()

        score_in_chosen_category: int = score[category]
        return score_in_chosen_category + bonus


class YahtzeeEnv(gym.Env[Observation, Action]):
    """
    Yahtzee environment for Gymnasium.

    This is a blank template - implementation to be added later.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}  # noqa: RUF012
    info: dict[str, Any] = {}  # noqa: RUF012
    observation_space: spaces.Space[Observation]
    action_space: spaces.Space[Action]

    def __init__(self, render_mode: Literal["human"] | None = None) -> None:  # noqa: ARG002
        """Initialize the Yahtzee environment."""
        # Define observation and action spaces
        observation_space = spaces.Dict(
            {
                "dice": spaces.Box(
                    low=1, high=6, shape=(5,), dtype=np.int32
                ),  # the raw face values of the 5 dice
                "rolls_used": spaces.Discrete(3),  # 0, 1, or 2
                "score_sheet": spaces.Box(
                    low=0, high=100, shape=(13,), dtype=np.int32
                ),  # 13 categories, scores from 0 to 100
                "score_sheet_available_mask": spaces.MultiBinary(
                    13
                ),  # 13 categories, 1 if available, 0 if filled,
                "phase": spaces.Discrete(2),  # 0: rolling phase, 1: scoring phase
            }
        )
        action_space = spaces.Dict(
            {
                "hold_mask": spaces.MultiBinary(
                    5
                ),  # 5 binary values, one for each die, 1 means re-roll, 0 means hold
                "score_category": spaces.Discrete(13),  # choose which category to play (0-12)
            }
        )

        self.observation_space = cast("spaces.Space[Observation]", observation_space)
        self.action_space = cast("spaces.Space[Action]", action_space)

        self.state: DiceState = DiceState()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Execute one time step within the environment."""
        terminated = False
        reward = 0.0

        if self.state.phase == Phase.ROLLING:
            rolling_action: RollingAction = cast("RollingAction", action)
            if "hold_mask" not in rolling_action:
                raise ValueError()

            # action is the hold mask for the dice
            self.state.roll_dice(rolling_action["hold_mask"])
        else:
            scoring_action: ScoringAction = cast("ScoringAction", action)
            if "score_category" not in scoring_action:
                raise ValueError()
            # Final scoring action
            score_category = scoring_action["score_category"]
            reward = self.state.score_dice(score_category)
            # Here we would compute the score for the chosen category
            # For now, we just terminate the episode

        terminated = self.state.score_sheet_available_mask.sum() == 0

        observation: Observation = self.state.observation()
        return observation, reward, terminated, False, self.info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed, options=options)

        self.state.reset()

        return self.state.observation(), self.info

    def render(self) -> None:
        """Render the environment."""
        # TODO: Implement rendering
        pass

    def close(self) -> None:
        """Close the environment."""
        # TODO: Clean up any resources
        pass

    def observe(self) -> Observation:
        """Return the current observation."""
        return self.state.observation()
