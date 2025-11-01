"""
Yahtzee Gymnasium Environment

A blank template for the Yahtzee environment implementation.
"""

from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Any

@dataclass
class ScoreCategories:
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

@dataclass
class GameState:
    dice: np.ndarray  # Array of 5 dice values
    rolls_remaining: int  # Number of rolls remaining in the current turn
    current_turn: int  # Current turn number (0-12, with 13 being terminal)
    total_score: float  # Total score accumulated so far
    open_scores: np.ndarray  # Boolean array indicating which score categories are still open

    def __init__(self):
        self.dice = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        self.rolls_remaining = 3
        self.current_turn = 0
        self.total_score = 0.0
        self.open_scores = np.ones(13, dtype=bool)  # All score categories are initially open
        self.reset_dice()

    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self.current_turn >= 13
    
    def reset_dice(self) -> None:
        """Reset the dice to initial state."""
        self.dice = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        self.rolls_remaining = 3
        self.roll_dice(np.array([0, 0, 0, 0, 0]))

    def roll_dice(self, hold_mask) -> None:
        """Roll the dice, holding those indicated by the hold_mask."""
        new_dice = self.dice.copy()
        # Use mask to roll only dice that are not held (hold_mask == 0)
        roll_mask = hold_mask == 0
        new_dice[roll_mask] = np.random.randint(1, 7, size=np.sum(roll_mask))
        # Ensure the dice are int32 to match observation space
        new_dice = new_dice.astype(np.int32)

        self.dice = new_dice
        self.rolls_remaining = self.rolls_remaining - 1

    def score_dice(self, category: int) -> float:
        """Score the current dice in the given category and return new state and score."""
        score = 0.0
        dice_counts = np.bincount(self.dice, minlength=7)[1:]  # Counts of dice values 1-6

        match category:
            case ScoreCategories.ONES:
                score = dice_counts[0] * 1
            case ScoreCategories.TWOS:
                score = dice_counts[1] * 2
            case ScoreCategories.THREES:
                score = dice_counts[2] * 3
            case ScoreCategories.FOURS:
                score = dice_counts[3] * 4
            case ScoreCategories.FIVES:
                score = dice_counts[4] * 5
            case ScoreCategories.SIXES:
                score = dice_counts[5] * 6
            case ScoreCategories.THREE_OF_A_KIND:
                if np.any(dice_counts >= 3):
                    score = np.sum(self.dice)
            case ScoreCategories.FOUR_OF_A_KIND:
                if np.any(dice_counts >= 4):
                    score = np.sum(self.dice)
            case ScoreCategories.FULL_HOUSE:
                if 3 in dice_counts and 2 in dice_counts:
                    score = 25
            case ScoreCategories.SMALL_STRAIGHT:
                if (dice_counts[0:4] >= 1).all() or (dice_counts[1:5] >= 1).all() or (dice_counts[2:6] >= 1).all():
                    score = 30
            case ScoreCategories.LARGE_STRAIGHT:
                if (dice_counts[0:5] >= 1).all() or (dice_counts[1:6] >= 1).all():
                    score = 40
            case ScoreCategories.YAHTZEE:
                if np.any(dice_counts == 5):
                    score = 50
            case ScoreCategories.CHANCE:
                score = np.sum(self.dice)

        self.current_turn = self.current_turn + 1
        self.total_score = self.total_score + score
        self.open_scores[category] = 0

        self.reset_dice()

        return score

    def observation(self) -> dict:
        """Convert to observation format."""
        return {
            "dice": self.dice,
            "rolls_remaining": self.rolls_remaining,
            "current_turn": self.current_turn,
            "open_scores": self.open_scores.astype(np.int8)
        }

class YahtzeeEnv(gym.Env):
    """
    Yahtzee environment for Gymnasium.
    
    This is a blank template - implementation to be added later.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None) -> None:
        """Initialize the Yahtzee environment."""
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            "dice": spaces.Box(low=1, high=6, shape=(5,), dtype=np.int32),
            "open_scores": spaces.MultiBinary(13),  # 13 score categories
            "rolls_remaining": spaces.Discrete(3),  # 0, 1, or 2
            "current_turn": spaces.Discrete(14)  # 0 to 13 (where 13 is terminal)
        })
        self.action_space = spaces.Dict({
            "dice_hold": spaces.MultiBinary(5),  # 5 binary values, one for each die
            "scoring": spaces.Discrete(13)  # 13 score categories
        })

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Execute one time step within the environment."""
        truncated = False
        info = {}
        reward = 0.0

        dice_hold = action['dice_hold']
        scoring = action['scoring']

        if self.state.rolls_remaining == 0:
            # Scoring phase
            reward = self.state.score_dice(scoring)
        else:
            # Rolling phase
            self.state.roll_dice(dice_hold)

        terminated = self.state.is_terminal()

        return self.state.observation(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:  # type: ignore
        """Reset the environment to an initial state."""
        super().reset(seed=seed, options=options)
        
        self.state = GameState()

        info = {}
        
        return self.state.observation(), info
    
    def render(self) -> None:
        """Render the environment."""
        print("Current Dice:", self.state.dice)
        print("Rolls Remaining:", self.state.rolls_remaining)
        print("Current Turn:", self.state.current_turn)
    
    def close(self) -> None:
        """Close the environment."""
        # TODO: Clean up any resources
        pass
