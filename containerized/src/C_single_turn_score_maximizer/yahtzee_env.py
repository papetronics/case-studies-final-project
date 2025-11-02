"""
Yahtzee Gymnasium Environment

A blank template for the Yahtzee environment implementation.
"""

from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any

from src.scoring_helper import get_all_scores
            
from gymnasium.envs.registration import register
    
register(
    id='Yahtzee-v1',
    entry_point='src.C_single_turn_score_maximizer.yahtzee_env:YahtzeeEnv',
    max_episode_steps=3,  # 3 rounds: roll, roll, score
)

@dataclass
class DiceState:
    NUM_DICE: int = 5
    
    def __post_init__(self):
        self.dice: np.ndarray = np.zeros(self.NUM_DICE, dtype=np.int32)
        self.rolls_used = 0  # Track rolls used instead of remaining (0, 1, 2)
        self.phase = 0  # 0: rolling phase, 1: scoring phase
        
        # Initialize the scoresheet to randomly pick 0->12 categories being full,
        # then randomly pick which categories are full
        # 1 = available, 0 = filled
        self.score_sheet_available_mask: np.ndarray = np.ones(13, dtype=np.int32)
        self.randomly_fill_scoresheet()

        self.__state = {
            "dice": self.dice,
            "rolls_used": self.rolls_used,
            "phase": self.phase,
            "score_sheet_available_mask": self.score_sheet_available_mask  # 1 if available, 0 if filled
        }
    
    def randomly_fill_scoresheet(self) -> None:
        """Randomly fill the score sheet."""
        self.score_sheet_available_mask[:] = 1
        num_filled = np.random.randint(0, 13)  # 0 to 12 inclusive
        filled_indices = np.random.choice(13, size=num_filled, replace=False)
        self.score_sheet_available_mask[filled_indices] = 0

    def observation(self) -> dict:
        """Convert to observation format."""
        # Update the state dict with current values
        self.__state['rolls_used'] = self.rolls_used
        self.__state['phase'] = self.phase
        return self.__state

    def reset(self) -> None:
        """Reset the dice state with separate initialization."""
        self.rolls_used = 0
        self.phase = 0  # Reset to rolling phase
        # Directly initialize dice with random values (no artificial roll_dice call)
        self.dice[:] = np.random.randint(1, 7, size=self.NUM_DICE)
        self.dice.sort()  # Sort in-place

        self.randomly_fill_scoresheet()

    def roll_dice(self, roll_mask: np.ndarray) -> None:
        """Roll the dice, holding those indicated by the hold_mask."""
        # Only roll dice that aren't held (where roll_mask is 1)
        # Convert to boolean array for proper indexing
        num_to_roll = np.sum(roll_mask)
        
        if num_to_roll > 0:
            # Generate only the random numbers we need
            new_rolls = np.random.randint(1, 7, size=num_to_roll)
            # Use boolean indexing to update only non-held dice
            self.dice[roll_mask] = new_rolls

        self.rolls_used += 1
        self.dice.sort()  # Sort in-place after rolling

        self.phase = 0 if self.rolls_used < 2 else 1  # Update phase

    def score_dice(self, category: int) -> int:
        """Score the dice for the given category and update the scoresheet."""
        score, _ = get_all_scores(self.dice, self.score_sheet_available_mask)
        if self.score_sheet_available_mask[category] == 1:
            self.score_sheet_available_mask[category] = 0  # Mark as filled
            return score[category] 
        else:
            raise ValueError("Category already filled.")

class YahtzeeEnv(gym.Env):
    """
    Yahtzee environment for Gymnasium.
    
    This is a blank template - implementation to be added later.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    info = {}
    
    def __init__(self, render_mode=None) -> None:
        """Initialize the Yahtzee environment."""
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            "dice": spaces.Box(low=1, high=6, shape=(5,), dtype=np.int32), # the raw face values of the 5 dice
            "rolls_used": spaces.Discrete(3),  # 0, 1, or 2
            "score_sheet_available_mask": spaces.MultiBinary(13),  # 13 categories, 1 if available, 0 if filled,
            "phase": spaces.Discrete(2)  # 0: rolling phase, 1: scoring phase
        })
        self.action_space = spaces.Dict({
            "hold_mask": spaces.MultiBinary(5),  # 5 binary values, one for each die, 1 means re-roll, 0 means hold
            "score_category": spaces.Discrete(13)  # choose which category to play (0-12)
        })

        self.state: DiceState = DiceState()
    
    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Execute one time step within the environment."""

        terminated = False
        observation = {}
        reward = 0.0

        if self.state.rolls_used < 2:
            # action is the hold mask for the dice
            self.state.roll_dice(action["hold_mask"])
            terminated = False
        else:
            # Final scoring action
            score_category = action["score_category"]
            reward = self.state.score_dice(score_category)
            # Here we would compute the score for the chosen category
            # For now, we just terminate the episode
            terminated = True

        observation = self.state.observation()
        return observation, reward, terminated, False, self.info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:  # type: ignore
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
