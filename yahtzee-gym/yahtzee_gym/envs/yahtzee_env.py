"""
Yahtzee Gymnasium Environment

A blank template for the Yahtzee environment implementation.
"""

from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any


@dataclass
class DiceState:
    NUM_DICE: int = 5
    
    def __post_init__(self):
        self.dice: np.ndarray = np.zeros(self.NUM_DICE, dtype=np.int32)
        self.rolls_used = 0  # Track rolls used instead of remaining (0, 1, 2)
        self.__state = {
            "dice": self.dice,
            "rolls_used": self.rolls_used
        }
    
    def observation(self) -> dict:
        """Convert to observation format."""
        # Update the state dict with current values
        self.__state['rolls_used'] = self.rolls_used
        return self.__state

    def reset(self) -> None:
        """Reset the dice state with separate initialization."""
        self.rolls_used = 0
        # Directly initialize dice with random values (no artificial roll_dice call)
        self.dice[:] = np.random.randint(1, 7, size=self.NUM_DICE)

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
            "dice": spaces.Box(low=1, high=6, shape=(5,), dtype=np.int32),
            "rolls_used": spaces.Discrete(3),  # 0, 1, or 2
        })
        self.action_space = spaces.MultiBinary(5)  # 5 binary values, one for each die, 1 means re-roll, 0 means hold

        self.state: DiceState = DiceState()
    
    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Execute one time step within the environment."""

        # action is the hold mask for the dice
        self.state.roll_dice(action)
        observation = self.state.observation()
        terminated = self.state.rolls_used == 2
        
        # Give reward as sum of dice when episode terminates
        reward = np.sum(self.state.dice) if terminated else 0.0

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
