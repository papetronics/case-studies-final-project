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
class DiceState:
    dice: np.ndarray  # Array of 5 dice values
    rolls_remaining: int  # Number of rolls remaining in the current turn
    
    def observation(self) -> dict:
        """Convert to observation format."""
        return {
            "dice": self.dice,
            "rolls_remaining": self.rolls_remaining
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
            "rolls_remaining": spaces.Discrete(3),  # 0, 1, or 2
        })
        self.action_space = spaces.MultiBinary(5)  # 5 binary values, one for each die
    
    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Execute one time step within the environment."""
        truncated = False
        info = {}

        # action is the hold mask for the dice
        self.state = self.__roll_dice(self.state, action)

        terminated = self.state.rolls_remaining == 0
        
        # Give reward as sum of dice when episode terminates
        if terminated:
            reward = float(np.sum(self.state.dice))
        else:
            reward = 0.0

        return self.state.observation(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:  # type: ignore
        """Reset the environment to an initial state."""
        super().reset(seed=seed, options=options)
        
        # TODO: Reset game state
        self.state = self.__roll_dice(
            DiceState(dice=np.array([1, 1, 1, 1, 1], dtype=np.int32), rolls_remaining=3),
            hold_mask=np.array([0, 0, 0, 0, 0])
        )

        info = {}
        
        return self.state.observation(), info
    
    def render(self) -> None:
        """Render the environment."""
        # TODO: Implement rendering
        pass
    
    def close(self) -> None:
        """Close the environment."""
        # TODO: Clean up any resources
        pass

    def __roll_dice(self, initial_state, hold_mask) -> DiceState:
        """Roll the dice, holding those indicated by the hold_mask."""
        new_dice = initial_state.dice.copy()
        # Use mask to roll only dice that are not held (hold_mask == 0)
        roll_mask = hold_mask == 0
        new_dice[roll_mask] = np.random.randint(1, 7, size=np.sum(roll_mask))
        # Ensure the dice are int32 to match observation space
        new_dice = new_dice.astype(np.int32)
        return DiceState(dice=new_dice, rolls_remaining=initial_state.rolls_remaining - 1)
