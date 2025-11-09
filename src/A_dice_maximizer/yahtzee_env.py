from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


class Observation(TypedDict):
    """Observation from the Yahtzee environment."""

    dice: np.ndarray
    rolls_used: int


Action = np.ndarray  # MultiBinary(5) - array of 5 binary values for each die

register(
    id="Yahtzee-v0",
    entry_point="A_dice_maximizer.yahtzee_env:YahtzeeEnv",
    max_episode_steps=13,  # 13 rounds in Yahtzee
)


@dataclass
class DiceState:
    """Class to represent the state of the dice in Yahtzee."""

    NUM_DICE: int = 5

    def __post_init__(self) -> None:
        """Initialize the dice state."""
        self.dice: np.ndarray = np.zeros(self.NUM_DICE, dtype=np.int32)
        self.rolls_used = 0  # Track rolls used instead of remaining (0, 1, 2)
        self.__state: Observation = {"dice": self.dice, "rolls_used": self.rolls_used}

    def observation(self) -> Observation:
        """Convert to observation format."""
        # Update the state dict with current values
        self.__state["rolls_used"] = self.rolls_used
        return self.__state

    def reset(self) -> None:
        """Reset the dice state with separate initialization."""
        self.rolls_used = 0
        # Directly initialize dice with random values (no artificial roll_dice call)
        self.dice[:] = np.random.randint(1, 7, size=self.NUM_DICE)
        self.dice.sort()  # Sort in-place

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
                "dice": spaces.Box(low=1, high=6, shape=(5,), dtype=np.int32),
                "rolls_used": spaces.Discrete(3),  # 0, 1, or 2
            }
        )
        action_space = spaces.MultiBinary(
            5
        )  # 5 binary values, one for each die, 1 means re-roll, 0 means hold

        self.observation_space = cast("spaces.Space[Observation]", observation_space)
        self.action_space = cast("spaces.Space[Action]", action_space)

        self.state: DiceState = DiceState()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Execute one time step within the environment."""
        # action is the hold mask for the dice
        self.state.roll_dice(action)
        observation = self.state.observation()
        terminated = self.state.rolls_used == 2  # noqa: PLR2004

        # Give reward as sum of dice when episode terminates
        reward = np.sum(self.state.dice) if terminated else 0.0

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
