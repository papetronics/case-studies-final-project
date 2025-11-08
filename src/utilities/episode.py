from typing import Any

import numpy as np
import torch


class Episode:
    """
    Class to store episode data for reinforcement learning.

    This is a (State, Action, LogProb) trajectory along with the final reward.
    """

    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []
        self.actions: list[np.ndarray | dict] = []
        self.log_probs: list[torch.Tensor] = []
        self.reward: float = 0.0

    def add_step(
        self, state: dict[str, Any], action: np.ndarray | dict, log_prob: torch.Tensor
    ) -> None:
        """Add a step to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)

    def set_reward(self, reward: float) -> None:
        """Set the final reward for the episode."""
        self.reward = float(reward)

    def __len__(self) -> int:
        """Return the number of steps in the episode."""
        return len(self.states)
