from typing import Generic, TypeVar

import torch

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Episode(Generic[ObsType, ActType]):
    """
    Class to store episode data for reinforcement learning.

    This is a (State, Action, LogProb) trajectory along with the final reward.
    """

    def __init__(self) -> None:
        self.states: list[ObsType] = []
        self.actions: list[ActType] = []
        self.log_probs: list[torch.Tensor] = []
        self.reward: float = 0.0

    def add_step(self, state: ObsType, action: ActType, log_prob: torch.Tensor) -> None:
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
