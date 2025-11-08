from typing import Any

import numpy as np
import torch


class Episode:
    def __init__(self):
        self.states: list[dict[str, Any]] = []
        self.actions: list[np.ndarray | dict] = []
        self.log_probs: list[torch.Tensor] = []
        self.reward: float = 0.0

    def add_step(self, state: dict[str, Any], action: np.ndarray | dict, log_prob: torch.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)

    def set_reward(self, reward: float):
        self.reward = float(reward)

    def __len__(self) -> int:
        return len(self.states)
