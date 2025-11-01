import torch
import numpy as np
from typing import Dict, Any, List


class Episode:
    def __init__(self):
        self.states: List[Dict[str, Any]] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        
    def add_step(self, state: Dict[str, Any], action: np.ndarray, log_prob: torch.Tensor, reward: float = 0.0):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(float(reward))
        
    def total_reward(self) -> float:
        return sum(self.rewards)
        
    def __len__(self) -> int:
        return len(self.states)