import torch
import numpy as np
from typing import Dict, Any, List


class Episode:
    def __init__(self):
        self.states: List[Dict[str, Any]] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[torch.Tensor] = []
        self.reward: float = 0.0
        
    def add_step(self, state: Dict[str, Any], action: np.ndarray, log_prob: torch.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        
    def set_reward(self, reward: float):
        self.reward = float(reward)
        
    def __len__(self) -> int:
        return len(self.states)