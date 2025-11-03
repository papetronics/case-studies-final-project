import torch
import numpy as np
from typing import Dict, Any, List


class Episode:
    def __init__(self):
        self.states: List[Dict[str, Any]] = []
        self.actions: List[np.ndarray | Dict] = []
        self.log_probs: List[torch.Tensor] = []
        self.Vs: List[torch.Tensor] = []
        self.reward: float = 0.0
        
    def add_step(self, state: Dict[str, Any], action: np.ndarray | Dict, log_prob: torch.Tensor, V: torch.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.Vs.append(V)
        
    def set_reward(self, reward: float):
        self.reward = float(reward)
        
    def __len__(self) -> int:
        return len(self.states)