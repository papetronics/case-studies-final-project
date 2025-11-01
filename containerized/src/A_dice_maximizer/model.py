import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class DiceSumMaximizer(nn.Module):
    def __init__(self, hidden_size: int = 64, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(DiceSumMaximizer, self).__init__()
        
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(33, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        input_tensor = self._observation_to_tensor(observation)
        output = self.network(input_tensor)
        return output.squeeze(0)
    
    def sample(self, observation: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.forward(observation)
        action_dist = torch.distributions.Bernoulli(action_probs)
        action_tensor = action_dist.sample()
        log_prob = action_dist.log_prob(action_tensor).sum()
        return action_tensor, log_prob
    
    def _observation_to_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        dice = observation['dice']
        rolls_remaining = observation['rolls_remaining']
        
        dice_onehot = np.eye(6)[dice - 1].flatten()
        rolls_onehot = np.eye(3)[rolls_remaining]
        
        input_vector = np.concatenate([dice_onehot, rolls_onehot])
        return torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
