import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class DiceSumMaximizer(nn.Module):
    def __init__(self, hidden_size: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(DiceSumMaximizer, self).__init__()
        
        self.device = device
        
        ## 33 model inputs:
        #   - Dice [30]: One-hot encoding of 5 dice (6 sides each) = 5 * 6 = 30
        #   - Rolls Used [3]: One-hot encoding of rolls used (0, 1, 2) = 3
        self.network = nn.Sequential(
            nn.Linear(33, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
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
        dice = observation['dice'] # numpy array showing the actual dice, e.g. [1, 3, 5, 6, 2]
        rolls_used = observation['rolls_used'] # integer: 0, 1, or 2

        dice_onehot = np.eye(6)[dice - 1].flatten()
        rolls_onehot = np.eye(3)[rolls_used]

        input_vector = np.concatenate([dice_onehot, rolls_onehot])
        return torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
