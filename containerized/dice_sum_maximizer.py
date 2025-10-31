import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class DiceSumMaximizer(nn.Module):
    def __init__(self, hidden_size: int = 64, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(DiceSumMaximizer, self).__init__()
        
        self.device = device
        
        # Input size: 5 dice * 6 values + 3 rolls_remaining = 33
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
        
        with torch.no_grad():
            output = self.network(input_tensor)
            
        return output.squeeze(0)
    
    def _observation_to_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        dice = observation['dice']
        rolls_remaining = observation['rolls_remaining']
        
        # One-hot encode dice (values 1-6) -> indices 0-5
        dice_onehot = np.eye(6)[dice - 1].flatten()
        
        # One-hot encode rolls_remaining (values 0-2)
        rolls_onehot = np.eye(3)[rolls_remaining]
        
        input_vector = np.concatenate([dice_onehot, rolls_onehot])
        return torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
