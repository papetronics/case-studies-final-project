import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

def observation_to_tensor(observation: Dict[str, Any]) -> torch.Tensor:
    dice = observation['dice'] # numpy array showing the actual dice, e.g. [1, 3, 5, 6, 2]
    rolls_used = observation['rolls_used'] # integer: 0, 1, or 2
    available_categories = observation['score_sheet_available_mask'] # mask for available scoring categories (13,)
    phase = observation.get('phase', 0)  # Current phase of the game (0: rolling, 1: scoring)

    dice_onehot = np.eye(6)[dice - 1].flatten()
    rolls_onehot = np.eye(3)[rolls_used]

    #print(available_categories)

    input_vector = np.concatenate([dice_onehot, rolls_onehot, [phase], available_categories])
    return torch.FloatTensor(input_vector)

class TurnScoreMaximizer(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.0):
            super(TurnScoreMaximizer.Block, self).__init__()
            layers = [
                nn.Linear(in_features, out_features),
                nn.GELU(),
                nn.LayerNorm(out_features)
            ]
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
        
    class MaskedSoftmax(nn.Module):
        def __init__(self, mask_value: float = -float('inf')):
            super(TurnScoreMaximizer.MaskedSoftmax, self).__init__()
            self.mask_value = mask_value
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # Apply the mask
            x = x.masked_fill(mask == 0, self.mask_value)
            return self.softmax(x)

    def __init__(self, hidden_size: int = 64, num_hidden: int = 1, dropout_rate: float = 0.1, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(TurnScoreMaximizer, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        ## 46 model inputs:
        #   - Dice [30]: One-hot encoding of 5 dice (6 sides each) = 5 * 6 = 30
        #   - Rolls Used [3]: One-hot encoding of rolls used (0, 1, 2) = 3  (always 2 in this scenario)
        #   - Available Categories [13]: One-hot encoding of available scoring categories = 13
        #   - Current Phase [1]: Current phase of the game (0: rolling, 1: scoring) = 1
        input_size = 30 + 3 + 13 + 1

        ## 18 Model outputs:
        #   - Action Probabilities [5]: Probability of re-rolling each of the 5 dice
        #   - Scoring probabilities [13]: Probability of selecting each scoring category
        dice_output_size = 5
        scoring_output_size = 13

        layers = [TurnScoreMaximizer.Block(input_size, hidden_size, dropout_rate)]
        for _ in range(num_hidden - 1):
            layers.append(TurnScoreMaximizer.Block(hidden_size, hidden_size, dropout_rate))
            
        self.network = nn.Sequential(
            *layers
        )

        rolling_head_layers = []
        if dropout_rate > 0.0:
            rolling_head_layers.append(nn.Dropout(dropout_rate))
        rolling_head_layers.extend([
            nn.Linear(hidden_size, dice_output_size),
            nn.Sigmoid()
        ])
        self.rolling_head = nn.Sequential(*rolling_head_layers).to(self.device)

        scoring_head_layers = []
        if dropout_rate > 0.0:
            scoring_head_layers.append(nn.Dropout(dropout_rate))
        scoring_head_layers.append(nn.Linear(hidden_size, scoring_output_size))
        self.scoring_head = nn.Sequential(*scoring_head_layers).to(self.device)
        self.masked_softmax = TurnScoreMaximizer.MaskedSoftmax().to(self.device)

        baseline_head_layers = []
        if dropout_rate > 0.0:
            baseline_head_layers.append(nn.Dropout(dropout_rate))
        baseline_head_layers.append(nn.Linear(hidden_size, 1))
        self.baseline_head = nn.Sequential(*baseline_head_layers).to(self.device)

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def forward_observation(self, observation: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(observation_to_tensor(observation).unsqueeze(0).to(self.device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spine = self.network(x)

        rolling_output = self.rolling_head(spine)
        scoring_output = self.scoring_head(spine)
        baseline_output = self.baseline_head(spine)

        # select last 13 inputs as mask
        scoring_output = self.masked_softmax(scoring_output, x[:, -13:])

        return rolling_output.squeeze(0), scoring_output.squeeze(0), baseline_output.squeeze(0)

    def sample_observation(self, observation: Dict[str, Any]) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        return self.sample(observation_to_tensor(observation).unsqueeze(0).to(self.device))

    def sample(self, x: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        rolling_probs, scoring_probs, baseline_est = self.forward(x)
        #print(rolling_probs)
        #print(scoring_probs)
        rolling_dist = torch.distributions.Bernoulli(rolling_probs)
        rolling_tensor = rolling_dist.sample()
        rolling_log_prob = rolling_dist.log_prob(rolling_tensor).sum()

        scoring_dist = torch.distributions.Categorical(scoring_probs)
        scoring_tensor = scoring_dist.sample()
        scoring_log_prob = scoring_dist.log_prob(scoring_tensor).sum()

        return (rolling_tensor, scoring_tensor, baseline_est), (rolling_log_prob, scoring_log_prob)
    

