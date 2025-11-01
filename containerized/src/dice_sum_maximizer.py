import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class ResMLPBlock(nn.Module):
    def __init__(self, size: int):
        super(ResMLPBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
          # Residual connection
        return self.layer(x) + x

class DiceSumMaximizer(nn.Module):
    def __init__(self, hidden_size: int = 512, depth=6, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(DiceSumMaximizer, self).__init__()
        
        self.device = device
        
        # Observation side:
        #  - Dice one-hot: 5 dice * 6 sides = 30
        #  - Rolls remaining one-hot: 3 possible values = 3
        #  - Open scores one-hot: 13 possible categories = 13
        #  - Current turn one-hot: 13 possible values (0-12) = 13
        # Total input size = 30 + 3 + 13 + 13 = 59

        # Action space:
        #  - Dice hold: 5 binary values (sigmoid output)
        #  - Scoring category: one-hot encoded 13 categories (softmax output)
        # Total action size = 5 + 13 = 18

        mlpBlocks = [
            nn.Linear(59, hidden_size),
            nn.GELU()
        ]
        for _ in range(depth):
            mlpBlocks.append(ResMLPBlock(hidden_size))

        self.spine = nn.Sequential(
            *mlpBlocks
        ).to(device)

        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_size, 13),
            nn.Softmax(dim=-1)
        ).to(device)

        self.dice_hold_head = nn.Sequential(
            nn.Linear(hidden_size, 5),
            nn.Sigmoid()
        ).to(device)



    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        input_tensor = self._observation_to_tensor(observation)
        
        spine_output = self.spine(input_tensor)
        dice_hold_output = self.dice_hold_head(spine_output)
        scoring_output = self.scoring_head(spine_output)

        # Return as a (18,) tensor
        return torch.cat([dice_hold_output, scoring_output], dim=-1)

    def sample(self, observation: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs = self.forward(observation)

        # Squeeze out batch dimension for single observation
        action_probs = action_probs.squeeze(0)  # [1, 18] -> [18]

        # hold probs are first 5, scoring probs are last 13
        hold_probs = action_probs[:5]
        scoring_probs = action_probs[5:]

        hold_distribution = torch.distributions.Bernoulli(hold_probs)
        hold_action = hold_distribution.sample()

        # For scoring, we can only score in open categories
        # Convert numpy open_scores to tensor on same device
        open_scores_tensor = torch.FloatTensor(observation['open_scores']).to(self.device)
        open_scoring_probs = scoring_probs * open_scores_tensor
        # renormalize
        open_scoring_probs = open_scoring_probs / open_scoring_probs.sum()
        scoring_distribution = torch.distributions.Categorical(open_scoring_probs)
        scoring_action = scoring_distribution.sample()

        # Only use log prob from the relevant head based on game phase
        if observation['rolls_remaining'] > 0:
            # Rolling phase - use dice hold log prob
            log_prob = hold_distribution.log_prob(hold_action).sum()
        else:
            # Scoring phase - use scoring log prob
            log_prob = scoring_distribution.log_prob(scoring_action)

        return hold_action, scoring_action, log_prob

    def _observation_to_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        dice = observation['dice'] # array of 5 dice values (int)
        rolls_remaining = observation['rolls_remaining'] # int
        open_scores = observation['open_scores'] # multi-binary array
        current_turn = observation['current_turn'] # int

        # One-hot encode dice
        dice_onehot = np.eye(6)[dice - 1].flatten()
        rolls_onehot = np.eye(3)[rolls_remaining]
        open_scores_onehot = open_scores.astype(np.float32)
        current_turn_onehot = np.eye(13)[current_turn]

        input_vector = np.concatenate([dice_onehot, rolls_onehot, open_scores_onehot, current_turn_onehot])
        return torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
