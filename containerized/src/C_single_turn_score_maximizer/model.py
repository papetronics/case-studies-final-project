from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

@dataclass
class BonusFlags:
    ## The raw upper score 0-105
    TOTAL_UPPER_SCORE = 'total_upper_score'
    
    ## The raw upper score normalized to [0, 1]
    NORMALIZED_TOTAL_UPPER_SCORE = 'normalized_total_upper_score'
    
    ## Points away from bonus, i.e. max(0, 63-upper)
    POINTS_AWAY_FROM_BONUS = 'points_away_from_bonus'

    ## Percent progress towards bonus (0.0 to 1.0)
    PERCENT_PROGRESS_TOWARDS_BONUS = 'percent_progress_towards_bonus'

    ## Golf Scoring (sum each scored category as +/- from getting 3 dice of that category)
    GOLF_SCORING = 'golf_scoring'

    ## Golf Scoring normalized to [-1, 1]
    NORMALIZED_GOLF_SCORING = 'normalized_golf_scoring'

    ## Per-category score needed to achieve bonus
    AVERAGE_SCORE_NEEDED_PER_OPEN_CATEGORY = 'average_score_needed_per_open_category'

UPPER_SCORE_THRESHOLD = 63
MAX_UPPER_SCORE = 5 * (1 + 2 + 3 + 4 + 5 + 6)  # 5 dice, each can contribute 1-6
GOLF_TARGET = (np.arange(6) + 1) * 3  # Target score for golf scoring per category

def observation_to_tensor(observation: Dict[str, Any], bonusFlags: list[str]) -> torch.Tensor:
    dice = observation['dice'] # numpy array showing the actual dice, e.g. [1, 3, 5, 6, 2]
    dice_counts = np.bincount(dice, minlength=7)[1:]  # counts of dice faces from 1 to 6
    rolls_used = observation['rolls_used'] # integer: 0, 1, or 2
    available_categories = observation['score_sheet_available_mask'] # mask for available scoring categories (13,)
    phase = observation.get('phase', 0)  # Current phase of the game (0: rolling, 1: scoring)

    bonus_information = []

    total_upper_score = observation['score_sheet'][:6].sum()

    golf_score = np.sum( (observation['score_sheet'][:6] - GOLF_TARGET) * (1 - available_categories[:6]) )

    normalized_golf_score = golf_score / UPPER_SCORE_THRESHOLD if golf_score < 0 else golf_score / (MAX_UPPER_SCORE - UPPER_SCORE_THRESHOLD)


    if BonusFlags.TOTAL_UPPER_SCORE in bonusFlags:
        bonus_information.append(total_upper_score)
    
    if BonusFlags.NORMALIZED_TOTAL_UPPER_SCORE in bonusFlags:
        bonus_information.append(total_upper_score / MAX_UPPER_SCORE)
    
    if BonusFlags.POINTS_AWAY_FROM_BONUS in bonusFlags:
        points_away = max(0, UPPER_SCORE_THRESHOLD - total_upper_score)
        bonus_information.append(points_away)
    
    if BonusFlags.PERCENT_PROGRESS_TOWARDS_BONUS in bonusFlags:
        percent_progress = min(1.0, total_upper_score / UPPER_SCORE_THRESHOLD)
        bonus_information.append(percent_progress)
    
    if BonusFlags.GOLF_SCORING in bonusFlags:
        bonus_information.append(normalized_golf_score)

    if BonusFlags.AVERAGE_SCORE_NEEDED_PER_OPEN_CATEGORY in bonusFlags:
        num_open_categories = np.sum(available_categories[:6])  # Only upper categories
        if num_open_categories > 0:
            points_needed = max(0, UPPER_SCORE_THRESHOLD - total_upper_score)
            average_needed = points_needed / num_open_categories
        else:
            average_needed = 0.0
        bonus_information.append(average_needed)


    dice_norm = (dice - 1) / 5.0 # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    rolls_norm = rolls_used / 2.0  # 0.0, 0.5, 1.0
    bins_norm = dice_counts / 5.0  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    #print(available_categories)

    input_vector = np.concatenate([dice_norm, bins_norm, [rolls_norm], np.array(bonus_information), [phase], available_categories])
    return torch.FloatTensor(input_vector)

class TurnScoreMaximizer(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.0, activation = nn.PReLU):
            super(TurnScoreMaximizer.Block, self).__init__()
            layers = [
                nn.Linear(in_features, out_features),
                activation(),
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

    def __init__(self,
                 hidden_size: int = 64,
                 num_hidden: int = 1,
                 dropout_rate: float = 0.1, 
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 activation_function: str = 'GELU'):
        super(TurnScoreMaximizer, self).__init__()

        activation = {
            'ReLU': nn.ReLU,
            'GELU': nn.GELU,
            'CELU': nn.CELU,
            'PReLU': nn.PReLU,
            'ELU': nn.ELU,
            'Tanh': nn.Tanh,
            'LeakyReLU': nn.LeakyReLU,
            'Softplus': nn.Softplus,
            'Softsign': nn.Softsign,
            'Mish': nn.Mish,
            'Swish': nn.SiLU,
            'SeLU': nn.SELU
        }[activation_function]

        if activation is None:
            raise ValueError(f"Unsupported activation function: {activation_function}")
        
        self.dropout_rate = dropout_rate

        self.bonus_flags = [
            BonusFlags.PERCENT_PROGRESS_TOWARDS_BONUS
        ]
        
        ## 46 model inputs:
        #   - Dice [5]: Dice, normalized to [0, 1]
        #   - Rolls Used [1]: Normalized rolls used (0, 0.5, 1) = 1
        #   - Available Categories [13]: Available scoring categories, 0/1 = 13
        #   - Current Phase [1]: Current phase of the game (0: rolling, 1: scoring) = 1
        #   - Dice Counts [6]: Counts of each die face (1-6) = 6, normalized to [0, 1]
        input_size = 5 + 1 + 13 + 1 + 6 + len(self.bonus_flags)

        ## 18 Model outputs:
        #   - Action Probabilities [5]: Probability of re-rolling each of the 5 dice
        #   - Scoring probabilities [13]: Probability of selecting each scoring category
        dice_output_size = 5
        scoring_output_size = 13

        layers = [TurnScoreMaximizer.Block(input_size, hidden_size, dropout_rate, activation)]
        for _ in range(num_hidden - 1):
            layers.append(TurnScoreMaximizer.Block(hidden_size, hidden_size, dropout_rate, activation))

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

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def forward_observation(self, observation: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(observation_to_tensor(observation, self.bonus_flags).unsqueeze(0).to(self.device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spine = self.network(x)

        rolling_output = self.rolling_head(spine)
        scoring_output = self.scoring_head(spine)

        # select last 13 inputs as mask
        scoring_output = self.masked_softmax(scoring_output, x[:, -13:])

        return rolling_output.squeeze(0), scoring_output.squeeze(0)

    def sample_observation(self, observation: Dict[str, Any]) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        return self.sample(observation_to_tensor(observation, self.bonus_flags).unsqueeze(0).to(self.device))

    def sample(self, x: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        rolling_probs, scoring_probs = self.forward(x)
        #print(rolling_probs)
        #print(scoring_probs)
        rolling_dist = torch.distributions.Bernoulli(rolling_probs)
        rolling_tensor = rolling_dist.sample()
        rolling_log_prob = rolling_dist.log_prob(rolling_tensor).sum()

        scoring_dist = torch.distributions.Categorical(scoring_probs)
        scoring_tensor = scoring_dist.sample()
        scoring_log_prob = scoring_dist.log_prob(scoring_tensor).sum()

        return (rolling_tensor, scoring_tensor), (rolling_log_prob, scoring_log_prob)
    

