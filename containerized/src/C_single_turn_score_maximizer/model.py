import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

def observation_to_tensor(observation: Dict[str, Any]) -> torch.Tensor:
    dice = observation['dice'] # numpy array showing the actual dice, e.g. [1, 3, 5, 6, 2]
    rolls_used = observation['rolls_used'] # integer: 0, 1, or 2
    available_categories = observation['score_sheet_available_mask'] # mask for available scoring categories (13,)
    phase = observation.get('phase', 0)  # Current phase of the game (0: rolling, 1: scoring)

    rolls_onehot = np.eye(3)[rolls_used]

    #print(available_categories)

    input_vector = np.concatenate([dice, rolls_onehot, [phase], available_categories])
    return torch.Tensor(input_vector)

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
                 embedding_dim: int | None = None,  # Will be auto-calculated if None
                 num_hidden: int = 1,
                 dropout_rate: float = 0.1, 
                 
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 activation_function: str = 'GELU'):
        super(TurnScoreMaximizer, self).__init__()

        # Ensure hidden_size is divisible by 5 for clean embedding dimension calculation
        if hidden_size % 5 != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by 5 for optimal dice embedding allocation")
        
        # Auto-calculate embedding_dim if not provided
        if embedding_dim is None:
            embedding_dim = hidden_size // 5  # Each die gets 1/5 of the hidden space

        print(f"Using embedding_dim: {embedding_dim} for hidden_size: {hidden_size}")
        
        # Validate that embedding_dim works with the architecture
        if hidden_size != 5 * embedding_dim:
            print(f"Warning: hidden_size ({hidden_size}) != 5 * embedding_dim ({5 * embedding_dim}). "
                  f"This may cause dimension mismatches.")
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

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
        
        ## 22 model inputs:
        #   - Dice [5]: Raw dice values (1-6)
        #   - Rolls Used [3]: One-hot encoding of rolls used (0, 1, 2) = 3
        #   - Available Categories [13]: One-hot encoding of available scoring categories = 13
        #   - Current Phase [1]: Current phase of the game (0: rolling, 1: scoring) = 1
        input_size = 5 + 3 + 13 + 1

        ## 18 Model outputs:
        #   - Action Probabilities [5]: Probability of re-rolling each of the 5 dice
        #   - Scoring probabilities [13]: Probability of selecting each scoring category
        dice_output_size = 5
        scoring_output_size = 13

        self.dice_embedding = nn.Embedding(6, embedding_dim=self.embedding_dim)

        # Cross-attention: dice embeddings attend to game state
        self.dice_cross_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4, batch_first=True)

        # Size of non-dice inputs: rolls_used(3) + phase(1) + available_categories(13) = 17
        non_dice_input_size = 3 + 1 + 13
        # Project game state directly to hidden_size (no intermediate projection needed!)
        self.game_state_projector = TurnScoreMaximizer.Block(non_dice_input_size, self.hidden_size, dropout_rate, activation)
        
        # No feature projector needed anymore since we go directly to hidden_size!

        layers = []
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
        return self.forward(observation_to_tensor(observation).unsqueeze(0).to(self.device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract dice values (first 5 elements) and convert to 0-based indices for embedding
        dice_indices = x[:, :5].long() - 1  # Convert to 0-based indices
        dice_embeddings = self.dice_embedding(dice_indices)  # (batch_size, 5, embedding_dim)

        # Extract non-dice inputs (rolls_used[3] + phase[1] + available_categories[13] = 17)
        non_dice_inputs = x[:, 5:]  # (batch_size, 17)
        
        # Project game state directly to hidden_size
        game_state_features = self.game_state_projector(non_dice_inputs)  # (batch_size, hidden_size)
        
        # Reshape game state to match dice sequence for cross-attention
        game_state_kv = game_state_features.view(x.size(0), 5, self.embedding_dim)  # (batch_size, 5, embedding_dim)

        # Cross-attention: dice embeddings (Q) attend to game state (K, V)
        context_aware_dice, _ = self.dice_cross_attention(dice_embeddings, game_state_kv, game_state_kv)
        
        # Flatten context-aware dice embeddings - this is already hidden_size!
        features = context_aware_dice.view(x.size(0), -1)  # (batch_size, hidden_size)

        # Pass through the main network
        spine = self.network(features)

        rolling_output = self.rolling_head(spine)
        scoring_output = self.scoring_head(spine)

        # Use the available categories mask for scoring (last 13 elements of original input)
        available_categories_mask = x[:, -13:]
        scoring_output = self.masked_softmax(scoring_output, available_categories_mask)

        return rolling_output.squeeze(0), scoring_output.squeeze(0)

    def sample_observation(self, observation: Dict[str, Any]) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        return self.sample(observation_to_tensor(observation).unsqueeze(0).to(self.device))

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
    

