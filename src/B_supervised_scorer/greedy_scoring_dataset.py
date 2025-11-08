import numpy as np
import torch

from B_supervised_scorer.model import observation_to_tensor
from utilities.scoring_helper import get_all_scores


class GreedyScoringDataset(torch.utils.data.Dataset):
    """Dataset for generating greedy scoring samples."""

    def __init__(self, size: int = 1000):
        self.size = size

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a sample by computing the max scoring opportunity for random dice and open categories."""
        # Here we are going to generate a sample, any dice, we figure out what the max scoring opp is, thats the label

        # dice that we are going to use
        dice = np.random.randint(1, 7, size=5)
        dice.sort()

        # Uniformly sample number of categories to keep open (1-13)
        num_open = np.random.randint(1, 14)  # 1 to 13 inclusive

        # Start with all categories closed
        open_scores = np.zeros(13, dtype=int)

        # Randomly select which categories to open
        open_indices = np.random.choice(13, size=num_open, replace=False)
        open_scores[open_indices] = 1

        all_scores, max_scoring_target = get_all_scores(dice, open_scores)
        observation = {"dice": dice, "available_categories": open_scores, "rolls_used": 2}

        return (
            observation_to_tensor(observation),
            torch.tensor(max_scoring_target, dtype=torch.float32),
            torch.tensor(all_scores, dtype=torch.float32),
        )
