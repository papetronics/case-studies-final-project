from dataclasses import dataclass
import torch
import numpy as np

from src.B_supervised_scorer.model import observation_to_tensor

@dataclass
class ScoreCategory:
    ONES: int = 0
    TWOS: int = 1
    THREES: int = 2
    FOURS: int = 3
    FIVES: int = 4
    SIXES: int = 5
    THREE_OF_A_KIND: int = 6
    FOUR_OF_A_KIND: int = 7
    FULL_HOUSE: int = 8
    SMALL_STRAIGHT: int = 9
    LARGE_STRAIGHT: int = 10
    YAHTZEE: int = 11
    CHANCE: int = 12

    LABELS = [
        "Ones",
        "Twos",
        "Threes",
        "Fours",
        "Fives",
        "Sixes",
        "Three of a Kind",
        "Four of a Kind",
        "Full House",
        "Small Straight",
        "Large Straight",
        "Yahtzee",
        "Chance"
    ]


class GreedyScoringDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
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

        max_scoring_option = self._get_max_scoring_option(dice, open_scores)
        observation = {
            'dice': dice,
            'available_categories': open_scores,
            'rolls_used': 2
        }

        return observation_to_tensor(observation), torch.tensor(max_scoring_option, dtype=torch.long)

    def _get_max_scoring_option(self, dice: np.ndarray, open_scores: np.ndarray) -> int:
        """Given a set of dice, compute the maximum scoring option."""
        # Count occurrences of each die face (1-6)
        counts = np.bincount(dice, minlength=7)[1:]  # Ignore index 0
        upper_scores = counts * np.arange(1, 7)

        # Calculate scores for lower section
        three_of_a_kind = np.sum(dice) if np.any(counts >= 3) else 0
        four_of_a_kind = np.sum(dice) if np.any(counts >= 4) else 0
        full_house = 25 if (np.any(counts == 3) and np.any(counts == 2)) else 0
        small_straight = 30 if (self._has_small_straight(counts)) else 0
        large_straight = 40 if (self._has_large_straight(counts)) else 0
        yahtzee = 50 if np.any(counts == 5) else 0
        chance = np.sum(dice)

        lower_scores = np.array([
            three_of_a_kind,
            four_of_a_kind,
            full_house,
            small_straight,
            large_straight,
            yahtzee,
            chance
        ])

        all_scores = np.concatenate([upper_scores, lower_scores])

        # Mask out closed categories
        all_scores = all_scores * open_scores

        return int(np.argmax(all_scores))
    
    def _has_small_straight(self, counts: np.ndarray) -> bool:
        """Check for small straight (4 consecutive numbers)."""
        straights = [
            counts[0:4],  # 1-4
            counts[1:5],  # 2-5
            counts[2:6]   # 3-6
        ]
        return any(np.all(straight >= 1) for straight in straights)
    
    def _has_large_straight(self, counts: np.ndarray) -> bool:
        """Check for large straight (5 consecutive numbers)."""
        return bool(np.all(counts[0:5] >= 1) or np.all(counts[1:6] >= 1))