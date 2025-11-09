from abc import ABC, abstractmethod

from utilities.episode import Episode


class ReturnCalculator(ABC):
    """Abstract base class for return calculators."""

    gamma: float

    @abstractmethod
    def calculate_returns(self, episode: Episode) -> list[float]:
        """Calculate returns for each time step in the episode."""
        pass


class MonteCarloReturnCalculator(ReturnCalculator):
    """Monte Carlo return calculator."""

    def __init__(self, gamma: float = 1.0):
        self.gamma: float = gamma

    def calculate_returns(self, episode: Episode) -> list[float]:
        """Calculate returns for each time step in the episode using Monte Carlo method."""
        num_steps = len(episode.states)
        returns: list[float] = []

        # Calculate returns backward from the final step
        g: float = 0  # Initialize return
        for t in reversed(range(num_steps)):
            # For the last step, add the episode reward
            g = float(episode.reward) if t == num_steps - 1 else self.gamma * g
            returns.insert(0, g)

        return returns
