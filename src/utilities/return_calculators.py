from abc import ABC, abstractmethod

from src.utilities.episode import Episode


class ReturnCalculator(ABC):
    """Abstract base class for return calculators."""

    @abstractmethod
    def calculate_returns(self, episode: Episode) -> list[float]:
        """Calculate returns for each time step in the episode."""
        pass


class MonteCarloReturnCalculator(ReturnCalculator):
    """Monte Carlo return calculator."""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def calculate_returns(self, episode: Episode) -> list[float]:
        """Calculate returns for each time step in the episode using Monte Carlo method."""
        num_steps = len(episode.states)
        episode_return = float(episode.reward)
        returns = [episode_return] * num_steps
        return returns
