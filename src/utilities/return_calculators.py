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


class TD0ReturnCalculator(ReturnCalculator):
    """TD(0) return calculator - uses one-step bootstrapping."""

    def __init__(self, gamma: float = 1.0):
        self.gamma: float = gamma

    def calculate_returns(self, episode: Episode) -> list[float]:
        """Calculate TD(0) targets for each time step: r_t + gamma * V(s_{t+1}).

        Note: This method is not used in the vectorized training loop.
        TD(0) targets are calculated directly in the trainer using the batch data.
        """
        # This is kept for compatibility but won't be called in practice
        # The actual TD(0) calculation happens in trainer.py using vectorized operations
        return [0.0] * len(episode.states)
