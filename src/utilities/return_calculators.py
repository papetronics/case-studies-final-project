from abc import ABC, abstractmethod

from src.utilities.episode import Episode


class ReturnCalculator(ABC):
    @abstractmethod
    def calculate_returns(self, episode: Episode) -> list[float]:
        pass


class MonteCarloReturnCalculator(ReturnCalculator):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def calculate_returns(self, episode: Episode) -> list[float]:
        num_steps = len(episode.states)
        episode_return = float(episode.reward)
        returns = [episode_return] * num_steps
        return returns


class TD0ReturnCalculator(ReturnCalculator):
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def calculate_returns(self, episode: Episode) -> list[float]:
        mc_calculator = MonteCarloReturnCalculator(self.gamma)
        return mc_calculator.calculate_returns(episode)


class TDLambdaReturnCalculator(ReturnCalculator):
    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        self.gamma = gamma
        self.lambda_ = lambda_

    def calculate_returns(self, episode: Episode) -> list[float]:
        mc_calculator = MonteCarloReturnCalculator(self.gamma)
        return mc_calculator.calculate_returns(episode)
