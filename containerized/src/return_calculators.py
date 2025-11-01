import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class ReturnCalculator(ABC):
    @abstractmethod
    def calculate_returns(self, episode) -> List[float]:
        pass


class MonteCarloReturnCalculator(ReturnCalculator):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
    
    def calculate_returns(self, episode) -> List[float]:
        num_steps = len(episode.states)
        returns = []
        
        # Calculate discounted return for each step
        for t in range(num_steps):
            G_t = 0.0
            for k in range(t, num_steps):
                G_t += (self.gamma ** (k - t)) * episode.rewards[k]
            returns.append(G_t)
            
        return returns


class TD0ReturnCalculator(ReturnCalculator):
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
    
    def calculate_returns(self, episode) -> List[float]:
        mc_calculator = MonteCarloReturnCalculator(self.gamma)
        return mc_calculator.calculate_returns(episode)


class TDLambdaReturnCalculator(ReturnCalculator):
    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        self.gamma = gamma
        self.lambda_ = lambda_
    
    def calculate_returns(self, episode) -> List[float]:
        mc_calculator = MonteCarloReturnCalculator(self.gamma)
        return mc_calculator.calculate_returns(episode)