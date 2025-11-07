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
        
        # Calculate returns backward from the final step
        G = 0  # Initialize return
        for t in reversed(range(num_steps)):
            # For the last step, add the episode reward
            if t == num_steps - 1:
                G = float(episode.reward)
            else:
                # Discount the return from the next step
                G = self.gamma * G
            returns.insert(0, G)
        
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