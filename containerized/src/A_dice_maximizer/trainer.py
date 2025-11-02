import torch
import pytorch_lightning as L
import numpy as np
from typing import Dict, Any, Optional

import gymnasium as gym
import yahtzee_gym
from .model import DiceSumMaximizer
from src.return_calculators import ReturnCalculator, MonteCarloReturnCalculator
from src.episode import Episode


class REINFORCEWithBaselineTrainer(L.LightningModule):
    
    def __init__(
        self,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        episodes_per_batch: int = 32,
        baseline_alpha: float = 0.1,
        return_calculator: Optional[ReturnCalculator] = None,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['return_calculator'])
        
        self.policy_net = DiceSumMaximizer(hidden_size=hidden_size)
        
        self.learning_rate = learning_rate
        self.episodes_per_batch = episodes_per_batch
        self.baseline_alpha = baseline_alpha
        
        self.return_calculator = return_calculator or MonteCarloReturnCalculator()
        
        self.baseline = 0.0
        
        self.env = gym.make('Yahtzee-v0')
        
    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        return self.policy_net(observation)
        
    def collect_episode(self) -> Episode:
        episode = Episode()
        observation, _ = self.env.reset()
        
        step_count = 0
        while True:
            action_tensor, log_prob = self.policy_net.sample(observation)
            action = action_tensor.cpu().numpy().astype(bool)

            episode.add_step(observation, action, log_prob)

            observation, reward, terminated, truncated, _ = self.env.step(action)
            step_count += 1
            
            if terminated or truncated:
                episode.set_reward(float(reward))
                break
                
        return episode
        
    def training_step(self, batch, batch_idx):
        episodes = []
        total_reward = 0.0
        
        for _ in range(self.episodes_per_batch):
            episode: Episode = self.collect_episode()
            episodes.append(episode)
            total_reward += episode.reward

        policy_loss = torch.tensor([0.0], device=self.device)

        for episode in episodes:
            returns = self.return_calculator.calculate_returns(episode)
            
            for step_idx, log_prob in enumerate(episode.log_probs):
                step_return = returns[step_idx]
                advantage = step_return - self.baseline
                policy_loss -= log_prob * advantage
                
        policy_loss /= self.episodes_per_batch
        
        avg_reward = total_reward / self.episodes_per_batch
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * avg_reward
        
        self.log('train/policy_loss', policy_loss, prog_bar=True)
        self.log('train/avg_reward', avg_reward, prog_bar=True)
        self.log('train/baseline', self.baseline, prog_bar=False)
        
        return policy_loss
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def on_train_start(self):
        pass
        
    def on_train_end(self):
        if self.env is not None:
            self.env.close()