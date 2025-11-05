import torch
import pytorch_lightning as L
import numpy as np
from typing import Dict, Any, Optional, cast

import gymnasium as gym
from src.environments.full_yahtzee_env import YahtzeeEnv
from .model import TurnScoreMaximizer
from src.utilities.return_calculators import ReturnCalculator, MonteCarloReturnCalculator
from src.utilities.episode import Episode


class SingleTurnScoreMaximizerREINFORCETrainer(L.LightningModule):
    
    def __init__(
        self,
        hidden_size: int,
        learning_rate: float,
        episodes_per_batch: int,
        num_hidden: int,
        dropout_rate: float,
        activation_function: str,
        max_epochs: int,
        min_lr_ratio: float,
        return_calculator: Optional[ReturnCalculator] = None,
        baseline_alpha: float = 0.1,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['return_calculator'])

        self.policy_net = TurnScoreMaximizer(
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            activation_function=activation_function
        )

        self.learning_rate = learning_rate
        self.episodes_per_batch = episodes_per_batch
        self.baseline_alpha = baseline_alpha
        self.max_epochs = max_epochs
        self.min_lr_ratio = min_lr_ratio
        
        self.return_calculator = return_calculator or MonteCarloReturnCalculator()
        
        self.baseline = 0.0
        
        self.env = gym.make('FullYahtzee-v1')
        self.env.reset()
        self.full_env = gym.make('FullYahtzee-v1')  # For validation
        
    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        return self.policy_net(observation)
        
    def collect_episode(self) -> Episode:
        episode = Episode()
        unwrapped: YahtzeeEnv = cast(YahtzeeEnv, self.env.unwrapped)
        observation = unwrapped.observe()

        # This is a bit of a hack, the environment supports full turns, but our model is single-turn
        # so we are just going to cut it off after 3 steps and pretend that is an episode.
        # we reset the environment whenever it terminates, that brings us back to an empty scoresheet.
        reward = 0.0

        for _ in range(3):  # roll, roll, score
            actions, log_probs = self.policy_net.sample_observation(observation)
            rolling_action_tensor, scoring_action_tensor = actions
            rolling_log_prob, scoring_log_prob = log_probs

            action = {}
            if observation['phase'] == 0:
                action = {
                    'hold_mask': rolling_action_tensor.cpu().numpy().astype(bool)
                }
                log_prob = rolling_log_prob
            else:
                action = {
                    'score_category': scoring_action_tensor.cpu().item()
                }
                log_prob = scoring_log_prob
            
            episode.add_step(observation, action, log_prob)

            observation, reward, terminated, truncated, _ = self.env.step(action)
            
            

            if terminated or truncated:
                observation, _ = self.env.reset()

        episode.set_reward(float(reward))
        # print ("Episode reward:", episode.reward)

        # sanity check: after 3 rolls we should always have rolls_used == 0 (new turn) and phase == 0
        assert observation['rolls_used'] == 0 and observation['phase'] == 0
                
        return episode
    
    def run_full_game_episode(self) -> float:
        """Run a complete Yahtzee game using the full environment and return total score."""
        observation, _ = self.full_env.reset()
        total_score = 0.0
        
        while True:
            # Use the trained policy to select actions
            with torch.no_grad():
                actions, _ = self.policy_net.sample_observation(observation)
                rolling_action_tensor, scoring_action_tensor = actions

                action = {}
                if observation['phase'] == 0:
                    action = {
                        'hold_mask': rolling_action_tensor.cpu().numpy().astype(bool)
                    }
                else:
                    action = {
                        'score_category': scoring_action_tensor.cpu().item()
                    }

            observation, reward, terminated, truncated, _ = self.full_env.step(action)
            total_score += float(reward)
            
            if terminated or truncated:
                break
                
        return total_score
    
    def validation_step(self, batch, batch_idx):
        """Run validation using the full Yahtzee environment."""
        num_validation_games = 50
        total_scores = []
        
        for _ in range(num_validation_games):
            score = self.run_full_game_episode()
            total_scores.append(score)
        
        mean_total_score = float(np.mean(total_scores))
        std_total_score = float(np.std(total_scores))
        
        self.log('val/mean_total_score', mean_total_score, prog_bar=True)
        self.log('val/std_total_score', std_total_score, prog_bar=False)
        
        # Return a dict for PyTorch Lightning compatibility
        return {'val_loss': -mean_total_score}  # Negative because higher scores are better
        
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
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=False)
        
        return policy_loss
            
    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,  # Start at full learning rate
            end_factor=self.min_lr_ratio,  # End at min_lr_ratio of initial LR
            total_iters=self.max_epochs  # Linear decay over training epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1            # Check every epoch
            }
        }
    
    def on_train_start(self):
        pass
        
    def on_train_end(self):
        if self.env is not None:
            self.env.close()
        if self.full_env is not None:
            self.full_env.close()