from typing import SupportsFloat, cast

import gymnasium as gym
import numpy as np
import pytorch_lightning as lightning
import torch

from environments.full_yahtzee_env import Action, Observation, YahtzeeEnv
from utilities.episode import Episode
from utilities.return_calculators import MonteCarloReturnCalculator, ReturnCalculator

from .model import ActivationFunctionName, TurnScoreMaximizer


class SingleTurnScoreMaximizerREINFORCETrainer(lightning.LightningModule):
    """PyTorch Lightning trainer for single-turn Yahtzee score maximization using REINFORCE."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_size: int,
        learning_rate: float,
        episodes_per_batch: int,
        num_hidden: int,
        dropout_rate: float,
        activation_function: ActivationFunctionName,
        max_epochs: int,
        min_lr_ratio: float,
        gamma_max: float,
        gamma_min: float,
        return_calculator: ReturnCalculator | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["return_calculator"])

        self.policy_net: TurnScoreMaximizer = TurnScoreMaximizer(
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
        )

        self.learning_rate: float = learning_rate
        self.episodes_per_batch: int = episodes_per_batch
        self.max_epochs: int = max_epochs
        self.min_lr_ratio: float = min_lr_ratio
        self.gamma_max: float = gamma_max
        self.gamma_min: float = gamma_min

        self.return_calculator: ReturnCalculator = return_calculator or MonteCarloReturnCalculator()
        self.return_calculator.gamma = self.gamma_min

        self.env: gym.Env[Observation, Action] = gym.make("FullYahtzee-v1")
        self.env.reset()
        self.full_env: gym.Env[Observation, Action] = gym.make("FullYahtzee-v1")  # For validation

    def collect_episode(self) -> Episode:
        """Collect a single episode using the current policy."""
        episode = Episode()
        unwrapped: YahtzeeEnv = cast("YahtzeeEnv", self.env.unwrapped)
        observation = unwrapped.observe()

        # This is a bit of a hack, the environment supports full turns, but our model is single-turn
        # so we are just going to cut it off after 3 steps and pretend that is an episode.
        # we reset the environment whenever it terminates, that brings us back to an empty scoresheet.
        reward: SupportsFloat = 0.0

        for _ in range(3):  # roll, roll, score
            actions, log_probs, v_est = self.policy_net.sample_observation(observation)
            rolling_action_tensor, scoring_action_tensor = actions
            rolling_log_prob, scoring_log_prob = log_probs

            action: Action
            if observation["phase"] == 0:
                action = {"hold_mask": rolling_action_tensor.cpu().numpy().astype(bool)}
                log_prob = rolling_log_prob
            else:
                score_category: int = int(scoring_action_tensor.cpu().item())
                action = {"score_category": score_category}
                log_prob = scoring_log_prob

            episode.add_step(observation, action, log_prob, v_est)

            observation, reward, terminated, truncated, _ = self.env.step(action)

            if terminated or truncated:
                observation, _ = self.env.reset()

        episode.set_reward(float(reward))

        # sanity check: after 3 rolls we should always have rolls_used == 0 (new turn) and phase == 0
        assert observation["rolls_used"] == 0 and observation["phase"] == 0

        return episode

    def run_full_game_episode(self) -> float:
        """Run a complete Yahtzee game using the full environment and return total score."""
        observation, _ = self.full_env.reset()
        total_score = 0.0

        while True:
            # Use the trained policy to select actions
            with torch.no_grad():
                actions, _, _ = self.policy_net.sample_observation(observation)
                rolling_action_tensor, scoring_action_tensor = actions

                action: Action
                if observation["phase"] == 0:
                    action = {"hold_mask": rolling_action_tensor.cpu().numpy().astype(bool)}
                else:
                    score_category: int = int(scoring_action_tensor.cpu().item())
                    action = {"score_category": score_category}

            observation, reward, terminated, truncated, _ = self.full_env.step(action)
            total_score += float(reward)

            if terminated or truncated:
                break

        return total_score

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, float]:  # noqa: ARG002
        """Run validation using the full Yahtzee environment."""
        num_validation_games = 50
        total_scores = []

        for _ in range(num_validation_games):
            score = self.run_full_game_episode()
            total_scores.append(score)

        mean_total_score = float(np.mean(total_scores))
        std_total_score = float(np.std(total_scores))

        self.log("val/mean_total_score", mean_total_score, prog_bar=True)
        self.log("val/std_total_score", std_total_score, prog_bar=False)

        # Return a dict for PyTorch Lightning compatibility
        return {"val_loss": -mean_total_score}  # Negative because higher scores are better

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Perform a training step using REINFORCE algorithm."""
        total_reward = 0.0

        # Collect all advantages and log probabilities
        advantages = []
        log_probs_list = []
        v_ests_list = []
        returns_list = []

        for _ in range(self.episodes_per_batch):
            episode: Episode = self.collect_episode()
            total_reward += episode.reward

            returns = self.return_calculator.calculate_returns(episode)

            for step_idx, log_prob in enumerate(episode.log_probs):
                step_return = returns[step_idx]
                v_est = episode.v_ests[step_idx]

                advantage = (
                    step_return - v_est.item()
                )  # use item so we don't double backprop through v_est

                advantages.append(advantage)
                log_probs_list.append(log_prob)
                v_ests_list.append(v_est)
                returns_list.append(step_return)

        # Convert to tensor and normalize advantages
        advantages_tensor = torch.tensor(advantages, device=self.device)
        normalized_advantages = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # Calculate policy loss with normalized advantages
        policy_loss = torch.tensor([0.0], device=self.device)
        for log_prob, normalized_advantage in zip(
            log_probs_list, normalized_advantages, strict=True
        ):
            policy_loss -= log_prob * normalized_advantage

        policy_loss /= self.episodes_per_batch

        avg_reward = total_reward / self.episodes_per_batch

        # Calculate value loss (batched)

        ## TODO: try huber loss torch.nn.functional.smooth_l1_loss
        ## TODO: try centering:
        # G = torch.tensor(returns_list, device=self.device, dtype=v_ests_list[0].dtype)
        # Gc = G - G.mean()
        # value_loss = torch.nn.functional.smooth_l1_loss(torch.stack(v_ests_list).squeeze(), Gc)
        v_loss = torch.nn.functional.mse_loss(
            torch.stack(v_ests_list).squeeze(), torch.tensor(returns_list, device=self.device)
        )

        self.log("train/policy_loss", policy_loss, prog_bar=True)
        self.log("train/avg_reward", avg_reward, prog_bar=True)
        self.log("train/v_loss", v_loss, prog_bar=False)
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=False)

        return policy_loss + 0.05 * v_loss

    def configure_optimizers(self):  # noqa: ANN201
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,  # Start at full learning rate
            end_factor=self.min_lr_ratio,  # End at min_lr_ratio of initial LR
            total_iters=self.max_epochs,  # Linear decay over training epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,  # Check every epoch
            },
        }

    def on_train_epoch_start(self) -> None:
        """Update gamma linearly from gamma_min to gamma_max over training."""
        if self.max_epochs > 1:
            # Linear interpolation from gamma_min to gamma_max
            progress = self.current_epoch / (self.max_epochs - 1)
            current_gamma = self.gamma_min + progress * (self.gamma_max - self.gamma_min)
        else:
            current_gamma = self.gamma_max

        self.return_calculator.gamma = current_gamma
        self.log("train/gamma", current_gamma, prog_bar=False)

    def on_train_start(self) -> None:
        """Initialize environments at the start of training."""
        pass

    def on_train_end(self) -> None:
        """Close environments at the end of training."""
        if self.env is not None:
            self.env.close()
        if self.full_env is not None:
            self.full_env.close()
