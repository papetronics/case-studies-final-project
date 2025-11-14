import gymnasium as gym
import numpy as np
import pytorch_lightning as lightning
import torch

from environments.full_yahtzee_env import Action, Observation
from utilities.activation_functions import ActivationFunctionName
from utilities.return_calculators import MonteCarloReturnCalculator, ReturnCalculator

from .model import YahtzeeAgent, phi, sample_action
from .self_play_dataset import EpisodeBatch


class SingleTurnScoreMaximizerREINFORCETrainer(lightning.LightningModule):
    """PyTorch Lightning trainer for single-turn Yahtzee score maximization using REINFORCE."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_size: int,
        learning_rate: float,
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

        self.policy_net: YahtzeeAgent = YahtzeeAgent(
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
        )

        self.learning_rate: float = learning_rate
        self.max_epochs: int = max_epochs
        self.min_lr_ratio: float = min_lr_ratio
        self.gamma_max: float = gamma_max
        self.gamma_min: float = gamma_min

        self.return_calculator: ReturnCalculator = return_calculator or MonteCarloReturnCalculator()
        self.return_calculator.gamma = self.gamma_min

        self.full_env: gym.Env[Observation, Action] = gym.make("FullYahtzee-v1")  # For validation

    def run_full_game_episode(self) -> float:
        """Run a complete Yahtzee game using the full environment and return total score."""
        observation, _ = self.full_env.reset()
        total_score = 0.0

        while True:
            with torch.no_grad():
                # Use the trained policy to select actions
                input_tensor = phi(
                    observation, self.policy_net.bonus_flags, self.policy_net.device
                ).unsqueeze(0)
                rolling_probs, scoring_probs, v_est = self.policy_net.forward(input_tensor)
                actions, _, _ = sample_action(rolling_probs, scoring_probs, v_est)
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

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Perform a training step using REINFORCE algorithm with vectorized operations."""
        # batch is an EpisodeBatch dict with keys:
        # - "states": (BATCH_SIZE, 3, state_size) float32
        # - "rolling_actions": (BATCH_SIZE, 3, 5) int
        # - "scoring_actions": (BATCH_SIZE, 3) int
        # - "rewards": (BATCH_SIZE, 3) float32
        # - "next_states": (BATCH_SIZE, 3, state_size) float32
        # - "phases": (BATCH_SIZE, 3) int

        states = batch["states"]
        rolling_actions = batch["rolling_actions"]
        scoring_actions = batch["scoring_actions"]
        rewards = batch["rewards"]
        # next_states = batch["next_states"]  # Not used yet
        phases = batch["phases"]

        batch_size = states.shape[0]
        num_steps = states.shape[1]

        # Reshape to (BATCH_SIZE * 3, ...) to process all steps together
        states_flat = states.view(-1, states.shape[2])  # (BATCH_SIZE * 3, state_size)
        rolling_actions_flat = rolling_actions.view(-1, 5)  # (BATCH_SIZE * 3, 5)
        scoring_actions_flat = scoring_actions.view(-1)  # (BATCH_SIZE * 3,)
        rewards_flat = rewards.view(-1)  # (BATCH_SIZE * 3,)
        phases_flat = phases.view(-1)  # (BATCH_SIZE * 3,)

        # Forward pass through current policy to get probabilities and value estimates
        rolling_probs, scoring_probs, v_ests = self.policy_net.forward(states_flat)

        # Recompute log probabilities from stored actions
        rolling_dist = torch.distributions.Bernoulli(rolling_probs)
        rolling_log_probs = rolling_dist.log_prob(rolling_actions_flat).sum(
            dim=1
        )  # (BATCH_SIZE * 3,)

        scoring_dist = torch.distributions.Categorical(scoring_probs)
        scoring_log_probs = scoring_dist.log_prob(scoring_actions_flat)  # (BATCH_SIZE * 3,)

        # Select the appropriate log prob based on phase
        log_probs = torch.where(
            phases_flat == 0, rolling_log_probs, scoring_log_probs
        )  # (BATCH_SIZE * 3,)

        # Calculate returns using Monte Carlo (backward pass through episodes)
        gamma = self.return_calculator.gamma
        returns = torch.zeros_like(rewards_flat)  # (BATCH_SIZE * 3,)

        for batch_idx_inner in range(batch_size):
            g = 0.0
            for t in reversed(range(num_steps)):
                flat_idx = batch_idx_inner * num_steps + t
                g = rewards_flat[flat_idx] + gamma * g
                returns[flat_idx] = g

        # Calculate advantages
        advantages = returns - v_ests.detach().squeeze()

        # Normalize advantages
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate policy loss
        policy_loss = -(log_probs * normalized_advantages).mean()

        # Calculate average reward per episode (last step contains full return)
        episode_returns = returns.view(batch_size, num_steps)[:, -1]  # (BATCH_SIZE,)
        avg_reward = episode_returns.mean()

        # Calculate value loss
        v_loss = torch.nn.functional.mse_loss(v_ests.squeeze(), returns)

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
        if self.full_env is not None:
            self.full_env.close()
