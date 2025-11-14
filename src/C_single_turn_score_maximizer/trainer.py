import gymnasium as gym
import numpy as np
import pytorch_lightning as lightning
import torch

from environments.full_yahtzee_env import Action, Observation
from utilities.activation_functions import ActivationFunctionName
from utilities.return_calculators import MonteCarloReturnCalculator, ReturnCalculator

from .model import YahtzeeAgent, phi, sample_action


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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Perform a training step using REINFORCE algorithm with vectorized operations."""
        # batch shape: (BATCH_SIZE, 3, 3)
        # where dim 1 is steps (3 steps per episode)
        # and dim 2 is [return, log_prob, v_est]

        batch = batch.to(self.device)

        # Reshape to (BATCH_SIZE * 3, 3) to process all steps together
        batch_flat = batch.view(-1, 3)

        # Split into components
        returns = batch_flat[:, 0]  # (BATCH_SIZE * 3,)
        log_probs = batch_flat[:, 1]  # (BATCH_SIZE * 3,)
        v_ests = batch_flat[:, 2]  # (BATCH_SIZE * 3,)

        # Calculate advantages (vectorized)
        advantages = returns - v_ests.detach()  # (BATCH_SIZE * 3,)

        # Normalize advantages
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate policy loss (vectorized)
        policy_loss = -(log_probs * normalized_advantages).mean()

        # Calculate average reward per episode (sum of returns for last step of each episode)
        # The last step of each episode (step 2) contains the full return
        # Get returns from the last step of each episode (index 2)
        episode_returns = batch[:, 2, 0]  # (BATCH_SIZE,)
        avg_reward = episode_returns.mean()

        # Calculate value loss (vectorized)
        v_loss = torch.nn.functional.mse_loss(v_ests, returns)

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
