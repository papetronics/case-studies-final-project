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

        self.validation_envs: list[gym.Env[Observation, Action]] = []  # Created on demand

    def run_batched_validation_games(self, num_games: int) -> list[float]:
        """Run multiple Yahtzee games in parallel with batched forward passes.

        Parameters
        ----------
        num_games : int
            Number of parallel games to run

        Returns
        -------
        list[float]
            List of total scores for each game
        """
        # Create environments if needed
        if len(self.validation_envs) < num_games:
            for _ in range(num_games - len(self.validation_envs)):
                self.validation_envs.append(gym.make("FullYahtzee-v1"))

        # Reset all environments
        observations = []
        for env in self.validation_envs[:num_games]:
            obs, _ = env.reset()
            observations.append(obs)

        # Track state for each game
        active_indices = list(range(num_games))
        total_scores = [0.0] * num_games

        # Run all games until completion
        with torch.no_grad():
            while active_indices:
                # Gather observations from active games
                active_observations = [observations[i] for i in active_indices]

                # Batch convert observations to state tensors
                state_tensors = torch.stack(
                    [
                        phi(obs, self.policy_net.bonus_flags, self.policy_net.device)
                        for obs in active_observations
                    ]
                )  # (num_active, state_size)

                # Single batched forward pass
                rolling_probs, scoring_probs, v_ests = self.policy_net.forward(state_tensors)

                # Sample actions for all active games
                actions_list, _, _ = sample_action(rolling_probs, scoring_probs, v_ests)
                rolling_action_tensors, scoring_action_tensors = actions_list

                # Step each active environment
                newly_inactive = []
                for batch_idx, game_idx in enumerate(active_indices):
                    obs = active_observations[batch_idx]
                    env = self.validation_envs[game_idx]

                    # Convert action based on phase
                    action: Action
                    if obs["phase"] == 0:
                        action = {
                            "hold_mask": rolling_action_tensors[batch_idx]
                            .cpu()
                            .numpy()
                            .astype(bool)
                        }
                    else:
                        score_category: int = int(scoring_action_tensors[batch_idx].cpu().item())
                        action = {"score_category": score_category}

                    # Step environment
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    total_scores[game_idx] += float(reward)

                    if terminated or truncated:
                        newly_inactive.append(game_idx)
                    else:
                        observations[game_idx] = next_obs

                # Remove completed games from active list
                for game_idx in newly_inactive:
                    active_indices.remove(game_idx)

        return total_scores

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, float]:  # noqa: ARG002
        """Run validation using batched parallel games."""
        num_validation_games = 1000

        # Run all games in parallel with batched forward passes
        total_scores = self.run_batched_validation_games(num_validation_games)

        mean_total_score = float(np.mean(total_scores))
        std_total_score = float(np.std(total_scores))

        self.log("val/mean_total_score", mean_total_score, prog_bar=True)
        self.log("val/std_total_score", std_total_score, prog_bar=False)

        # Return a dict for PyTorch Lightning compatibility
        return {"val_loss": -mean_total_score}  # Negative because higher scores are better

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Perform a training step using REINFORCE algorithm with vectorized operations."""
        # batch is an EpisodeBatch dict with pre-flattened tensors:
        # - "states": (BATCH_SIZE*3, state_size) float32
        # - "rolling_actions": (BATCH_SIZE*3, 5) int
        # - "scoring_actions": (BATCH_SIZE*3,) int
        # - "rewards": (BATCH_SIZE*3,) float32
        # - "next_states": (BATCH_SIZE*3, state_size) float32
        # - "phases": (BATCH_SIZE*3,) int

        # Dataset pre-flattens, so we just extract directly
        states_flat = batch["states"]
        rolling_actions_flat = batch["rolling_actions"]
        scoring_actions_flat = batch["scoring_actions"]
        rewards_flat = batch["rewards"]
        # next_states = batch["next_states"]  # Not used yet
        phases_flat = batch["phases"]

        # Calculate batch_size and num_steps from flattened shape
        total_steps = states_flat.shape[0]
        num_steps = 3
        batch_size = total_steps // num_steps

        # Forward pass through current policy to get probabilities and value estimates
        rolling_probs, scoring_probs, v_ests = self.policy_net.forward(states_flat)

        # Recompute log probabilities from stored actions
        # Note: Bernoulli.log_prob expects float values, so cast rolling_actions to float
        rolling_dist = torch.distributions.Bernoulli(rolling_probs)
        rolling_log_probs = rolling_dist.log_prob(rolling_actions_flat.float()).sum(
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
        """Close validation environments at the end of training."""
        for env in self.validation_envs:
            env.close()
        self.validation_envs.clear()
