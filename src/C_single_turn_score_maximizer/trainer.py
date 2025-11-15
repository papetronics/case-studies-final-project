from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytorch_lightning as lightning
import torch

from environments.full_yahtzee_env import Action, Observation
from utilities.activation_functions import ActivationFunctionName
from utilities.diagnostics import (
    compute_action_concentration,
    compute_advantage_stats,
    compute_critic_explained_variance,
    compute_entropy_stats,
    compute_kl_divergence,
    compute_phase_balance,
    compute_return_stats,
    compute_rolling_mask_diversity,
)
from utilities.return_calculators import MonteCarloReturnCalculator, ReturnCalculator

from .model import YahtzeeAgent, phi, sample_action
from .self_play_dataset import EpisodeBatch


class SingleTurnScoreMaximizerREINFORCETrainer(lightning.LightningModule):
    """PyTorch Lightning trainer for single-turn Yahtzee score maximization using REINFORCE.

    DIAGNOSTIC QUICK REFERENCE:
    ✅ Healthy training:
       - train/avg_reward: steadily rising
       - train/critic_ev: 0.3-0.7 and trending up
       - train/entropy_roll: 1.0-2.4 (gradual decline from high)
       - train/kl_roll, train/kl_score: 0.002-0.02 (nonzero movement)
       - train/score_top1: <0.7 (not collapsed)
       - diagnostics/training_health: >0.4

    🚨 Red flags:
       - Plateauing: avg_reward flat + KL ≈ 0 = policy frozen
       - Collapse: entropy_roll <1.2 early OR score_top1 >0.7 = deterministic too soon
       - Bad critic: critic_ev stuck at 0 or negative = not learning
       - Phase imbalance: adv_roll_std ≪ adv_score_std (>10x diff) = one phase dwarfed

    See DIAGNOSTICS.md for full details.
    """

    def __init__(  # noqa: PLR0913
        self,
        hidden_size: int,
        learning_rate: float,
        num_hidden: int,
        dropout_rate: float,
        activation_function: ActivationFunctionName,
        epochs: int,
        min_lr_ratio: float,
        gamma_max: float,
        gamma_min: float,
        return_calculator: ReturnCalculator | None = None,
    ):
        super().__init__()

        self.policy_net: YahtzeeAgent = YahtzeeAgent(
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
        )

        self.learning_rate: float = learning_rate
        self.max_epochs: int = epochs
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

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002, PLR0915, C901
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
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=False)

        # ===== DIAGNOSTIC METRICS =====
        # Advantage stats: adv_mean ~0 (by construction), adv_std nonzero (signal strength)
        # Red flag: adv_std → 0 = tiny gradients; roll vs score std differs >10x = phase imbalance
        adv_stats = compute_advantage_stats(advantages, phases_flat)
        self.log_dict({f"train/{k}": v for k, v in adv_stats.items()}, prog_bar=False)

        # Return std: high early (stochastic), stabilizes; collapsing to tiny values = degenerate policy
        ret_stats = compute_return_stats(episode_returns)
        self.log_dict({f"train/{k}": v for k, v in ret_stats.items()}, prog_bar=False)

        # Critic EV: Good = 0.3-0.7, rising. Red flags: stuck at 0 (not learning), <0 (harmful), wild swings (unstable)
        critic_ev = compute_critic_explained_variance(returns, v_ests)
        self.log("train/critic_ev", critic_ev, prog_bar=False)
        if hasattr(self, "_epoch_metrics"):
            self._epoch_metrics["critic_ev"].append(float(critic_ev.item()))

        # Entropy: roll ~2.0-2.4 early → ~1.0 later; score moderate, gradual decline
        # Red flags: roll <1.2 early = premature collapse; score near-zero = deterministic too soon
        entropy_stats = compute_entropy_stats(rolling_probs, scoring_probs, phases_flat)
        self.log_dict({f"train/{k}": v for k, v in entropy_stats.items()}, prog_bar=False)
        if hasattr(self, "_epoch_metrics"):
            self._epoch_metrics["entropy_roll"].append(float(entropy_stats["entropy_roll"].item()))
            self._epoch_metrics["entropy_score"].append(
                float(entropy_stats["entropy_score"].item())
            )

        # Concentration: top1 starts ~0.2-0.4, climbs slowly; top3 >0.6 early
        # Red flag: top1 >0.7 in first 10-20% of training = over-confident, collapsed
        concentration_stats = compute_action_concentration(scoring_probs)
        self.log_dict({f"train/{k}": v for k, v in concentration_stats.items()}, prog_bar=False)

        # Mask diversity: high and steady; monotonic decline = converging to "always keep X" rut
        mask_diversity = compute_rolling_mask_diversity(rolling_actions_flat, phases_flat)
        if mask_diversity is not None:
            self.log("train/roll_mask_diversity", mask_diversity, prog_bar=False)

        # KL divergence: Good = 0.002-0.02 (policy moving). Red flags: ≈0 = frozen; sudden spikes = too hot
        # Initialize caches if needed
        with torch.no_grad():
            if not hasattr(self, "_prev_roll_p"):
                self._prev_roll_p = rolling_probs.detach()
                self._prev_score_p = scoring_probs.detach()

        kl_stats, self._prev_roll_p, self._prev_score_p = compute_kl_divergence(
            rolling_probs, scoring_probs, phases_flat, self._prev_roll_p, self._prev_score_p
        )
        if kl_stats["kl_roll"] is not None:
            self.log("train/kl_roll", kl_stats["kl_roll"], prog_bar=False)
            if hasattr(self, "_epoch_metrics"):
                self._epoch_metrics["kl_roll"].append(float(kl_stats["kl_roll"].item()))
        if kl_stats["kl_score"] is not None:
            self.log("train/kl_score", kl_stats["kl_score"], prog_bar=False)
            if hasattr(self, "_epoch_metrics"):
                self._epoch_metrics["kl_score"].append(float(kl_stats["kl_score"].item()))

        # Phase balance: should be ~0.67 (2 rolls + 1 score per turn); extreme drift = bad logic
        phase_balance = compute_phase_balance(phases_flat)
        self.log("train/frac_roll_steps", phase_balance, prog_bar=False)

        return policy_loss + 0.05 * v_loss

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Add custom gradient clipping that logs pre-clipping gradient norms."""
        cv: int | float = gradient_clip_val if gradient_clip_val is not None else 0.5

        # global grad norm before clipping
        parameters = [p for p in self.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return

        # L2 norm of all grads
        # (detach so we don't mess with autograd graph)
        grads = [cast("torch.Tensor", p.grad).detach().flatten() for p in parameters]
        flat = torch.cat(grads)
        total_norm = flat.norm(2)

        # log grad norm
        self.log(
            "train/grad_norm",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        clipped_flag = float(total_norm > cv)
        self.log(
            "train/grad_clipped",
            clipped_flag,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        # let Lightning actually do the clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=cv,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

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

        # Reset epoch-level trackers for custom visualizations
        if not hasattr(self, "_epoch_metrics"):
            self._epoch_metrics: dict[str, list[float]] = {
                "critic_ev": [],
                "entropy_roll": [],
                "entropy_score": [],
                "kl_roll": [],
                "kl_score": [],
            }
        else:
            for key in self._epoch_metrics:
                self._epoch_metrics[key].clear()

    def on_train_epoch_end(self) -> None:
        """Log epoch-level summary statistics and custom visualizations."""
        # Create custom metrics for health monitoring
        if hasattr(self, "_epoch_metrics") and len(self._epoch_metrics.get("critic_ev", [])) > 0:
            # Compute epoch averages
            critic_ev_mean = float(np.mean(self._epoch_metrics["critic_ev"]))
            entropy_roll_mean = float(np.mean(self._epoch_metrics["entropy_roll"]))
            entropy_score_mean = float(np.mean(self._epoch_metrics["entropy_score"]))

            # Health score: combination of critic quality and entropy
            # Healthy: critic_ev 0.3-0.7, entropy_roll ~1.0-2.4, entropy_score moderate
            # Score ranges 0-1; good training should be >0.4
            critic_component = (
                max(0, min(critic_ev_mean / 0.7, 1.0)) * 0.5
            )  # Cap at 0.7, scale to 0-0.5
            roll_ent_component = (
                max(0, min(entropy_roll_mean / 2.4, 1.0)) * 0.25
            )  # Cap at 2.4, scale to 0-0.25
            score_ent_component = (
                max(0, min(entropy_score_mean / 1.5, 1.0)) * 0.25
            )  # Cap at 1.5, scale to 0-0.25
            health_score = critic_component + roll_ent_component + score_ent_component

            self.log("diagnostics/training_health", health_score, prog_bar=False)
            self.log("diagnostics/critic_ev_epoch", critic_ev_mean, prog_bar=False)
            self.log("diagnostics/entropy_roll_epoch", entropy_roll_mean, prog_bar=False)
            self.log("diagnostics/entropy_score_epoch", entropy_score_mean, prog_bar=False)

            # Compute KL statistics if available
            if self._epoch_metrics.get("kl_roll") and self._epoch_metrics.get("kl_score"):
                kl_roll_mean = float(
                    np.mean([k for k in self._epoch_metrics["kl_roll"] if k is not None])
                )
                kl_score_mean = float(
                    np.mean([k for k in self._epoch_metrics["kl_score"] if k is not None])
                )

                # Policy movement: Good = 0.002-0.02; below 0.001 = frozen
                policy_movement = (kl_roll_mean + kl_score_mean) / 2.0
                self.log("diagnostics/policy_movement", policy_movement, prog_bar=False)

    def on_train_start(self) -> None:
        """Initialize environments and configure metric visualizations."""
        # Configure WandB-specific visualizations if using WandB
        if (
            self.logger is not None
            and hasattr(self.logger, "experiment")
            and hasattr(cast("Any", self.logger).experiment, "define_metric")
        ):
            logger = cast("Any", self.logger).experiment

            # Set step metric for all training metrics
            logger.define_metric("train/*", step_metric="trainer/global_step")

            # Group related metrics for better dashboard organization
            # Advantage metrics
            logger.define_metric("train/adv_mean", summary="last")
            logger.define_metric("train/adv_std", summary="mean")

            # Phase-specific advantages (useful to spot if one phase dominates)
            logger.define_metric("train/adv_roll_mean", summary="last")
            logger.define_metric("train/adv_score_mean", summary="last")

            # Critic quality (should trend toward 1.0)
            logger.define_metric("train/critic_ev", summary="max,mean")

            # Entropy (should stay high; low = collapse)
            logger.define_metric("train/entropy_roll", summary="min,mean")
            logger.define_metric("train/entropy_score", summary="min,mean")

            # Concentration (high top1/top3 = policy collapse)
            logger.define_metric("train/score_top1", summary="mean,max")
            logger.define_metric("train/score_top3", summary="mean,max")

            # Diversity (should stay high)
            logger.define_metric("train/roll_mask_diversity", summary="min,mean")

            # KL divergence (how much policy is changing)
            logger.define_metric("train/kl_roll", summary="mean")
            logger.define_metric("train/kl_score", summary="mean")

    def on_train_end(self) -> None:
        """Close validation environments at the end of training."""
        for env in self.validation_envs:
            env.close()
        self.validation_envs.clear()
