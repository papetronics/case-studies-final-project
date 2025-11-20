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
from yahtzee_agent.features import PhiFeature

from .model import (
    RollingActionRepresentation,
    YahtzeeAgent,
    convert_rolling_action_to_hold_mask,
    phi,
    sample_action,
    select_action,
)
from .self_play_dataset import EpisodeBatch


class YahtzeeAgentTrainer(lightning.LightningModule):
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
        entropy_coeff_rolling_max: float,
        entropy_coeff_rolling_min: float,
        entropy_coeff_scoring_max: float,
        entropy_coeff_scoring_min: float,
        entropy_hold_period: float,
        entropy_anneal_period: float,
        critic_coeff: float,
        num_steps_per_episode: int,
        features: list[PhiFeature],
        rolling_action_representation: str,
        he_kaiming_initialization: bool,
        return_calculator: ReturnCalculator | None = None,
        mode: str = "reinforce",
    ):
        super().__init__()

        # Convert string to enum
        self.rolling_action_representation = RollingActionRepresentation(
            rolling_action_representation
        )

        self.policy_net: YahtzeeAgent = YahtzeeAgent(
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
            features=features,
            rolling_action_representation=self.rolling_action_representation,
            he_kaiming_initialization=he_kaiming_initialization,
        )

        self.learning_rate: float = learning_rate
        self.max_epochs: int = epochs
        self.min_lr_ratio: float = min_lr_ratio
        self.gamma_max: float = gamma_max
        self.gamma_min: float = gamma_min
        self.entropy_coeff_rolling_max: float = entropy_coeff_rolling_max
        self.entropy_coeff_rolling_min: float = entropy_coeff_rolling_min
        self.entropy_coeff_scoring_max: float = entropy_coeff_scoring_max
        self.entropy_coeff_scoring_min: float = entropy_coeff_scoring_min
        self.entropy_hold_period: float = entropy_hold_period
        self.entropy_anneal_period: float = entropy_anneal_period
        self.critic_coeff: float = critic_coeff
        self.num_steps_per_episode: int = num_steps_per_episode

        self.return_calculator: ReturnCalculator = return_calculator or MonteCarloReturnCalculator()
        self.return_calculator.gamma = self.gamma_min

        self.mode: str = mode
        self.env_steps_seen: int = 0
        self.examples_seen: int = 0

        # Collector and dataset will be set by main
        self.collector: Any | None = None
        self.algorithm: str = mode

        self.validation_envs: list[gym.Env[Observation, Action]] = []  # Created on demand

    def run_batched_validation_games(  # noqa: C901, PLR0912
        self, num_games: int, run_deterministic: bool = True, run_stochastic: bool = False
    ) -> tuple[list[float], list[float]]:
        """Run multiple Yahtzee games in parallel with both deterministic and stochastic action selection.

        Runs num_games with deterministic actions and num_games with stochastic actions
        simultaneously, using 2*num_games environments total.

        Parameters
        ----------
        num_games : int
            Number of parallel games to run for each mode (deterministic and stochastic)
        run_deterministic : bool, optional
            Whether to run deterministic evaluation (default: True)
        run_stochastic : bool, optional
            Whether to run stochastic evaluation (default: False)

        Returns
        -------
        tuple[list[float], list[float]]
            (deterministic_scores, stochastic_scores) - Lists of total scores for each game
        """
        # Calculate number of environments needed based on which modes are enabled
        num_det = num_games if run_deterministic else 0
        num_stoch = num_games if run_stochastic else 0
        total_envs_needed = num_det + num_stoch

        # Early return if neither mode is enabled
        if total_envs_needed == 0:
            return [], []

        # Create environments if needed
        if len(self.validation_envs) < total_envs_needed:
            for _ in range(total_envs_needed - len(self.validation_envs)):
                self.validation_envs.append(gym.make("FullYahtzee-v1"))

        # Reset all environments
        observations = []
        for env in self.validation_envs[:total_envs_needed]:
            obs, _ = env.reset()
            observations.append(obs)

        # Track state for each game
        # First num_det are deterministic, next num_stoch are stochastic
        active_indices = list(range(total_envs_needed))
        total_scores = [0.0] * total_envs_needed

        # Run all games until completion
        with torch.no_grad():
            while active_indices:
                # Gather observations from active games
                active_observations = [observations[i] for i in active_indices]

                # Batch convert observations to state tensors
                state_tensors = torch.stack(
                    [
                        phi(
                            obs,
                            self.policy_net.features,
                            self.policy_net.device,
                        )
                        for obs in active_observations
                    ]
                )  # (num_active, state_size)

                # Single batched forward pass
                rolling_probs, scoring_probs, v_ests = self.policy_net.forward(state_tensors)

                # Separate deterministic and stochastic actions
                # Deterministic for first num_det, stochastic for remaining
                det_mask = torch.tensor(
                    [idx < num_det for idx in active_indices], device=rolling_probs.device
                )

                # Get actions for both modes
                det_actions = select_action(
                    rolling_probs, scoring_probs, self.rolling_action_representation
                )
                stoch_actions, _, _ = sample_action(
                    rolling_probs, scoring_probs, v_ests, self.rolling_action_representation
                )

                # Select appropriate actions based on game index
                # For BERNOULLI representation, rolling actions are (batch, 5), so expand mask
                if self.rolling_action_representation == RollingActionRepresentation.BERNOULLI:
                    rolling_mask = det_mask.unsqueeze(1)  # (batch,) -> (batch, 1)
                else:
                    rolling_mask = det_mask

                rolling_action_tensors = torch.where(rolling_mask, det_actions[0], stoch_actions[0])
                scoring_action_tensors = torch.where(det_mask, det_actions[1], stoch_actions[1])

                # Step each active environment
                newly_inactive = []
                for batch_idx, game_idx in enumerate(active_indices):
                    obs = active_observations[batch_idx]
                    env = self.validation_envs[game_idx]

                    # Convert action based on phase
                    action: Action
                    if obs["phase"] == 0:
                        mask = convert_rolling_action_to_hold_mask(
                            rolling_action_tensors[batch_idx], self.rolling_action_representation
                        )
                        action = {"hold_mask": mask}
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

        # Split results into deterministic and stochastic
        deterministic_scores = total_scores[:num_det]
        stochastic_scores = total_scores[num_det:]

        return deterministic_scores, stochastic_scores

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, float]:  # noqa: ARG002
        """Run validation using batched parallel games."""
        num_validation_games = 1000

        # Run all games in parallel with batched forward passes (only deterministic by default)
        det_scores, stoch_scores = self.run_batched_validation_games(
            num_validation_games, run_deterministic=True, run_stochastic=False
        )

        # Log deterministic results if available
        if det_scores:
            det_mean = float(np.mean(det_scores))
            det_std = float(np.std(det_scores))
            self.log("val/mean_total_score_det", det_mean, prog_bar=True)
            self.log("val/std_total_score_det", det_std, prog_bar=False)
        else:
            det_mean = 0.0

        # Log stochastic results if available
        if stoch_scores:
            stoch_mean = float(np.mean(stoch_scores))
            stoch_std = float(np.std(stoch_scores))
            self.log("val/mean_total_score_stoch", stoch_mean, prog_bar=False)
            self.log("val/std_total_score_stoch", stoch_std, prog_bar=False)

        # Return a dict for PyTorch Lightning compatibility (use deterministic for loss)
        return {"val_loss": -det_mean}  # Negative because higher scores are better

    def run_sweep_test_evaluation(
        self, run_deterministic: bool = True, run_stochastic: bool = False
    ) -> None:
        """Run comprehensive test evaluation for hyperparameter sweeps.

        Runs 10000 games with deterministic and/or stochastic action selection.
        Logs test/mean_total_score_det, test/mean_total_score_stoch and their stds.
        This is called at the end of training.

        Parameters
        ----------
        run_deterministic : bool, optional
            Whether to run deterministic evaluation (default: True)
        run_stochastic : bool, optional
            Whether to run stochastic evaluation (default: False)
        """
        num_test_games = 10000

        # Run all games in parallel with batched forward passes
        det_scores, stoch_scores = self.run_batched_validation_games(
            num_test_games, run_deterministic=run_deterministic, run_stochastic=run_stochastic
        )

        # Log deterministic results if available
        if det_scores:
            det_mean = float(np.mean(det_scores))
            det_std = float(np.std(det_scores))
            self.log("test/mean_total_score_det", det_mean, prog_bar=True)
            self.log("test/std_total_score_det", det_std, prog_bar=False)

        # Log stochastic results if available
        if stoch_scores:
            stoch_mean = float(np.mean(stoch_scores))
            stoch_std = float(np.std(stoch_scores))
            self.log("test/mean_total_score_stoch", stoch_mean, prog_bar=False)
            self.log("test/std_total_score_stoch", stoch_std, prog_bar=False)

    def get_entropy_coefs(self) -> tuple[float, float]:
        """Get current entropy coefficients for rolling and scoring heads.

        Schedule:
        - Hold at max for entropy_hold_period (default 40%) of training
        - Anneal linearly to min over entropy_anneal_period (default 35%) of training
        - Hold at min for remaining training

        Returns
        -------
        tuple[float, float]
            (rolling_coef, scoring_coef) - current entropy coefficients
        """
        # Calculate epoch boundaries
        hold_end_epoch = self.max_epochs * self.entropy_hold_period
        anneal_end_epoch = hold_end_epoch + self.max_epochs * self.entropy_anneal_period

        # Determine current phase and progress
        if self.current_epoch < hold_end_epoch:
            # Phase 1: Hold at max
            rolling_coef = self.entropy_coeff_rolling_max
            scoring_coef = self.entropy_coeff_scoring_max
        elif self.current_epoch < anneal_end_epoch:
            # Phase 2: Anneal from max to min
            anneal_progress = (self.current_epoch - hold_end_epoch) / (
                self.max_epochs * self.entropy_anneal_period
            )
            rolling_coef = (
                self.entropy_coeff_rolling_max
                + (self.entropy_coeff_rolling_min - self.entropy_coeff_rolling_max)
                * anneal_progress
            )
            scoring_coef = (
                self.entropy_coeff_scoring_max
                + (self.entropy_coeff_scoring_min - self.entropy_coeff_scoring_max)
                * anneal_progress
            )
        else:
            # Phase 3: Hold at min
            rolling_coef = self.entropy_coeff_rolling_min
            scoring_coef = self.entropy_coeff_scoring_min

        return rolling_coef, scoring_coef

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002, C901, PLR0912, PLR0915
        """Perform a training step with mode-specific loss computation."""
        # Extract batch data (batch_size timesteps)
        states_flat = batch["states"]
        rolling_actions_flat = batch["rolling_actions"]
        scoring_actions_flat = batch["scoring_actions"]
        phases_flat = batch["phases"]
        # logp_old available in batch for future importance sampling support

        batch_size = states_flat.shape[0]

        # Track custom counters
        self.examples_seen += batch_size
        self.env_steps_seen += batch_size

        # Forward pass through current policy
        rolling_probs, scoring_probs, v_ests = self.policy_net.forward(states_flat)

        # Recompute log probabilities from stored actions
        rolling_dist: torch.distributions.Distribution
        if self.rolling_action_representation == RollingActionRepresentation.CATEGORICAL:
            rolling_dist = torch.distributions.Categorical(rolling_probs)
            rolling_log_probs = rolling_dist.log_prob(rolling_actions_flat.float())
        else:  # BERNOULLI
            rolling_dist = torch.distributions.Bernoulli(rolling_probs)
            rolling_log_probs = rolling_dist.log_prob(rolling_actions_flat.float()).sum(dim=1)

        scoring_dist = torch.distributions.Categorical(scoring_probs)
        scoring_log_probs = scoring_dist.log_prob(scoring_actions_flat)

        # Select log prob based on phase
        log_probs = torch.where(phases_flat == 0, rolling_log_probs, scoring_log_probs)

        # Branch on mode for loss computation
        if self.mode == "reinforce":
            # REINFORCE: use MC returns as targets
            returns = batch.get("returns")
            if returns is None:
                raise ValueError("Returns not computed for REINFORCE mode")  # noqa: TRY003

            # Value used as baseline
            advantages = returns - v_ests.detach().squeeze()
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss with importance sampling support
            policy_loss = -(log_probs * normalized_advantages).mean()

            # Critic loss to learn value function
            v_loss = torch.nn.functional.mse_loss(v_ests.squeeze(), returns)

            # Compute avg reward for logging (approximation from returns)
            avg_reward = returns.mean()

        elif self.mode == "td0":
            # TD(0): use TD targets for value, advantages for policy
            td_target = batch.get("td_target")
            if td_target is None:
                raise ValueError("TD target not computed for TD mode")  # noqa: TRY003

            # TD advantage: r + gamma*V(s') - V(s)
            v_current = v_ests.squeeze()
            advantages = td_target - v_current.detach()
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            policy_loss = -(log_probs * normalized_advantages).mean()

            # Critic loss from TD target
            v_loss = torch.nn.functional.mse_loss(v_current, td_target.detach())

            # Compute avg reward for logging
            rewards = batch.get("rewards")
            if rewards is None:
                raise ValueError("Rewards not in batch for TD mode")  # noqa: TRY003
            avg_reward = rewards.mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")  # noqa: TRY003

        ## Entropy
        if self.rolling_action_representation == RollingActionRepresentation.CATEGORICAL:
            rolling_entropy = rolling_dist.entropy()  # (BATCH_SIZE*39,)
        else:  # BERNOULLI
            rolling_entropy = rolling_dist.entropy().sum(dim=1)  # (BATCH_SIZE*39,)
        scoring_entropy = scoring_dist.entropy()  # (BATCH_SIZE*39,)

        # Compute mean entropy for each head based on which steps use that head
        rolling_mask = phases_flat == 0
        scoring_mask = phases_flat == 1

        rolling_entropy_mean = (
            rolling_entropy[rolling_mask].mean() if rolling_mask.any() else torch.tensor(0.0)
        )
        scoring_entropy_mean = (
            scoring_entropy[scoring_mask].mean() if scoring_mask.any() else torch.tensor(0.0)
        )

        # Get current entropy coefficients for both heads
        rolling_coef, scoring_coef = self.get_entropy_coefs()

        # Compute separate entropy bonuses
        rolling_entropy_bonus = rolling_coef * rolling_entropy_mean
        scoring_entropy_bonus = scoring_coef * scoring_entropy_mean
        total_entropy_bonus = rolling_entropy_bonus + scoring_entropy_bonus

        loss = policy_loss + self.critic_coeff * v_loss - total_entropy_bonus

        # Compute explained variance for logging
        target_values = batch.get("returns") if self.mode == "reinforce" else batch.get("td_target")

        if target_values is not None:
            diff = target_values - v_ests.squeeze()
            ev = 1 - diff.var() / (target_values.var() + 1e-8)
            self.log("train/value_explained_var", ev, prog_bar=False)

        # Log separate entropy metrics for each head
        self.log("train/entropy_roll", rolling_entropy_mean, prog_bar=False)
        self.log("train/entropy_score", scoring_entropy_mean, prog_bar=False)
        self.log("train/entropy_bonus_roll", rolling_entropy_bonus, prog_bar=False)
        self.log("train/entropy_bonus_score", scoring_entropy_bonus, prog_bar=False)
        self.log("train/entropy_bonus_total", total_entropy_bonus, prog_bar=False)
        self.log("train/entropy_coef_roll", rolling_coef, prog_bar=False)
        self.log("train/entropy_coef_score", scoring_coef, prog_bar=False)
        self.log("train/policy_loss", policy_loss, prog_bar=True)
        self.log("train/v_loss", v_loss, prog_bar=False)

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=False)

        self.log("train/avg_reward", avg_reward, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)

        # Log custom counters
        self.log("counters/examples_seen", float(self.examples_seen), prog_bar=False)
        self.log("counters/env_steps_seen", float(self.env_steps_seen), prog_bar=False)

        # ===== DIAGNOSTIC METRICS =====
        # Advantage stats: adv_mean ~0 (by construction), adv_std nonzero (signal strength)
        # Red flag: adv_std → 0 = tiny gradients; roll vs score std differs >10x = phase imbalance
        adv_stats = compute_advantage_stats(advantages, phases_flat)
        self.log_dict({f"train/{k}": v for k, v in adv_stats.items()}, prog_bar=False)

        # Return std: only for REINFORCE mode
        returns = batch.get("returns")
        if self.mode == "reinforce" and returns is not None:
            ret_stats = compute_return_stats(returns)
            self.log_dict({f"train/{k}": v for k, v in ret_stats.items()}, prog_bar=False)

        # Critic EV: Good = 0.3-0.7, rising. Red flags: stuck at 0 (not learning), <0 (harmful), wild swings (unstable)
        if target_values is not None:
            critic_ev = compute_critic_explained_variance(target_values, v_ests)
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

        return loss

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Add custom gradient clipping that logs pre-clipping gradient norms."""
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

        # Check if clipping is disabled (None or 0.0)
        if gradient_clip_val is None or gradient_clip_val == 0.0:
            # No clipping - log that clipping is disabled
            self.log(
                "train/grad_clipped",
                0.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            return

        # Clipping is enabled
        clipped_flag = float(total_norm > gradient_clip_val)
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
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self):  # noqa: ANN201
        """Configure optimizers and learning rate schedulers.

        Schedule: 5% warmup from min_lr_ratio to 1.0, 70% flat, 25% decay to min_lr_ratio.
        """
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        warmup_epochs = int(0.05 * self.max_epochs)
        decay_start_epoch = int(0.75 * self.max_epochs)

        def lr_lambda(epoch: int) -> float:
            """Return LR multiplier for custom schedule with warmup, flat, and decay phases."""
            if epoch < warmup_epochs:
                # Warmup: linear from min_lr_ratio to 1.0
                progress = epoch / warmup_epochs
                return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * progress
            elif epoch < decay_start_epoch:
                # Flat: hold at 1.0
                return 1.0
            else:
                # Decay: linear from 1.0 to min_lr_ratio
                decay_epochs = self.max_epochs - decay_start_epoch
                progress = (epoch - decay_start_epoch) / decay_epochs
                return 1.0 - (1.0 - self.min_lr_ratio) * progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,  # Check every epoch
            },
        }

    def on_train_epoch_start(self) -> None:
        """Update gamma and refresh rollout buffer at the start of each epoch."""
        # Update gamma linearly from gamma_min to gamma_max over training
        if self.max_epochs > 1:
            # Linear interpolation from gamma_min to gamma_max
            progress = self.current_epoch / (self.max_epochs - 1)
            current_gamma = self.gamma_min + progress * (self.gamma_max - self.gamma_min)
        else:
            current_gamma = self.gamma_max

        self.return_calculator.gamma = current_gamma
        self.log("train/gamma", current_gamma, prog_bar=False)

        # Refresh rollout buffer for this epoch (skip on first epoch since already initialized)
        if self.current_epoch > 0 and self.collector is not None:
            from yahtzee_agent.rollout_collector import postprocess_rollout

            buffer = self.collector.collect()
            postprocess_rollout(
                buffer, self.algorithm, self.return_calculator.gamma, self.policy_net
            )

            # Update dataset buffer
            train_dataloader = self.trainer.train_dataloader
            if train_dataloader is not None:
                dataset = train_dataloader.dataset
                if hasattr(dataset, "buffer"):
                    dataset.buffer = buffer

            # Clear KL divergence caches since buffer refreshed
            if hasattr(self, "_prev_roll_p"):
                delattr(self, "_prev_roll_p")
            if hasattr(self, "_prev_score_p"):
                delattr(self, "_prev_score_p")

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
        """Log epoch-level summary statistics."""
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
        # Run comprehensive test evaluation at the end of training
        self.run_sweep_test_evaluation(run_deterministic=True, run_stochastic=False)

        for env in self.validation_envs:
            env.close()
        self.validation_envs.clear()
