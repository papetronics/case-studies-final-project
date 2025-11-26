import json
import os
from enum import Enum
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytorch_lightning as lightning
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer

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
    compute_training_health_score,
)
from utilities.scoring_helper import BONUS_POINTS, MINIMUM_UPPER_SCORE_FOR_BONUS, YAHTZEE_SCORE
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

# Load DP baseline once at module level
_DP_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dp_ideal_baseline.json"
)
with open(_DP_BASELINE_PATH) as f:
    _DP_BASELINE = torch.tensor(json.load(f), dtype=torch.float32)[:, 0]  # Extract means only


class Algorithm(Enum):
    """Training algorithm."""

    REINFORCE = "reinforce"  # Monte Carlo policy gradient (REINFORCE)
    A2C = "a2c"  # Advantage Actor-Critic (A2C) with TD(0)
    PPO = "ppo"  # Proximal Policy Optimization (PPO) with clipped surrogate objective


class YahtzeeAgentTrainer(lightning.LightningModule):
    """PyTorch Lightning trainer for Yahtzee agents using policy gradient methods (REINFORCE/A2C)."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_size: int,
        learning_rate: float,
        num_hidden: int,
        dropout_rate: float,
        activation_function: ActivationFunctionName,
        epochs: int,
        min_lr_ratio: float,
        gamma_start: float,
        gamma_end: float,
        gamma_anneal_period: float,
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
        algorithm: Algorithm,
        upper_score_regression_loss_weight: float,
        upper_score_shaping_weight: float,
        gae_lambda: float,
        clip_epsilon: float,
        ppo_games_per_minibatch: int,
        ppo_epochs: int,
        gradient_clip_val: float,
        use_layer_norm: bool = True,
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
            use_layer_norm=use_layer_norm,
        )

        self.algorithm = algorithm
        self.learning_rate: float = learning_rate
        self.max_epochs: int = epochs
        self.min_lr_ratio: float = min_lr_ratio
        self.gamma_start: float = gamma_start
        self.gamma_end: float = gamma_end
        self.gamma_anneal_period: float = gamma_anneal_period
        self.entropy_coeff_rolling_max: float = entropy_coeff_rolling_max
        self.entropy_coeff_rolling_min: float = entropy_coeff_rolling_min
        self.entropy_coeff_scoring_max: float = entropy_coeff_scoring_max
        self.entropy_coeff_scoring_min: float = entropy_coeff_scoring_min
        self.entropy_hold_period: float = entropy_hold_period
        self.entropy_anneal_period: float = entropy_anneal_period
        self.critic_coeff: float = critic_coeff
        self.num_steps_per_episode: int = num_steps_per_episode
        self.upper_score_regression_loss_weight: float = upper_score_regression_loss_weight
        self.upper_score_shaping_weight: float = upper_score_shaping_weight
        self.gae_lambda: float = gae_lambda
        self.clip_epsilon: float = clip_epsilon
        self._gradient_clip_val: float = gradient_clip_val  # manual clipping for PPO

        self.validation_envs: list[gym.Env[Observation, Action]] = []  # Created on demand

        # Always set PPO-specific attributes, regardless of algorithm
        self.ppo_games_per_minibatch: int = ppo_games_per_minibatch
        self.ppo_epochs: int = ppo_epochs
        # for PPO style minibatching we need to turn off Lightning's automatic updating
        if self.algorithm == Algorithm.PPO:
            self.automatic_optimization = False

    def run_batched_validation_games(  # noqa: C901, PLR0912, PLR0915
        self, num_games: int, run_deterministic: bool = True, run_stochastic: bool = False
    ) -> tuple[list[float], list[float], dict[str, Any], dict[str, Any]]:
        """Run multiple Yahtzee games in parallel with both deterministic and stochastic action selection."""
        # Calculate number of environments needed based on which modes are enabled
        num_det = num_games if run_deterministic else 0
        num_stoch = num_games if run_stochastic else 0
        total_envs_needed = num_det + num_stoch

        # Early return if neither mode is enabled
        if total_envs_needed == 0:
            return [], [], {}, {}

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

        # Track scorecard metrics for each game - shape (total_envs_needed, 13)
        scorecards = torch.zeros((total_envs_needed, 13), dtype=torch.float32)

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
                rolling_probs, scoring_probs, v_ests, _ = self.policy_net.forward(state_tensors)

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
                        # Game ended - save final scorecard
                        scorecards[game_idx] = torch.from_numpy(next_obs["score_sheet"]).float()
                        newly_inactive.append(game_idx)
                    else:
                        observations[game_idx] = next_obs

                # Remove completed games from active list
                for game_idx in newly_inactive:
                    active_indices.remove(game_idx)

        # Split results into deterministic and stochastic
        deterministic_scores = total_scores[:num_det]
        stochastic_scores = total_scores[num_det:]

        # Compute metrics from scorecards
        def calculate_scorecard_statistics(scorecards_tensor: torch.Tensor) -> dict[str, Any]:
            """Compute scorecard metrics from a tensor of scorecards."""
            if scorecards_tensor.shape[0] == 0:
                return {}

            # A) % of DP baseline for each category
            category_means = scorecards_tensor.mean(dim=0)
            # First 13 entries of _DP_BASELINE are the category means
            category_pct_of_dp = (category_means / _DP_BASELINE[:13]) * 100

            # B) % of games with a Yahtzee (category 11, score = 50)
            has_yahtzee = (scorecards_tensor[:, 11] == YAHTZEE_SCORE).float()
            pct_yahtzee = float(has_yahtzee.mean() * 100)

            # C) % of games with bonus (upper section sum >= 63)
            upper_section_sum = scorecards_tensor[:, 0:6].sum(dim=1)
            has_bonus_mask = (upper_section_sum >= MINIMUM_UPPER_SCORE_FOR_BONUS).float()
            pct_bonus = float(has_bonus_mask.mean() * 100)

            metrics = {
                "category_pct_of_dp": category_pct_of_dp,
                "pct_yahtzee": pct_yahtzee,
                "pct_bonus": pct_bonus,
            }
            return metrics

        det_metrics = calculate_scorecard_statistics(scorecards[:num_det])
        stoch_metrics = calculate_scorecard_statistics(scorecards[num_det:])

        return deterministic_scores, stochastic_scores, det_metrics, stoch_metrics

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, float]:  # noqa: ARG002
        """Run validation using batched parallel games."""
        num_validation_games = 1000

        # Run all games in parallel with batched forward passes (only deterministic by default)
        det_scores, stoch_scores, det_metrics, stoch_metrics = self.run_batched_validation_games(
            num_validation_games, run_deterministic=True, run_stochastic=False
        )

        category_labels = [
            "ones",
            "twos",
            "threes",
            "fours",
            "fives",
            "sixes",
            "3ok",
            "4ok",
            "fh",
            "ss",
            "ls",
            "yahtzee",
            "chance",
        ]

        # Log deterministic results if available
        if det_scores:
            det_mean = float(np.mean(det_scores))
            det_std = float(np.std(det_scores))
            self.log("val/mean_total_score_det", det_mean, prog_bar=True)
            self.log("val/std_total_score_det", det_std, prog_bar=False)

            if det_metrics:
                for idx, label in enumerate(category_labels):
                    self.log(
                        f"val/scorecard/{label}_pct_of_dp",
                        float(det_metrics["category_pct_of_dp"][idx]),
                        prog_bar=False,
                    )

                self.log("val/pct_yahtzee", det_metrics["pct_yahtzee"], prog_bar=False)
                self.log("val/pct_bonus", det_metrics["pct_bonus"], prog_bar=True)
        else:
            det_mean = 0.0

        # Log stochastic results if available
        if stoch_scores:
            stoch_mean = float(np.mean(stoch_scores))
            stoch_std = float(np.std(stoch_scores))
            self.log("val/mean_total_score_stoch", stoch_mean, prog_bar=False)
            self.log("val/std_total_score_stoch", stoch_std, prog_bar=False)

            if stoch_metrics:
                for idx, label in enumerate(category_labels):
                    self.log(
                        f"val/scorecard_stoch/{label}_pct_of_dp",
                        float(stoch_metrics["category_pct_of_dp"][idx]),
                        prog_bar=False,
                    )
                self.log("val/pct_yahtzee_stoch", stoch_metrics["pct_yahtzee"], prog_bar=False)
                self.log("val/pct_bonus_stoch", stoch_metrics["pct_bonus"], prog_bar=False)

        return {"val_loss": -det_mean}  # Negative because higher scores are better

    def get_gamma(self) -> float:
        """Get current gamma (discount factor) value.

        Anneals from gamma_start to gamma_end over first gamma_anneal_period of training,
        then holds at gamma_end for remaining training.
        """
        end_gamma_anneal_epoch = self.max_epochs * self.gamma_anneal_period

        if self.current_epoch < end_gamma_anneal_epoch:
            progress = self.current_epoch / end_gamma_anneal_epoch
            return self.gamma_start - progress * (self.gamma_start - self.gamma_end)
        else:
            # Second half: hold at min
            return self.gamma_end

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

    def training_step(self, batch: EpisodeBatch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Perform a training step using REINFORCE, A2C, or PPO algorithm with vectorized operations."""
        # batch is an EpisodeBatch dict with pre-flattened tensors:
        # - "states": (BATCH_SIZE*39, state_size) float32
        # - "rolling_actions": (BATCH_SIZE*39, 5) int
        # - "scoring_actions": (BATCH_SIZE*39,) int
        # - "rewards": (BATCH_SIZE*39,) float32
        # - "next_states": (BATCH_SIZE*39, state_size) float32
        # - "phases": (BATCH_SIZE*39,) int
        # - "v_baseline": (BATCH_SIZE*39,) float32 - for advantage calculation
        # - "next_v_baseline": (BATCH_SIZE*39,) float32 - for TD(0)
        # - "current_upper_potential": (BATCH_SIZE*39,) float32 - Φ(s_t) predictions
        # - "next_upper_potential": (BATCH_SIZE*39,) float32 - Φ(s_{t+1}) predictions
        # - "upper_score_actual": (BATCH_SIZE*39,) float32 - actual final upper scores
        # - "old_log_probs": (BATCH_SIZE*39,) float32 - for PPO

        # Dataset pre-flattens, so we just extract directly
        states_flat = batch["states"]
        rolling_actions_flat = batch["rolling_actions"]
        scoring_actions_flat = batch["scoring_actions"]
        rewards_flat = batch["rewards"]
        v_baseline = batch["v_baseline"]
        next_v_baseline = batch["next_v_baseline"]
        phases_flat = batch["phases"]
        upper_score_actual = batch["upper_score_actual"]
        current_upper_potential = batch["current_upper_potential"]
        next_upper_potential = batch["next_upper_potential"]
        old_log_probs = batch["old_log_probs"]

        # Calculate num_episodes and steps_per_episode from flattened shape
        # single_turn: 338 episodes x 3 steps = 1014 total
        # full_game: 26 episodes x 39 steps = 1014 total
        total_steps = states_flat.shape[0]
        steps_per_episode = self.num_steps_per_episode
        num_episodes = total_steps // steps_per_episode

        loss: torch.Tensor

        if self.algorithm == Algorithm.PPO:
            loss = self.run_ppo_minibatching(
                states_flat,
                rolling_actions_flat,
                scoring_actions_flat,
                rewards_flat,
                v_baseline,
                next_v_baseline,
                phases_flat,
                current_upper_potential,
                next_upper_potential,
                old_log_probs,
                total_steps,
                steps_per_episode,
                num_episodes,
            )

            # Get upper score predictions for regression loss (if needed)
            # PPO computes these inside minibatches, but we need them here for the full batch
            if self.upper_score_regression_loss_weight != 0:
                with torch.no_grad():
                    _, _, _, upper_score_logit = self.policy_net.forward(states_flat)
            else:
                upper_score_logit = None
        else:
            # Forward pass through current policy to get probabilities and value estimates
            rolling_probs, scoring_probs, v_ests, upper_score_logit = self.policy_net.forward(
                states_flat
            )

            normalized_advantage, returns = self.get_advantage(
                num_episodes,
                steps_per_episode,
                rewards_flat,
                next_v_baseline,
                v_ests,
                phases_flat,
                current_upper_potential,
                next_upper_potential,
            )

            policy_loss, entropy_loss = self.get_policy_loss(
                rolling_probs,
                rolling_actions_flat,
                scoring_probs,
                scoring_actions_flat,
                phases_flat,
                normalized_advantage,
                old_log_probs,
            )

            loss = (
                policy_loss
                + self.critic_coeff * self.get_value_loss(v_ests, returns)
                + entropy_loss
            )

        ## Standard regression loss for upper score prediction
        # Both A2C and PPO can train the upper score head if the weight is non-zero
        if (
            self.algorithm != Algorithm.REINFORCE
            and self.upper_score_regression_loss_weight != 0
            and upper_score_logit is not None
        ):
            upper_score_loss = torch.nn.functional.mse_loss(
                upper_score_logit.squeeze(), upper_score_actual.float()
            )
            loss += self.upper_score_regression_loss_weight * upper_score_loss
            self.log(
                "train/upper_score_loss",
                self.upper_score_regression_loss_weight * upper_score_loss,
                prog_bar=False,
            )

            self.log("train/upper_score_loss_raw", upper_score_loss, prog_bar=False)
            self.log(
                "train/pred_upper",
                upper_score_logit.mean().item() * MINIMUM_UPPER_SCORE_FOR_BONUS
                + MINIMUM_UPPER_SCORE_FOR_BONUS,
                prog_bar=False,
            )
            self.log(
                "train/target_upper",
                upper_score_actual.mean().item() * MINIMUM_UPPER_SCORE_FOR_BONUS
                + MINIMUM_UPPER_SCORE_FOR_BONUS,
                prog_bar=False,
            )

        self.log("train/total_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=False)
        self.log("train/frac_roll_steps", compute_phase_balance(phases_flat), prog_bar=False)

        if self.algorithm != Algorithm.PPO:
            self.log_kl_diagnostics(scoring_probs, rolling_probs, phases_flat)

        return loss

    def run_ppo_minibatching(  # noqa: PLR0913
        self,
        states_flat: torch.Tensor,
        rolling_actions_flat: torch.Tensor,
        scoring_actions_flat: torch.Tensor,
        rewards_flat: torch.Tensor,
        v_baseline: torch.Tensor,
        next_v_baseline: torch.Tensor,
        phases_flat: torch.Tensor,
        current_upper_potential: torch.Tensor,
        next_upper_potential: torch.Tensor,
        old_log_probs: torch.Tensor,
        total_steps: int,
        steps_per_episode: int,
        num_episodes: int,
    ) -> torch.Tensor:
        """Run PPO training with minibatching and multiple epochs per batch."""
        optimizer = cast("LightningOptimizer", self.optimizers())

        # 1) Compute advantages & returns for the whole batch, respecting episode boundaries
        normalized_advantages, returns_full = self.get_advantage(
            num_episodes,
            steps_per_episode,
            rewards_flat,
            next_v_baseline,
            v_baseline,
            phases_flat,
            current_upper_potential,
            next_upper_potential,
        )

        batch_size = self.ppo_games_per_minibatch * steps_per_episode
        assert total_steps % batch_size == 0
        num_minibatches = total_steps // batch_size

        ppo_epochs = self.ppo_epochs  # e.g. 3

        total_loss_sum = torch.tensor(0.0, device=states_flat.device)

        # Track KL divergence for PPO diagnostics (comparing old vs new policy)
        kl_sum = torch.tensor(0.0, device=states_flat.device)

        for _ in range(ppo_epochs):
            # 2) Shuffle indices each epoch
            perm = torch.randperm(total_steps, device=states_flat.device)

            for minibatch_idx in range(num_minibatches):
                start_idx = minibatch_idx * batch_size
                end_idx = start_idx + batch_size
                idx = perm[start_idx:end_idx]

                mb_states = states_flat[idx]
                mb_rolling_actions = rolling_actions_flat[idx]
                mb_scoring_actions = scoring_actions_flat[idx]
                mb_phases = phases_flat[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = normalized_advantages[idx]
                mb_returns = returns_full[idx]

                # Forward pass on minibatch
                # PPO does not use upper_score_logit (upper score regression loss), unlike A2C.
                mb_rolling_logits, mb_scoring_probs, mb_v_ests, _ = self.policy_net.forward(
                    mb_states
                )

                mb_policy_loss, mb_entropy_loss = self.get_policy_loss(
                    mb_rolling_logits,
                    mb_rolling_actions,
                    mb_scoring_probs,
                    mb_scoring_actions,
                    mb_phases,
                    mb_advantages,
                    mb_old_log_probs,
                )

                # Compute KL divergence between old and new policy for PPO diagnostics
                with torch.no_grad():
                    # Get new log probs
                    rolling_dist: torch.distributions.Distribution
                    if (
                        self.rolling_action_representation
                        == RollingActionRepresentation.CATEGORICAL
                    ):
                        rolling_dist = torch.distributions.Categorical(logits=mb_rolling_logits)
                        new_rolling_log_probs = rolling_dist.log_prob(mb_rolling_actions)
                    else:  # BERNOULLI
                        rolling_dist = torch.distributions.Bernoulli(logits=mb_rolling_logits)
                        new_rolling_log_probs = rolling_dist.log_prob(
                            mb_rolling_actions.float()
                        ).sum(dim=1)

                    scoring_dist = torch.distributions.Categorical(mb_scoring_probs)
                    new_scoring_log_probs = scoring_dist.log_prob(mb_scoring_actions)

                    new_log_probs = torch.where(
                        mb_phases == 0, new_rolling_log_probs, new_scoring_log_probs
                    )

                    # KL(old || new) ≈ old_log_prob - new_log_prob for same action
                    mb_kl = (mb_old_log_probs - new_log_probs).mean()
                    kl_sum += mb_kl

                value_loss = self.get_value_loss(mb_v_ests, mb_returns)

                mb_loss: torch.Tensor = (
                    mb_policy_loss + self.critic_coeff * value_loss + mb_entropy_loss
                )

                # unwrap to real torch optimizer
                assert isinstance(optimizer, LightningOptimizer)
                t_optimizer: torch.optim.Optimizer = optimizer.optimizer

                optimizer.zero_grad()
                self.manual_backward(mb_loss)

                # gradient clipping if you want it
                self.configure_gradient_clipping(
                    t_optimizer,
                    gradient_clip_val=self._gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )

                optimizer.step()

                total_loss_sum += mb_loss.detach()

        loss = total_loss_sum / (ppo_epochs * num_minibatches)

        # Log average KL divergence across all updates
        avg_kl = kl_sum / (ppo_epochs * num_minibatches)
        self.log("train/ppo_kl_divergence", avg_kl, prog_bar=False)

        return loss

    def get_advantage(  # noqa: PLR0913
        self,
        num_episodes: int,
        steps_per_episode: int,
        rewards_flat: torch.Tensor,  # [B], B = num_episodes * steps_per_episode
        next_v_baseline: torch.Tensor,  # [B], V(s_{t+1})
        v_ests: torch.Tensor,  # [B] or [B, 1], V(s_t)
        phases_flat: torch.Tensor,
        current_upper_potential: torch.Tensor,
        next_upper_potential: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate advantages using REINFORCE, A2C TD(0), or A2C+GAE, and log diagnostics."""
        gamma = self.get_gamma()
        # Make sure value estimates are a flat 1D tensor
        v_flat = v_ests.detach().view(-1)
        next_v_flat = next_v_baseline.detach().view(-1)

        B = num_episodes * steps_per_episode  # noqa: N806
        assert rewards_flat.numel() == B
        assert v_flat.numel() == B
        assert next_v_flat.numel() == B

        # Calculate returns without gradients, since they are just targets for the critic / multipliers for the actor loss
        with torch.no_grad():
            returns = torch.zeros_like(rewards_flat)
            advantages = torch.zeros_like(rewards_flat)

            if self.algorithm == Algorithm.REINFORCE:
                # Monte Carlo: backward pass per episode
                for episode_idx in range(num_episodes):
                    g = torch.tensor(0.0, device=rewards_flat.device)
                    base = episode_idx * steps_per_episode
                    for t in reversed(range(steps_per_episode)):
                        idx = base + t
                        g = rewards_flat[idx] + gamma * g
                        returns[idx] = g

                # Advantages = returns - V(s)
                advantages = returns - v_flat
            else:  # A2C or PPO
                # Apply reward shaping only if upper_score_shaping_weight is non-zero
                if self.upper_score_shaping_weight != 0:
                    shaped_rewards = self.shaped_reward(
                        rewards_flat, current_upper_potential, next_upper_potential
                    )
                else:
                    shaped_rewards = rewards_flat

                # A2C / Actor-Critic branch: TD(0) or GAE
                use_gae = self.gae_lambda > 0.0
                if use_gae:
                    lam = self.gae_lambda
                    # GAE per episode
                    for episode_idx in range(num_episodes):
                        base = episode_idx * steps_per_episode
                        gae = torch.tensor(0.0, device=shaped_rewards.device)
                        for t in reversed(range(steps_per_episode)):
                            idx = base + t
                            # δ_t = r_t + γ V(s_{t+1}) - V(s_t)  # noqa: RUF003
                            delta = shaped_rewards[idx] + gamma * next_v_flat[idx] - v_flat[idx]
                            gae = delta + gamma * lam * gae
                            advantages[idx] = gae
                            # Return target: V_target = A + V(s_t)
                            returns[idx] = gae + v_flat[idx]
                else:
                    # Plain A2C TD(0): r_t + γ V(s_{t+1})  # noqa: RUF003
                    returns = shaped_rewards + gamma * next_v_flat
                    advantages = returns - v_flat

        # -------------------------------------------------------
        # Normalize advantages for the policy loss
        # -------------------------------------------------------
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -------------------------------------------------------
        # Diagnostics
        # -------------------------------------------------------
        episode_rewards = rewards_flat.view(num_episodes, steps_per_episode).sum(dim=1)
        avg_reward = episode_rewards.mean()
        self.log("train/avg_reward", avg_reward, prog_bar=True)

        ret_stats = compute_return_stats(episode_rewards)
        self.log_dict({f"train/{k}": v for k, v in ret_stats.items()}, prog_bar=False)

        adv_stats = compute_advantage_stats(advantages, phases_flat)
        self.log_dict({f"train/{k}": v for k, v in adv_stats.items()}, prog_bar=False)

        return normalized_advantages, returns

    def shaped_reward(
        self,
        rewards_flat: torch.Tensor,
        current_upper_potential: torch.Tensor,
        next_upper_potential: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate shaped rewards using potential-based shaping.

        Converts normalized potential back to raw score, clamps to [0, 63],
        and computes potential difference scaled by BONUS_POINTS.
        """
        # Convert normalized potential back to raw score: potential * 63 + 63
        # Then clamp to [0, 63]
        next_score = torch.clamp(
            next_upper_potential.squeeze() * MINIMUM_UPPER_SCORE_FOR_BONUS
            + MINIMUM_UPPER_SCORE_FOR_BONUS,
            min=0.0,
            max=float(MINIMUM_UPPER_SCORE_FOR_BONUS),
        )
        current_score = torch.clamp(
            current_upper_potential.squeeze() * MINIMUM_UPPER_SCORE_FOR_BONUS
            + MINIMUM_UPPER_SCORE_FOR_BONUS,
            min=0.0,
            max=float(MINIMUM_UPPER_SCORE_FOR_BONUS),
        )

        # Compute potential: (score / 63) * BONUS_POINTS
        next_potential = (next_score / MINIMUM_UPPER_SCORE_FOR_BONUS) * BONUS_POINTS
        current_potential = (current_score / MINIMUM_UPPER_SCORE_FOR_BONUS) * BONUS_POINTS

        # Calculate potential difference using Ng's formula
        potential_diff = next_potential - current_potential
        shaping_bonus = self.upper_score_shaping_weight * potential_diff

        # Shaped reward = r + weight * (Phi(s') - Phi(s))
        shaped_rewards: torch.Tensor = rewards_flat + shaping_bonus

        ## =========================================================================================
        ## Diagnostics
        self.log("train/shaping_weight", self.upper_score_shaping_weight, prog_bar=False)
        self.log("train/current_potential_mean", current_potential.mean(), prog_bar=False)
        self.log("train/next_potential_mean", next_potential.mean(), prog_bar=False)
        self.log("train/potential_diff_mean", potential_diff.mean(), prog_bar=False)
        self.log("train/potential_diff_abs_mean", potential_diff.abs().mean(), prog_bar=False)
        self.log("train/shaping_bonus_mean", shaping_bonus.mean(), prog_bar=True)
        self.log("train/shaping_bonus_abs_mean", shaping_bonus.abs().mean(), prog_bar=False)
        self.log("train/raw_reward_mean", rewards_flat.mean(), prog_bar=False)
        self.log("train/shaped_reward_mean", shaped_rewards.mean(), prog_bar=False)
        ## =========================================================================================

        return shaped_rewards

    def get_policy_loss(  # noqa: PLR0913
        self,
        rolling_logits: torch.Tensor,
        rolling_actions_flat: torch.Tensor,
        scoring_probs: torch.Tensor,
        scoring_actions_flat: torch.Tensor,
        phases_flat: torch.Tensor,
        normalized_advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate policy loss using log probabilities and advantages."""
        rolling_dist: torch.distributions.Distribution

        rolling_probs = (
            torch.nn.functional.sigmoid(rolling_logits)
            if self.rolling_action_representation == RollingActionRepresentation.BERNOULLI
            else torch.nn.functional.softmax(rolling_logits, dim=-1)
        )

        if self.rolling_action_representation == RollingActionRepresentation.CATEGORICAL:
            rolling_dist = torch.distributions.Categorical(probs=rolling_probs)
            rolling_log_probs = rolling_dist.log_prob(
                rolling_actions_flat.float()
            )  # (BATCH_SIZE * 39,)
        else:  # BERNOULLI
            rolling_dist = torch.distributions.Bernoulli(probs=rolling_probs)
            rolling_log_probs = rolling_dist.log_prob(rolling_actions_flat.float()).sum(
                dim=1
            )  # (BATCH_SIZE * 39,)

        scoring_dist = torch.distributions.Categorical(scoring_probs)
        scoring_log_probs = scoring_dist.log_prob(scoring_actions_flat)  # (BATCH_SIZE * 39,)

        # Select the appropriate log prob based on phase
        log_probs = torch.where(
            phases_flat == 0, rolling_log_probs, scoring_log_probs
        )  # (BATCH_SIZE * 39,)

        # Calculate policy loss based on algorithm
        if self.algorithm == Algorithm.PPO:
            # PPO: clipped surrogate objective
            # ratio = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped objective: L^CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            policy_loss_unclipped = ratio * normalized_advantages
            policy_loss_clipped = clipped_ratio * normalized_advantages
            policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()

            # Log PPO-specific metrics
            with torch.no_grad():
                # Fraction of steps where clipping was active
                clip_fraction = (
                    ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon))
                    .float()
                    .mean()
                )
                self.log("train/ppo_clip_fraction", clip_fraction, prog_bar=False)
                # Average ratio (should stay near 1.0)
                self.log("train/ppo_ratio_mean", ratio.mean(), prog_bar=False)
                self.log("train/ppo_ratio_std", ratio.std(), prog_bar=False)
        else:
            # REINFORCE or A2C: standard policy gradient
            policy_loss = -(log_probs * normalized_advantages).mean()

        ## =========================================================================================
        ## Diagnostics
        entropy_stats = compute_entropy_stats(
            rolling_probs,
            scoring_probs,
            phases_flat,
            self.rolling_action_representation == RollingActionRepresentation.BERNOULLI,
        )
        if hasattr(self, "_epoch_metrics"):
            self._epoch_metrics["entropy_roll"].append(float(entropy_stats["entropy_roll"].item()))
            self._epoch_metrics["entropy_score"].append(
                float(entropy_stats["entropy_score"].item())
            )
        concentration_stats = compute_action_concentration(
            scoring_probs,
            rolling_probs,
            phases_flat,
            self.rolling_action_representation == RollingActionRepresentation.BERNOULLI,
        )
        mask_diversity = compute_rolling_mask_diversity(rolling_actions_flat, phases_flat)

        self.log("train/policy_loss", policy_loss, prog_bar=True)
        self.log_dict({f"train/{k}": v for k, v in entropy_stats.items()}, prog_bar=False)
        self.log_dict({f"train/{k}": v for k, v in concentration_stats.items()}, prog_bar=False)
        if mask_diversity is not None:
            self.log("train/roll_mask_diversity", mask_diversity, prog_bar=False)
        ## =========================================================================================

        return policy_loss, self.get_entropy_loss(phases_flat, rolling_dist, scoring_dist)

    def get_value_loss(self, v_ests: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Calculate value (critic) loss and log explained variance."""
        # Calculate Huber loss
        # v_loss = torch.nn.functional.smooth_l1_loss(v_ests.squeeze(), returns)

        # Use MSE loss for critic
        v_loss = torch.nn.functional.mse_loss(v_ests.squeeze(), returns)

        ## =========================================================================================
        ## Diagnostics
        self.log("train/v_loss", v_loss, prog_bar=False)

        critic_ev = compute_critic_explained_variance(returns, v_ests)
        self.log("train/critic_ev", critic_ev, prog_bar=False)
        if hasattr(self, "_epoch_metrics"):
            self._epoch_metrics["critic_ev"].append(float(critic_ev.item()))
        ## =========================================================================================

        return v_loss

    def get_entropy_loss(
        self,
        phases_flat: torch.Tensor,
        rolling_dist: torch.distributions.Distribution,
        scoring_dist: torch.distributions.Distribution,
    ) -> torch.Tensor:
        """Calculate and log the entropy bonus."""
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

        rolling_coef, scoring_coef = self.get_entropy_coefs()

        rolling_entropy_bonus = rolling_coef * rolling_entropy_mean
        scoring_entropy_bonus = scoring_coef * scoring_entropy_mean
        total_entropy_bonus = rolling_entropy_bonus + scoring_entropy_bonus

        ## =========================================================================================
        ## Diagnostics
        self.log("train/entropy_roll", rolling_entropy_mean, prog_bar=False)
        self.log("train/entropy_score", scoring_entropy_mean, prog_bar=False)
        self.log("train/entropy_bonus_roll", rolling_entropy_bonus, prog_bar=False)
        self.log("train/entropy_bonus_score", scoring_entropy_bonus, prog_bar=False)
        self.log("train/entropy_bonus_total", total_entropy_bonus, prog_bar=False)
        self.log("train/entropy_coef_roll", rolling_coef, prog_bar=False)
        self.log("train/entropy_coef_score", scoring_coef, prog_bar=False)
        ## =========================================================================================

        return -total_entropy_bonus

    def log_kl_diagnostics(
        self, scoring_probs: torch.Tensor, rolling_probs: torch.Tensor, phases_flat: torch.Tensor
    ) -> None:
        """Log various diagnostic metrics."""
        ## =========================================================================================
        ## Diagnostics
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
        ## =========================================================================================

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
        """Initialize epoch-level trackers and log current gamma."""
        # Log current gamma value
        current_gamma = self.get_gamma()
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

            health_score = compute_training_health_score(
                critic_ev_mean, entropy_roll_mean, entropy_score_mean
            )
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
            logger.define_metric("train/roll_top1", summary="mean,max")
            logger.define_metric("train/roll_top3", summary="mean,max")

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
