from typing import TYPE_CHECKING, TypedDict, cast

import gymnasium as gym
import torch

from environments.full_yahtzee_env import (
    MINIMUM_UPPER_SCORE_FOR_BONUS,
    Action,
    Observation,
    YahtzeeEnv,
)
from yahtzee_agent.features import PhiFeature

from .model import convert_rolling_action_to_hold_mask, get_input_dimensions, phi, sample_action

if TYPE_CHECKING:
    from .model import YahtzeeAgent


class EpisodeBatch(TypedDict):
    """Typed dict for episode batch data returned by the dataset.

    Tensors are pre-flattened: (BATCH_SIZE * num_steps, ...) where:
    - BATCH_SIZE = number of parallel episodes collected
    - num_steps = 3 for single-turn episodes, 39 for full-game episodes

    The dataset collects in parallel with batched forward passes and flattens
    before returning to avoid reshape operations in the trainer.
    """

    states: torch.Tensor  # (BATCH_SIZE*num_steps, state_size) float32
    rolling_actions: (
        torch.Tensor
    )  # Bernoulli: (BATCH_SIZE*num_steps, 5) int, Categorical: (BATCH_SIZE*num_steps,) int
    scoring_actions: torch.Tensor  # (BATCH_SIZE*num_steps,) int (category index 0-12)
    rewards: torch.Tensor  # (BATCH_SIZE*num_steps,) float32
    next_states: torch.Tensor  # (BATCH_SIZE*num_steps, state_size) float32
    phases: torch.Tensor  # (BATCH_SIZE*num_steps,) int (0=rolling, 1=scoring)
    v_baseline: torch.Tensor  # (BATCH_SIZE*num_steps,) float32 - value estimates for current states
    next_v_baseline: (
        torch.Tensor
    )  # (BATCH_SIZE*num_steps,) float32 - value estimates for next states
    received_bonus: (
        torch.Tensor
    )  # (BATCH_SIZE*num_steps,) int (0 or 1) - whether episode received upper section bonus


class SelfPlayDataset(torch.utils.data.Dataset[EpisodeBatch]):
    """
    A dataset that collects episodes by playing against itself using the current policy.

    Each __getitem__ call collects a full batch of episodes in parallel with batched
    forward passes for GPU efficiency.

    Modes:
      - single_turn: num_steps_per_episode = 3   (roll, roll, score)
      - full_game:   num_steps_per_episode = 39  (13 turns * 3 steps)

    Returned batch (flattened over [batch_size, num_steps_per_episode]):
      - "states":          (B*num_steps, state_size)
      - "rolling_actions": Bernoulli: (B*num_steps, 5), Categorical: (B*num_steps,)
      - "scoring_actions": (B*num_steps,)
      - "rewards":         (B*num_steps,)
      - "next_states":     (B*num_steps, state_size)
      - "phases":          (B*num_steps,)
      - "v_baseline":      (B*num_steps,)        = V(s_t)
      - "next_v_baseline": (B*num_steps,)        = V(s_{t+1}) for t < T-1, 0 at last step
    """

    def __init__(
        self,
        policy_net: "YahtzeeAgent",
        size: int,
        batch_size: int,
        num_steps_per_episode: int,
        stagger_environments: bool = False,
    ) -> None:
        self.policy_net: YahtzeeAgent = policy_net
        self.size = size
        self.batch_size = batch_size
        self.num_steps_per_episode = num_steps_per_episode
        self.stagger_environments = stagger_environments

        # Create pool of environments for parallel collection
        self.envs: list[gym.Env[Observation, Action]] = []
        for env_idx in range(batch_size):
            env = gym.make("FullYahtzee-v1")
            env.reset()

            if self.stagger_environments:
                # Advance environment to turn (env_idx % 13) using random actions
                target_turn = env_idx % 13
                turn_count = 0

                while turn_count < target_turn:
                    unwrapped: YahtzeeEnv = cast("YahtzeeEnv", env.unwrapped)
                    obs = unwrapped.observe()

                    if obs["phase"] == 0:  # Rolling phase
                        sampled = env.action_space.sample()
                        action: Action = {"hold_mask": sampled["hold_mask"].astype(bool)}
                    else:  # Scoring phase
                        available_categories = [
                            i for i in range(13) if obs["score_sheet_available_mask"][i] == 1
                        ]
                        sampled = env.action_space.sample()
                        score_category = int(sampled["score_category"])
                        if score_category not in available_categories:
                            score_category = available_categories[0] if available_categories else 0
                        action = {"score_category": score_category}
                        turn_count += 1

                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        env.reset()
                        break

            self.envs.append(env)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.size

    def __getitem__(self, idx: int) -> EpisodeBatch:  # noqa: PLR0912, PLR0915
        """Collect a batch of self-play episodes using the current policy network."""
        # idx is unused: each call collects a fresh batch of self-play trajectories

        # Get current observations from all environments
        observations: list[Observation] = []
        for env in self.envs:
            unwrapped: YahtzeeEnv = cast("YahtzeeEnv", env.unwrapped)
            observations.append(unwrapped.observe())

        num_steps = self.num_steps_per_episode
        device = next(self.policy_net.parameters()).device
        features = cast("list[PhiFeature]", self.policy_net.features)
        state_size = get_input_dimensions(features)

        # Pre-allocate tensors: (B, T, ...)
        states = torch.zeros(
            self.batch_size, num_steps, state_size, dtype=torch.float32, device=device
        )

        rolling_action_representation = self.policy_net.rolling_action_representation
        if rolling_action_representation.value == "bernoulli":
            rolling_actions = torch.zeros(
                self.batch_size, num_steps, 5, dtype=torch.long, device=device
            )
        else:  # CATEGORICAL
            rolling_actions = torch.zeros(
                self.batch_size, num_steps, dtype=torch.long, device=device
            )

        scoring_actions = torch.zeros(self.batch_size, num_steps, dtype=torch.long, device=device)
        rewards = torch.zeros(self.batch_size, num_steps, dtype=torch.float32, device=device)
        next_states = torch.zeros(
            self.batch_size, num_steps, state_size, dtype=torch.float32, device=device
        )
        phases = torch.zeros(self.batch_size, num_steps, dtype=torch.long, device=device)
        v_baseline = torch.zeros(self.batch_size, num_steps, dtype=torch.float32, device=device)

        # Track whether each episode received the upper section bonus
        episode_received_bonus = torch.zeros(self.batch_size, dtype=torch.float32, device=device)

        # ---- Rollout collection ----
        with torch.no_grad():
            for step_idx in range(num_steps):
                # Batch encode all current observations
                state_tensors = torch.stack(
                    [phi(obs, features, device) for obs in observations]
                )  # (B, state_size)

                states[:, step_idx, :] = state_tensors

                # Single batched forward pass for all envs: π_roll, π_score, V(s_t)
                # ignore the bonus prediction, since we run a forward pass in training with grad
                rolling_probs, scoring_probs, v_ests, _ = self.policy_net.forward(state_tensors)
                v_baseline[:, step_idx] = v_ests.squeeze(-1)

                # Sample actions
                actions, _, _ = sample_action(
                    rolling_probs,
                    scoring_probs,
                    v_ests,
                    rolling_action_representation,
                )
                rolling_action_tensor, scoring_action_tensors = actions

                # Store actions by representation
                if rolling_action_representation.value == "bernoulli":
                    rolling_actions[:, step_idx, :] = rolling_action_tensor.long()
                else:
                    rolling_actions[:, step_idx] = rolling_action_tensor.long()
                scoring_actions[:, step_idx] = scoring_action_tensors.long()

                # Step each environment
                for env_idx, (env, obs) in enumerate(zip(self.envs, observations, strict=True)):
                    phase = obs["phase"]
                    phases[env_idx, step_idx] = phase

                    if phase == 0:
                        action: Action = {
                            "hold_mask": convert_rolling_action_to_hold_mask(
                                rolling_action_tensor[env_idx], rolling_action_representation
                            )
                        }
                    else:
                        score_category = int(scoring_action_tensors[env_idx].item())
                        action = {"score_category": score_category}

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    rewards[env_idx, step_idx] = float(reward)

                    # Encode next state
                    next_state_tensor = phi(next_obs, features, device)
                    next_states[env_idx, step_idx, :] = next_state_tensor
                    env_unwrapped: YahtzeeEnv = cast("YahtzeeEnv", env.unwrapped)
                    episode_received_bonus[env_idx] = (
                        env_unwrapped.state.score_sheet[0:6].sum() * 1.0
                        - MINIMUM_UPPER_SCORE_FOR_BONUS
                    ) / MINIMUM_UPPER_SCORE_FOR_BONUS

                    if terminated or truncated:
                        next_obs, _ = env.reset()
                    observations[env_idx] = next_obs

        # ---- Check which episodes received the upper section bonus ----
        # for env_idx, env in enumerate(self.envs):

        # if int(env_unwrapped.state.score_sheet[0:6].sum()) >= MINIMUM_UPPER_SCORE_FOR_BONUS:
        #    episode_received_bonus[env_idx] = 1

        # ---- Build next_v_baseline via time-shift (no second forward) ----
        # v_baseline[e, t] = V(s_t)
        # We want next_v_baseline[e, t] = V(s_{t+1}) for TD(0) bootstrapping.
        # For the last step (t = T-1), this value is unused; we set it to 0.
        next_v_baseline = torch.zeros_like(v_baseline)
        if num_steps > 1:
            next_v_baseline[:, :-1] = v_baseline[:, 1:]  # shift along time within each env

        # ---- Flatten all (B, T, ...) -> (B*T, ...) ----
        states_flat = states.view(-1, state_size)
        next_states_flat = next_states.view(-1, state_size)
        v_baseline_flat = v_baseline.view(-1)
        next_v_baseline_flat = next_v_baseline.view(-1)
        phases_flat = phases.view(-1)
        rewards_flat = rewards.view(-1)

        if rolling_action_representation.value == "bernoulli":
            rolling_actions_flat = rolling_actions.view(-1, 5)
        else:
            rolling_actions_flat = rolling_actions.view(-1)

        # Broadcast episode_received_bonus (B,) to (B, T) then flatten to (B*T,)
        received_bonus_expanded = episode_received_bonus.unsqueeze(1).expand(
            self.batch_size, num_steps
        )
        received_bonus_flat = received_bonus_expanded.reshape(-1)

        batch: EpisodeBatch = {
            "states": states_flat,
            "rolling_actions": rolling_actions_flat,
            "scoring_actions": scoring_actions.view(-1),
            "rewards": rewards_flat,
            "next_states": next_states_flat,
            "phases": phases_flat,
            "v_baseline": v_baseline_flat,
            "next_v_baseline": next_v_baseline_flat,
            "received_bonus": received_bonus_flat,
        }
        return batch
