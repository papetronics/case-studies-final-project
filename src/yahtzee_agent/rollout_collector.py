"""Rollout collector for parallel episode generation."""

from typing import TYPE_CHECKING, cast

import gymnasium as gym
import torch
from gymnasium.vector import SyncVectorEnv

from environments.full_yahtzee_env import YahtzeeEnv
from yahtzee_agent.features import PhiFeature

from .model import (
    convert_rolling_action_to_hold_mask,
    get_input_dimensions,
    phi,
    sample_action,
)

if TYPE_CHECKING:
    from .model import YahtzeeAgent


class RolloutBuffer:
    """Container for rollout data."""

    def __init__(  # noqa: PLR0913
        self,
        states: torch.Tensor,
        rolling_actions: torch.Tensor,
        scoring_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        phases: torch.Tensor,
        logp_old: torch.Tensor,
    ):
        self.states = states
        self.rolling_actions = rolling_actions
        self.scoring_actions = scoring_actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.phases = phases
        self.logp_old = logp_old
        self.returns: torch.Tensor | None = None
        self.td_target: torch.Tensor | None = None


def postprocess_rollout(
    buffer: RolloutBuffer, mode: str, gamma: float, value_fn: "YahtzeeAgent"
) -> None:
    """Post-process rollout buffer based on algorithm mode."""
    if mode == "reinforce":
        # Compute Monte Carlo returns per timestep
        num_steps = buffer.states.shape[0]
        returns = torch.zeros_like(buffer.rewards)

        # Need to identify episode boundaries using dones
        g = 0.0
        for t in reversed(range(num_steps)):
            if buffer.dones[t]:
                g = 0.0
            g = float(buffer.rewards[t].item()) + gamma * g
            returns[t] = g

        buffer.returns = returns

    elif mode == "td0":
        # Compute TD target per timestep
        with torch.no_grad():
            _, _, v_next = value_fn(buffer.next_states)
            v_next = v_next.squeeze(-1)
        buffer.td_target = buffer.rewards + gamma * (1.0 - buffer.dones.float()) * v_next


class RolloutCollector:
    """Collects episode rollouts using vectorized parallel environments."""

    def __init__(
        self,
        policy_net: "YahtzeeAgent",
        batch_size: int,
        num_steps_per_episode: int,
        stagger_environments: bool = False,
    ):
        """Initialize rollout collector with vectorized environments."""
        self.policy_net = policy_net
        self.batch_size = batch_size
        self.num_steps_per_episode = num_steps_per_episode
        self.stagger_environments = stagger_environments

        # Create vectorized environment
        self.vec_env = SyncVectorEnv(
            [lambda: gym.make("FullYahtzee-v1") for _ in range(batch_size)]
        )
        self.vec_env.reset()

        # Stagger environments if needed
        if self.stagger_environments:
            for env_idx in range(batch_size):
                target_turn = env_idx % 13
                if target_turn == 0:
                    continue

                # Step this env to target turn using random actions
                env = self.vec_env.envs[env_idx]
                turn_count = 0

                while turn_count < target_turn:
                    unwrapped: YahtzeeEnv = cast("YahtzeeEnv", env.unwrapped)
                    obs = unwrapped.observe()

                    if obs["phase"] == 0:  # Rolling
                        sampled = env.action_space.sample()
                        action = {"hold_mask": sampled["hold_mask"].astype(bool)}
                    else:  # Scoring
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

    def collect(self) -> RolloutBuffer:  # noqa: PLR0915
        """Collect episodes from vectorized parallel environments."""
        # Get current observations from vectorized env
        observations = [cast("YahtzeeEnv", env.unwrapped).observe() for env in self.vec_env.envs]

        device = next(self.policy_net.parameters()).device
        features = cast("list[PhiFeature]", self.policy_net.features)
        state_size = get_input_dimensions(features)
        rolling_action_representation = self.policy_net.rolling_action_representation

        # Pre-allocate tensors (batch_size, num_steps, ...)
        states = torch.zeros(
            self.batch_size,
            self.num_steps_per_episode,
            state_size,
            dtype=torch.float32,
            device=device,
        )

        if rolling_action_representation.value == "bernoulli":
            rolling_actions = torch.zeros(
                self.batch_size, self.num_steps_per_episode, 5, dtype=torch.long, device=device
            )
        else:  # CATEGORICAL
            rolling_actions = torch.zeros(
                self.batch_size, self.num_steps_per_episode, dtype=torch.long, device=device
            )

        scoring_actions = torch.zeros(
            self.batch_size, self.num_steps_per_episode, dtype=torch.long, device=device
        )
        rewards = torch.zeros(
            self.batch_size, self.num_steps_per_episode, dtype=torch.float32, device=device
        )
        next_states = torch.zeros(
            self.batch_size,
            self.num_steps_per_episode,
            state_size,
            dtype=torch.float32,
            device=device,
        )
        dones = torch.zeros(
            self.batch_size, self.num_steps_per_episode, dtype=torch.long, device=device
        )
        phases = torch.zeros(
            self.batch_size, self.num_steps_per_episode, dtype=torch.long, device=device
        )
        logp_old = torch.zeros(
            self.batch_size, self.num_steps_per_episode, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            for step_idx in range(self.num_steps_per_episode):
                # Batch convert observations to state tensors
                state_tensors = torch.stack([phi(obs, features, device) for obs in observations])
                states[:, step_idx, :] = state_tensors

                # Single batched forward pass
                rolling_probs, scoring_probs, v_ests = self.policy_net.forward(state_tensors)

                # Sample actions
                actions, log_probs, _ = sample_action(
                    rolling_probs, scoring_probs, v_ests, rolling_action_representation
                )
                rolling_action_tensor, scoring_action_tensor = actions
                rolling_log_prob, scoring_log_prob = log_probs

                # Store actions
                if rolling_action_representation.value == "bernoulli":
                    rolling_actions[:, step_idx, :] = rolling_action_tensor.long()
                else:  # CATEGORICAL
                    rolling_actions[:, step_idx] = rolling_action_tensor.long()
                scoring_actions[:, step_idx] = scoring_action_tensor.long()

                # Step each environment individually (mixed action types per step)
                for env_idx, obs in enumerate(observations):
                    env = self.vec_env.envs[env_idx]
                    phase = obs["phase"]
                    phases[env_idx, step_idx] = phase

                    if phase == 0:  # Rolling
                        logp_old[env_idx, step_idx] = rolling_log_prob[env_idx]
                        hold_mask = convert_rolling_action_to_hold_mask(
                            rolling_action_tensor[env_idx], rolling_action_representation
                        )
                        action = {"hold_mask": hold_mask}
                    else:  # Scoring
                        logp_old[env_idx, step_idx] = scoring_log_prob[env_idx]
                        score_category = int(scoring_action_tensor[env_idx].cpu().item())
                        action = {"score_category": score_category}  # type: ignore

                    # Step individual environment
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    rewards[env_idx, step_idx] = float(reward)
                    dones[env_idx, step_idx] = int(terminated or truncated)

                    # Get next state
                    next_state_tensor = phi(next_obs, features, device)
                    next_states[env_idx, step_idx, :] = next_state_tensor

                    # Update observation
                    if terminated or truncated:
                        next_obs, _ = env.reset()
                    observations[env_idx] = next_obs

        # Flatten batch and time dimensions
        if rolling_action_representation.value == "bernoulli":
            rolling_actions_flat = rolling_actions.view(-1, 5)
        else:
            rolling_actions_flat = rolling_actions.view(-1)

        return RolloutBuffer(
            states=states.view(-1, state_size),
            rolling_actions=rolling_actions_flat,
            scoring_actions=scoring_actions.view(-1),
            rewards=rewards.view(-1),
            next_states=next_states.view(-1, state_size),
            dones=dones.view(-1),
            phases=phases.view(-1),
            logp_old=logp_old.view(-1),
        )

    def close(self) -> None:
        """Close vectorized environment."""
        self.vec_env.close()
