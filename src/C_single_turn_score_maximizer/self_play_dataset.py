from typing import TYPE_CHECKING, TypedDict, cast

import gymnasium as gym
import numpy as np
import torch

from environments.full_yahtzee_env import Action, Observation, YahtzeeEnv
from utilities.return_calculators import ReturnCalculator

from .model import DICE_MASKS, get_input_dimensions, phi, sample_action

if TYPE_CHECKING:
    from .model import YahtzeeAgent


class EpisodeBatch(TypedDict):
    """Typed dict for episode batch data returned by the dataset.

    Tensors are pre-flattened: (BATCH_SIZE * num_steps, ...) where:
    - BATCH_SIZE = number of parallel episodes collected
    - num_steps = 3 for our single-turn episodes (roll, roll, score)

    The dataset collects in parallel with batched forward passes and flattens
    before returning to avoid reshape operations in the trainer.
    """

    states: torch.Tensor  # (BATCH_SIZE*3, state_size) float32
    rolling_actions: torch.Tensor  # (BATCH_SIZE*3, 5) int (binary mask)
    scoring_actions: torch.Tensor  # (BATCH_SIZE*3,) int (category index 0-12)
    rewards: torch.Tensor  # (BATCH_SIZE*3,) float32
    next_states: torch.Tensor  # (BATCH_SIZE*3, state_size) float32
    phases: torch.Tensor  # (BATCH_SIZE*3,) int (0=rolling, 1=scoring)


class SelfPlayDataset(torch.utils.data.Dataset[EpisodeBatch]):
    """
    A dataset that collects episodes by playing against itself using the current policy.

    This dataset generates episodes on-the-fly by interacting with the Yahtzee environment
    using the current policy network. Each __getitem__ call collects a full batch of episodes
    in parallel with batched forward passes for GPU efficiency.

    Returns a dict of pre-flattened tensors with shapes (BATCH_SIZE*3, ...):
      - "states": (BATCH_SIZE*3, state_size) - State tensor for each step
      - "rolling_actions": (BATCH_SIZE*3, 5) - Binary mask for which dice to reroll
      - "scoring_actions": (BATCH_SIZE*3,) - Category to score (0-12)
      - "rewards": (BATCH_SIZE*3,) - Immediate rewards
      - "next_states": (BATCH_SIZE*3, state_size) - Next state tensors
      - "phases": (BATCH_SIZE*3,) - Phase indicator (0=rolling, 1=scoring)
    """

    def __init__(
        self,
        policy_net: "YahtzeeAgent",
        return_calculator: ReturnCalculator,
        size: int,
        batch_size: int,
    ) -> None:
        """
        Initialize the self-play dataset.

        Args:
            policy_net: The policy network to use for collecting episodes
            return_calculator: The return calculator to compute returns for each episode
            size: The number of batches in the dataset (batches per epoch)
            batch_size: The number of parallel episodes to collect per batch
        """
        self.policy_net: YahtzeeAgent = policy_net
        self.return_calculator = return_calculator
        self.size = size
        self.batch_size = batch_size

        # Create pool of environments for parallel collection
        # Stagger each environment to a different turn (0-12) to avoid temporal bias
        self.envs: list[gym.Env[Observation, Action]] = []
        for env_idx in range(batch_size):
            env = gym.make("FullYahtzee-v1")
            env.reset()

            # Advance environment to turn (env_idx % 13) using random actions
            target_turn = env_idx % 13
            unwrapped: YahtzeeEnv = cast("YahtzeeEnv", env.unwrapped)

            # Number of scores filled = 13 - number of available categories
            obs = unwrapped.observe()
            num_scores_filled = 13 - int(obs["score_sheet_available_mask"].sum())

            while num_scores_filled < target_turn:
                obs = unwrapped.observe()
                # Sample action based on current phase
                if obs["phase"] == 0:  # Rolling phase
                    sampled = env.action_space.sample()
                    action: Action = {"hold_mask": sampled["hold_mask"].astype(bool)}
                else:  # Scoring phase
                    # Sample from available categories
                    available_categories = [
                        i for i in range(13) if obs["score_sheet_available_mask"][i] == 1
                    ]
                    sampled = env.action_space.sample()
                    score_category = int(sampled["score_category"])
                    # Ensure we pick a valid category
                    if score_category not in available_categories:
                        score_category = available_categories[0] if available_categories else 0
                    action = {"score_category": score_category}

                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset()
                    break
                obs = unwrapped.observe()
                num_scores_filled = 13 - int(obs["score_sheet_available_mask"].sum())

            self.envs.append(env)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> EpisodeBatch:
        """
        Collect and return a batch of episodes in parallel with batched forward passes.

        Args:
            idx: Index (not used, but required by Dataset interface)

        Returns
        -------
        EpisodeBatch
            A dict containing pre-flattened batch episode data with keys:
            - "states": (BATCH_SIZE*3, state_size) float32
            - "rolling_actions": (BATCH_SIZE*3, 5) int
            - "scoring_actions": (BATCH_SIZE*3,) int
            - "rewards": (BATCH_SIZE*3,) float32
            - "next_states": (BATCH_SIZE*3, state_size) float32
            - "phases": (BATCH_SIZE*3,) int
        """
        # Get current observations from all environments
        observations = []
        for env in self.envs:
            unwrapped: YahtzeeEnv = cast("YahtzeeEnv", env.unwrapped)
            observations.append(unwrapped.observe())

        num_steps = 3
        device = next(self.policy_net.parameters()).device
        state_size = get_input_dimensions(self.policy_net.bonus_flags)

        # Pre-allocate tensors for all episodes (BATCH_SIZE, 3, ...)
        states = torch.zeros(
            self.batch_size, num_steps, state_size, dtype=torch.float32, device=device
        )
        rolling_actions = torch.zeros(self.batch_size, num_steps, dtype=torch.long, device=device)
        scoring_actions = torch.zeros(self.batch_size, num_steps, dtype=torch.long, device=device)
        rewards = torch.zeros(self.batch_size, num_steps, dtype=torch.float32, device=device)
        next_states = torch.zeros(
            self.batch_size, num_steps, state_size, dtype=torch.float32, device=device
        )
        phases = torch.zeros(self.batch_size, num_steps, dtype=torch.long, device=device)

        with torch.no_grad():  # No gradients needed during data collection
            for step_idx in range(num_steps):  # roll, roll, score
                # Batch convert all observations to state tensors
                state_tensors = torch.stack(
                    [phi(obs, self.policy_net.bonus_flags, device) for obs in observations]
                )  # (BATCH_SIZE, state_size)

                states[:, step_idx, :] = state_tensors

                # Single batched forward pass for all environments
                rolling_probs, scoring_probs, _ = self.policy_net.forward(state_tensors)

                # Sample actions for all environments
                actions, _, _ = sample_action(
                    rolling_probs, scoring_probs, torch.zeros(self.batch_size, device=device)
                )
                rolling_action_tensor, scoring_action_tensors = actions

                rolling_actions[:, step_idx] = rolling_action_tensor.long()
                scoring_actions[:, step_idx] = scoring_action_tensors.long()

                # Step each environment and collect results
                for env_idx, (env, obs) in enumerate(zip(self.envs, observations, strict=True)):
                    phase = obs["phase"]
                    phases[env_idx, step_idx] = phase

                    # Convert action based on phase
                    action: Action
                    if phase == 0:
                        action = {
                            "hold_mask": np.array(
                                DICE_MASKS[int(rolling_action_tensor[env_idx].item())], dtype=bool
                            )
                        }
                    else:
                        score_category: int = int(scoring_action_tensors[env_idx].cpu().item())
                        action = {"score_category": score_category}

                    # Step environment
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    rewards[env_idx, step_idx] = float(reward)

                    # Get next state tensor
                    next_state_tensor = phi(next_obs, self.policy_net.bonus_flags, device)
                    next_states[env_idx, step_idx, :] = next_state_tensor

                    # Update observation for next step
                    if terminated or truncated:
                        next_obs, _ = env.reset()
                    observations[env_idx] = next_obs

        # Sanity check: after 3 steps all envs should be at start of new turn
        for obs in observations:
            assert obs["rolls_used"] == 0 and obs["phase"] == 0

        # Flatten batch and time dimensions: (BATCH_SIZE, 3, ...) -> (BATCH_SIZE*3, ...)
        return {
            "states": states.view(-1, state_size),
            "rolling_actions": rolling_actions.view(-1),
            "scoring_actions": scoring_actions.view(-1),
            "rewards": rewards.view(-1),
            "next_states": next_states.view(-1, state_size),
            "phases": phases.view(-1),
        }
