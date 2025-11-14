from typing import TYPE_CHECKING, TypedDict, cast

import gymnasium as gym
import torch

from environments.full_yahtzee_env import Action, Observation, YahtzeeEnv
from utilities.return_calculators import ReturnCalculator

from .model import get_input_dimensions, phi, sample_action

if TYPE_CHECKING:
    from .model import YahtzeeAgent


class EpisodeBatch(TypedDict):
    """Typed dict for episode batch data returned by the dataset.

    Each tensor has shape (num_steps, ...) where num_steps=3 for our single-turn episodes.
    DataLoader will stack these into (BATCH_SIZE, num_steps, ...).
    """

    states: torch.Tensor  # (3, state_size) float32
    rolling_actions: torch.Tensor  # (3, 5) int (binary mask)
    scoring_actions: torch.Tensor  # (3,) int (category index 0-12)
    rewards: torch.Tensor  # (3,) float32
    next_states: torch.Tensor  # (3, state_size) float32
    phases: torch.Tensor  # (3,) int (0=rolling, 1=scoring)


class SelfPlayDataset(torch.utils.data.Dataset[EpisodeBatch]):
    """
    A dataset that collects episodes by playing against itself using the current policy.

    This dataset generates episodes on-the-fly by interacting with the Yahtzee environment
    using the current policy network. Each episode represents a single turn (3 steps: roll, roll, score).

    Returns a dict of tensors with shapes (3, ...):
      - "states": (3, state_size) - State tensor for each step
      - "rolling_actions": (3, 5) - Binary mask for which dice to reroll
      - "scoring_actions": (3,) - Category to score (0-12)
      - "rewards": (3,) - Immediate rewards
      - "next_states": (3, state_size) - Next state tensors
      - "phases": (3,) - Phase indicator (0=rolling, 1=scoring)
    """

    def __init__(
        self,
        policy_net: "YahtzeeAgent",
        return_calculator: ReturnCalculator,
        size: int,
    ) -> None:
        """
        Initialize the self-play dataset.

        Args:
            policy_net: The policy network to use for collecting episodes
            return_calculator: The return calculator to compute returns for each episode
            size: The number of episodes in the dataset (episodes per epoch)
        """
        self.policy_net: YahtzeeAgent = policy_net
        self.return_calculator = return_calculator
        self.size = size
        self.env: gym.Env[Observation, Action] = gym.make("FullYahtzee-v1")
        self.env.reset()

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> EpisodeBatch:
        """
        Collect and return a single episode's data as SARS' tuples.

        Args:
            idx: Index (not used, but required by Dataset interface)

        Returns
        -------
        EpisodeBatch
            A dict containing episode data with keys:
            - "states": (3, state_size) float32
            - "rolling_actions": (3, 5) int
            - "scoring_actions": (3,) int
            - "rewards": (3,) float32
            - "next_states": (3, state_size) float32
            - "phases": (3,) int
        """
        unwrapped: YahtzeeEnv = cast("YahtzeeEnv", self.env.unwrapped)
        observation = unwrapped.observe()

        # This is a bit of a hack, the environment supports full turns, but our model is single-turn
        # so we are just going to cut it off after 3 steps and pretend that is an episode.
        # we reset the environment whenever it terminates, that brings us back to an empty scoresheet.

        num_steps = 3
        device = next(self.policy_net.parameters()).device

        # Calculate dimensions
        state_size = get_input_dimensions(self.policy_net.bonus_flags)

        # Pre-allocate tensors for episode (3 steps)
        states = torch.zeros(num_steps, state_size, dtype=torch.float32, device=device)
        rolling_actions = torch.zeros(num_steps, 5, dtype=torch.long, device=device)
        scoring_actions = torch.zeros(num_steps, dtype=torch.long, device=device)
        rewards = torch.zeros(num_steps, dtype=torch.float32, device=device)
        next_states = torch.zeros(num_steps, state_size, dtype=torch.float32, device=device)
        phases = torch.zeros(num_steps, dtype=torch.long, device=device)

        with torch.no_grad():  # No gradients needed during data collection
            for step_idx in range(num_steps):  # roll, roll, score
                # Get current state tensor
                state_tensor = phi(observation, self.policy_net.bonus_flags, device)
                states[step_idx] = state_tensor

                # Forward pass to get action probabilities
                input_tensor = state_tensor.unsqueeze(0)
                rolling_probs, scoring_probs, _ = self.policy_net.forward(input_tensor)

                # Sample actions
                actions, _, _ = sample_action(rolling_probs, scoring_probs, torch.tensor(0.0))
                rolling_action_tensor, scoring_action_tensor = actions

                rolling_actions[step_idx] = rolling_action_tensor.squeeze()
                scoring_actions[step_idx] = scoring_action_tensor.squeeze().long()

                # Execute action in environment
                action: Action
                phase = observation["phase"]
                phases[step_idx] = phase

                if phase == 0:
                    action = {"hold_mask": rolling_action_tensor.cpu().numpy().astype(bool)}
                else:
                    score_category: int = int(scoring_action_tensor.cpu().item())
                    action = {"score_category": score_category}

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                rewards[step_idx] = float(reward)

                # Get next state tensor
                next_state_tensor = phi(next_observation, self.policy_net.bonus_flags, device)
                next_states[step_idx] = next_state_tensor

                observation = next_observation

                if terminated or truncated:
                    observation, _ = self.env.reset()

        # sanity check: after 3 rolls we should always have rolls_used == 0 (new turn) and phase == 0
        assert observation["rolls_used"] == 0 and observation["phase"] == 0

        return {
            "states": states,
            "rolling_actions": rolling_actions,
            "scoring_actions": scoring_actions,
            "rewards": rewards,
            "next_states": next_states,
            "phases": phases,
        }
