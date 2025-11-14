from typing import TYPE_CHECKING, cast

import gymnasium as gym
import torch

from environments.full_yahtzee_env import Action, Observation, YahtzeeEnv
from utilities.return_calculators import ReturnCalculator

from .model import phi, sample_action

if TYPE_CHECKING:
    from .model import YahtzeeAgent


class SelfPlayDataset(torch.utils.data.Dataset[torch.Tensor]):
    """
    A dataset that collects episodes by playing against itself using the current policy.

    This dataset generates episodes on-the-fly by interacting with the Yahtzee environment
    using the current policy network. Each episode represents a single turn (3 steps: roll, roll, score).
    Returns a tensor of shape (3, 3) containing [return, log_prob, v_est] for each step.
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Collect and return a single episode's data as a tensor.

        Args:
            idx: Index (not used, but required by Dataset interface)

        Returns
        -------
        torch.Tensor
            A tensor of shape (3, 3) containing [return, log_prob, v_est] for each of the 3 steps
        """
        unwrapped: YahtzeeEnv = cast("YahtzeeEnv", self.env.unwrapped)
        observation = unwrapped.observe()

        # This is a bit of a hack, the environment supports full turns, but our model is single-turn
        # so we are just going to cut it off after 3 steps and pretend that is an episode.
        # we reset the environment whenever it terminates, that brings us back to an empty scoresheet.

        # Pre-allocate tensors for the episode (3 steps)
        num_steps = 3

        # Get device from policy network
        device = next(self.policy_net.parameters()).device

        # Pre-allocate tensors for log_probs, v_ests, and rewards (preserves gradients with indexing)
        log_probs_tensor = torch.zeros(num_steps, device=device)
        v_ests_tensor = torch.zeros(num_steps, device=device)
        rewards = torch.zeros(num_steps, dtype=torch.float32, device=device)

        for step_idx in range(num_steps):  # roll, roll, score
            input_tensor = phi(observation, self.policy_net.bonus_flags, device).unsqueeze(0)
            rolling_probs, scoring_probs, v_est = self.policy_net.forward(input_tensor)
            actions, log_probs, v_est = sample_action(rolling_probs, scoring_probs, v_est)
            rolling_action_tensor, scoring_action_tensor = actions
            rolling_log_prob, scoring_log_prob = log_probs

            action: Action
            if observation["phase"] == 0:
                action = {"hold_mask": rolling_action_tensor.cpu().numpy().astype(bool)}
                log_prob = rolling_log_prob
            else:
                score_category: int = int(scoring_action_tensor.cpu().item())
                action = {"score_category": score_category}
                log_prob = scoring_log_prob

            # Store log_prob and v_est using indexing (preserves gradients)
            log_probs_tensor[step_idx] = log_prob.squeeze()
            v_ests_tensor[step_idx] = v_est.squeeze()

            observation, reward, terminated, truncated, _ = self.env.step(action)
            rewards[step_idx] = float(reward)

            if terminated or truncated:
                observation, _ = self.env.reset()

        # sanity check: after 3 rolls we should always have rolls_used == 0 (new turn) and phase == 0
        assert observation["rolls_used"] == 0 and observation["phase"] == 0

        # Calculate returns using Monte Carlo (backward pass): G_t = r_t + gamma * G_{t+1}
        gamma = self.return_calculator.gamma
        returns = torch.zeros(num_steps, dtype=torch.float32, device=device)
        g = 0.0
        for t in reversed(range(num_steps)):
            g = rewards[t] + gamma * g
            returns[t] = g

        # Build final tensor of shape (3, 3) containing [return, log_prob, v_est]
        episode_tensor = torch.stack([returns, log_probs_tensor, v_ests_tensor], dim=1)

        return episode_tensor
