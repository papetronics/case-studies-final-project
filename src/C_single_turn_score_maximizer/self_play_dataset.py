from typing import TYPE_CHECKING, SupportsFloat, cast

import gymnasium as gym
import torch

from environments.full_yahtzee_env import Action, Observation, YahtzeeEnv
from utilities.episode import Episode
from utilities.return_calculators import ReturnCalculator

if TYPE_CHECKING:
    from .model import TurnScoreMaximizer


class SelfPlayDataset(torch.utils.data.Dataset[torch.Tensor]):
    """
    A dataset that collects episodes by playing against itself using the current policy.

    This dataset generates episodes on-the-fly by interacting with the Yahtzee environment
    using the current policy network. Each episode represents a single turn (3 steps: roll, roll, score).
    Returns a tensor of shape (3, 3) containing [return, log_prob, v_est] for each step.
    """

    def __init__(
        self,
        policy_net: "TurnScoreMaximizer",
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
        self.policy_net: TurnScoreMaximizer = policy_net
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
        episode = Episode()
        unwrapped: YahtzeeEnv = cast("YahtzeeEnv", self.env.unwrapped)
        observation = unwrapped.observe()

        # This is a bit of a hack, the environment supports full turns, but our model is single-turn
        # so we are just going to cut it off after 3 steps and pretend that is an episode.
        # we reset the environment whenever it terminates, that brings us back to an empty scoresheet.
        reward: SupportsFloat = 0.0

        for _ in range(3):  # roll, roll, score
            actions, log_probs, v_est = self.policy_net.sample_observation(observation)
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

            episode.add_step(observation, action, log_prob, v_est)

            observation, reward, terminated, truncated, _ = self.env.step(action)

            if terminated or truncated:
                observation, _ = self.env.reset()

        episode.set_reward(float(reward))

        # sanity check: after 3 rolls we should always have rolls_used == 0 (new turn) and phase == 0
        assert observation["rolls_used"] == 0 and observation["phase"] == 0

        # Calculate returns for this episode
        returns = self.return_calculator.calculate_returns(episode)

        # Build a tensor of shape (3, 3) containing [return, log_prob, v_est] for each step
        step_data = []
        for step_idx in range(len(episode)):
            step_return = returns[step_idx]
            log_prob = episode.log_probs[step_idx]
            v_est = episode.v_ests[step_idx]

            # Stack: [return, log_prob, v_est]
            # Keep tensors to preserve gradients - don't call .item()!
            step_tensor = torch.stack(
                [
                    torch.tensor(step_return, dtype=torch.float32).to(log_prob.device),
                    log_prob.squeeze(),
                    v_est.squeeze(),
                ]
            )
            step_data.append(step_tensor)

        # Stack all steps: shape (3, 3)
        episode_tensor = torch.stack(step_data)

        return episode_tensor
