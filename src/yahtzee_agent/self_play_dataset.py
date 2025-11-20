"""Dumb dataset that indexes pre-computed rollout buffer."""

from typing import TypedDict, cast

import torch

from yahtzee_agent.rollout_collector import RolloutBuffer


class EpisodeBatch(TypedDict):
    """Batch data returned by the dataset."""

    states: torch.Tensor
    rolling_actions: torch.Tensor
    scoring_actions: torch.Tensor
    phases: torch.Tensor
    logp_old: torch.Tensor
    returns: torch.Tensor | None  # For REINFORCE
    rewards: torch.Tensor | None  # For TD
    next_states: torch.Tensor | None  # For TD
    dones: torch.Tensor | None  # For TD
    td_target: torch.Tensor | None  # For TD


class SelfPlayDataset(torch.utils.data.Dataset[EpisodeBatch]):
    """Dataset that indexes pre-computed rollout buffer."""

    def __init__(self, buffer: RolloutBuffer, mode: str):
        """Initialize dataset with pre-computed buffer."""
        self.buffer = buffer
        self.mode = mode

    def __len__(self) -> int:
        """Return number of timesteps in buffer."""
        return int(self.buffer.states.shape[0])

    def __getitem__(self, idx: int) -> EpisodeBatch:
        """Return single timestep from buffer."""
        item: EpisodeBatch = cast(
            "EpisodeBatch",
            {
                "states": self.buffer.states[idx],
                "rolling_actions": self.buffer.rolling_actions[idx],
                "scoring_actions": self.buffer.scoring_actions[idx],
                "phases": self.buffer.phases[idx],
                "logp_old": self.buffer.logp_old[idx],
            },
        )

        if self.mode == "reinforce":
            if self.buffer.returns is not None:
                item["returns"] = self.buffer.returns[idx]

        elif self.mode == "td0":
            if self.buffer.td_target is not None:
                item["td_target"] = self.buffer.td_target[idx]
            item["rewards"] = self.buffer.rewards[idx]
            item["next_states"] = self.buffer.next_states[idx]
            item["dones"] = self.buffer.dones[idx]

        return item
