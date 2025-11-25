"""Evaluation script for trained Yahtzee models."""

import argparse
import json
from dataclasses import dataclass
from math import erf, sqrt
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from numpy.typing import NDArray
from tqdm import tqdm

from environments.full_yahtzee_env import Action, Observation
from utilities.scoring_helper import BONUS_POINTS, MINIMUM_UPPER_SCORE_FOR_BONUS, ScoreCategory
from yahtzee_agent.features import create_features
from yahtzee_agent.model import (
    YahtzeeAgent,
    convert_rolling_action_to_hold_mask,
    phi,
    select_action,
)


def create_env() -> gym.Env[Observation, Action]:
    """Create a single Yahtzee environment."""
    return gym.make("FullYahtzee-v1")


def load_model(checkpoint_path: str) -> YahtzeeAgent:
    """Load a trained model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]
    features = create_features(hparams["phi_features"].split(","))

    model = YahtzeeAgent(
        hidden_size=hparams["hidden_size"],
        num_hidden=hparams["num_hidden"],
        dropout_rate=hparams["dropout_rate"],
        activation_function=hparams["activation_function"],
        features=features,
        rolling_action_representation=hparams["rolling_action_representation"],
        he_kaiming_initialization=True,
    )

    policy_net_state_dict = {
        k.replace("policy_net.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("policy_net.")
    }
    model.load_state_dict(policy_net_state_dict)
    model.eval()
    return model


def run_batch_games(
    envs: SyncVectorEnv, model: YahtzeeAgent, batch_size: int
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Run batch of games and return (total_rewards, final_scoresheets)."""
    observations: Any = envs.reset()[0]
    total_rewards = np.zeros(batch_size, dtype=np.float64)

    with torch.no_grad():
        for _ in tqdm(range(39), desc="Steps", leave=False):
            input_tensors = torch.stack(
                [
                    phi(
                        {
                            "dice": observations["dice"][i],
                            "rolls_used": observations["rolls_used"][i],
                            "score_sheet": observations["score_sheet"][i],
                            "score_sheet_available_mask": observations[
                                "score_sheet_available_mask"
                            ][i],
                            "phase": observations["phase"][i],
                        },
                        model.features,
                        model.device,
                    )
                    for i in range(batch_size)
                ]
            )

            rolling_probs, scoring_probs, _, _ = model.forward(input_tensors)
            rolling_actions, scoring_actions = select_action(
                rolling_probs, scoring_probs, model.rolling_action_representation
            )

            actions = {
                "hold_mask": np.array(
                    [
                        convert_rolling_action_to_hold_mask(
                            rolling_actions[i] if batch_size > 1 else rolling_actions,
                            model.rolling_action_representation,
                        )
                        for i in range(batch_size)
                    ]
                ),
                "score_category": np.array(
                    [
                        scoring_actions[i].cpu().item()
                        if batch_size > 1
                        else scoring_actions.cpu().item()
                        for i in range(batch_size)
                    ]
                ),
            }

            rewards: NDArray[np.float64]
            observations, rewards = envs.step(actions)[:2]
            total_rewards += rewards

    return total_rewards, observations["score_sheet"]


@dataclass
class CategoryStats:
    """Statistics for a single category."""

    mean: float
    variance: float
    std: float
    n_samples: int


def compute_statistics(
    total_rewards: NDArray[np.float64], final_scoresheets: NDArray[np.int32]
) -> list[CategoryStats]:
    """Compute statistics for all categories, upper bonus, and final score."""
    n_games = len(total_rewards)
    stats = []

    for i in range(13):
        scores = final_scoresheets[:, i]
        stats.append(
            CategoryStats(
                mean=float(np.mean(scores)),
                variance=float(np.var(scores, ddof=1)),
                std=float(np.std(scores, ddof=1)),
                n_samples=n_games,
            )
        )

    upper_totals = np.sum(final_scoresheets[:, 0:6], axis=1)
    upper_bonuses = np.where(upper_totals >= MINIMUM_UPPER_SCORE_FOR_BONUS, BONUS_POINTS, 0)
    stats.append(
        CategoryStats(
            mean=float(np.mean(upper_bonuses)),
            variance=float(np.var(upper_bonuses, ddof=1)),
            std=float(np.std(upper_bonuses, ddof=1)),
            n_samples=n_games,
        )
    )

    stats.append(
        CategoryStats(
            mean=float(np.mean(total_rewards)),
            variance=float(np.var(total_rewards, ddof=1)),
            std=float(np.std(total_rewards, ddof=1)),
            n_samples=n_games,
        )
    )

    return stats


def load_baseline_stats(baseline_path: str) -> NDArray[np.float64]:
    """Load baseline statistics from JSON file as array of [mean, variance] pairs."""
    with open(baseline_path) as f:
        return np.array(json.load(f))


def compute_z_test(
    sample_mean: float, sample_std: float, n_samples: int, baseline_mean: float
) -> tuple[float, float]:
    """Compute one-sample Z-test returning (z_score, p_value)."""
    se = sample_std / np.sqrt(n_samples)

    if se == 0:
        if sample_mean == baseline_mean:
            return 0.0, 1.0
        z_score = np.inf if sample_mean > baseline_mean else -np.inf
        return z_score, 0.0

    z_score = (sample_mean - baseline_mean) / se
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z_score) / sqrt(2))))
    return z_score, p_value


def get_status_indicator(
    sample_mean: float, baseline_mean: float, p_value: float, alpha: float
) -> str:
    """Return ✅/⚠️/❌ based on statistical significance and direction."""
    if p_value >= alpha:
        return "✅"
    return "⚠️ " if sample_mean > baseline_mean else "❌"


def format_number(value: float, width: int, decimals: int = 2) -> str:
    """Format number with decimal alignment."""
    if value == float("inf"):
        return "inf".rjust(width)
    if value == float("-inf"):
        return "-inf".rjust(width)
    formatted = f"{value:.{decimals}f}"
    parts = formatted.split(".")
    return f"{parts[0].rjust(width - decimals - 1)}.{parts[1]}"


def format_pvalue(p: float) -> str:
    """Format p-value with appropriate precision."""
    if p == 0:
        return "0.0000"
    if p >= 0.0001:  # noqa: PLR2004
        return f"{p:.4f}"
    return f"{p:.2e}"


def print_results_table(
    stats: list[CategoryStats], baseline: NDArray[np.float64], alpha: float
) -> None:
    """Print formatted table of evaluation results with baseline comparison."""
    labels = [*ScoreCategory.LABELS, "Upper Bonus", "Final Score"]

    data = []
    for i, (stat, label) in enumerate(zip(stats, labels, strict=True)):
        base_mean, _ = baseline[i]
        diff = stat.mean - base_mean
        z_score, p_value = compute_z_test(stat.mean, stat.std, stat.n_samples, base_mean)
        indicator = get_status_indicator(stat.mean, base_mean, p_value, alpha)
        data.append((label, stat.mean, stat.variance, base_mean, diff, z_score, p_value, indicator))

    print(
        f"\n{'Category':<17} │ {'Mean':>8}   {'σ²':>8} │ {'DP-Mean':>8}   {'Diff':>8} │ {'Z':>8}   {'p':>10} │"
    )
    print("─" * 92)

    for label, mean, var, base_mean, diff, z, p, ind in data[:13]:
        print(
            f"{label:<17} │ {format_number(mean, 8)}   {format_number(var, 8)} │ "
            f"{format_number(base_mean, 8)}   {format_number(diff, 8)} │ {format_number(z, 8)}   "
            f"{format_pvalue(p):>10} │ {ind}"
        )

    print("─" * 92)
    label, mean, var, base_mean, diff, z, p, ind = data[13]
    print(
        f"{label:<17} │ {format_number(mean, 8)}   {format_number(var, 8)} │ "
        f"{format_number(base_mean, 8)}   {format_number(diff, 8)} │ {format_number(z, 8)}   "
        f"{format_pvalue(p):>10} │ {ind}"
    )

    print("─" * 92)
    label, mean, var, base_mean, diff, z, p, ind = data[14]
    print(
        f"{label:<17} │ {format_number(mean, 8)}   {format_number(var, 8)} │ "
        f"{format_number(base_mean, 8)}   {format_number(diff, 8)} │ {format_number(z, 8)}   "
        f"{format_pvalue(p):>10} │ {ind}"
    )


def evaluate_model(checkpoint_path: str, num_games: int, baseline_path: str, alpha: float) -> None:
    """Evaluate trained model over multiple games."""
    model = load_model(checkpoint_path)
    baseline = load_baseline_stats(baseline_path)

    envs = SyncVectorEnv([create_env for _ in range(num_games)])
    batch_rewards, batch_scoresheets = run_batch_games(envs, model, num_games)
    envs.close()

    stats = compute_statistics(batch_rewards, batch_scoresheets)
    print_results_table(stats, baseline, alpha)


def main() -> None:
    """Run evaluation script with command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Yahtzee model")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-games", type=int, default=10000, help="Number of games")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline JSON")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()

    evaluate_model(args.checkpoint_path, args.num_games, args.baseline, args.alpha)


if __name__ == "__main__":
    main()
