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
from utilities.scoring_helper import (
    BONUS_POINTS,
    MINIMUM_UPPER_SCORE_FOR_BONUS,
    NUMBER_OF_CATEGORIES,
    YAHTZEE_SCORE,
    ScoreCategory,
)
from yahtzee_agent.features import create_features
from yahtzee_agent.model import (
    YahtzeeAgent,
    convert_rolling_action_to_hold_mask,
    phi,
    select_action,
)
from yahtzee_agent.trainer import Algorithm


def create_env() -> gym.Env[Observation, Action]:
    """Create a single Yahtzee environment."""
    return gym.make("FullYahtzee-v1")


def load_model(checkpoint_path: str) -> YahtzeeAgent:
    """Load a trained model from a checkpoint."""
    torch.serialization.add_safe_globals([Algorithm])
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
        he_kaiming_initialization=False,
        use_layer_norm=hparams.get("use_layer_norm", True),
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
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Run batch of games and return (total_rewards, final_scoresheets, category_usage_by_turn).

    category_usage_by_turn is a 3D array of shape (batch_size, 13_turns, 2) where:
    - [:, :, 0] contains the category index used in that turn
    - [:, :, 1] contains the score earned in that category
    """
    observations: Any = envs.reset()[0]
    total_rewards = np.zeros(batch_size, dtype=np.float64)

    # Track which category was used in which turn and what score was earned
    # Shape: (batch_size, 13_turns, 2) - [category_index, score]
    category_usage_by_turn = np.zeros((batch_size, NUMBER_OF_CATEGORIES, 2), dtype=np.int32)
    turn_number = np.zeros(batch_size, dtype=np.int32)  # Track current turn for each game
    previous_phase = np.zeros(batch_size, dtype=np.int32)  # Track previous phase

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

            # Track category usage when scoring (phase is SCORING, i.e., phase == 1)
            for i in range(batch_size):
                if observations["phase"][i] == 1:  # SCORING phase
                    category = actions["score_category"][i]
                    if turn_number[i] < NUMBER_OF_CATEGORIES:
                        category_usage_by_turn[i, turn_number[i], 0] = category

            rewards: NDArray[np.float64]
            observations, rewards = envs.step(actions)[:2]
            total_rewards += rewards

            # Detect transition from SCORING (1) to ROLLING (0) phase - indicates turn completion
            for i in range(batch_size):
                if (
                    previous_phase[i] == 1
                    and observations["phase"][i] == 0
                    and turn_number[i] < NUMBER_OF_CATEGORIES
                ):
                    # We just completed a scoring action
                    category = category_usage_by_turn[i, turn_number[i], 0]
                    category_usage_by_turn[i, turn_number[i], 1] = observations["score_sheet"][i][
                        category
                    ]
                    turn_number[i] += 1
                previous_phase[i] = observations["phase"][i]

    return total_rewards, observations["score_sheet"], category_usage_by_turn


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


def print_bonus_and_yahtzee_stats(final_scoresheets: NDArray[np.int32]) -> None:
    """Print percentage of games earning bonus and Yahtzee."""
    n_games = len(final_scoresheets)

    # Calculate upper section totals and bonus eligibility
    upper_totals = np.sum(final_scoresheets[:, 0:6], axis=1)
    games_with_bonus = np.sum(upper_totals >= MINIMUM_UPPER_SCORE_FOR_BONUS)
    pct_bonus = (games_with_bonus / n_games) * 100

    # Calculate Yahtzee percentage (Yahtzee is category 11, scores YAHTZEE_SCORE)
    yahtzee_scores = final_scoresheets[:, ScoreCategory.YAHTZEE]
    games_with_yahtzee = np.sum(yahtzee_scores >= YAHTZEE_SCORE)
    pct_yahtzee = (games_with_yahtzee / n_games) * 100

    print("\n" + "=" * 60)
    print("BONUS AND YAHTZEE STATISTICS")
    print("=" * 60)
    print(f"% of games earning Upper Bonus (≥63): {pct_bonus:6.2f}%")
    print(f"% of games earning a Yahtzee:         {pct_yahtzee:6.2f}%")
    print("=" * 60)


def print_median_turn_by_category(category_usage: NDArray[np.int32]) -> None:
    """Print median turn number when each category was used.

    Args:
        category_usage: Array of shape (n_games, 13, 2) where [:, :, 0] is category index
    """
    n_games = category_usage.shape[0]

    # For each category, collect all turn numbers when it was used
    category_turns: dict[int, list[int]] = {i: [] for i in range(NUMBER_OF_CATEGORIES)}

    for game_idx in range(n_games):
        for turn_idx in range(NUMBER_OF_CATEGORIES):
            category = category_usage[game_idx, turn_idx, 0]
            category_turns[category].append(turn_idx + 1)  # +1 for 1-indexed turns

    print("\n" + "=" * 60)
    print("MEDIAN TURN NUMBER BY CATEGORY")
    print("=" * 60)
    print(f"{'Category':<20} │ {'Median Turn':>12}")
    print("─" * 60)

    for i, label in enumerate(ScoreCategory.LABELS):
        turns = category_turns[i]
        median_turn = np.median(turns) if turns else 0
        print(f"{label:<20} │ {median_turn:>12.1f}")

    print("=" * 60)


def print_top_categories_by_turn(category_usage: NDArray[np.int32]) -> None:
    """Print top 3 most used categories for each turn with usage % and median score.

    Args:
        category_usage: Array of shape (n_games, 13, 2) where [:, :, 0] is category, [:, :, 1] is score
    """
    n_games = category_usage.shape[0]

    print("\n" + "=" * 90)
    print("TOP 3 CATEGORIES BY TURN")
    print("=" * 90)
    print(f"{'Turn':<5} │ {'Category':<20} │ {'Usage %':>8} │ {'Median Score':>12}")
    print("─" * 90)

    for turn_idx in range(NUMBER_OF_CATEGORIES):
        # Count category usage for this turn across all games
        category_counts: dict[int, int] = {}
        category_scores: dict[int, list[int]] = {i: [] for i in range(NUMBER_OF_CATEGORIES)}

        for game_idx in range(n_games):
            category = category_usage[game_idx, turn_idx, 0]
            score = category_usage[game_idx, turn_idx, 1]
            category_counts[category] = category_counts.get(category, 0) + 1
            category_scores[category].append(score)

        # Sort by count and get top 3
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        for rank, (category, count) in enumerate(top_categories):
            usage_pct = (count / n_games) * 100
            median_score = np.median(category_scores[category])
            label = ScoreCategory.LABELS[category]

            if rank == 0:
                print(
                    f"{turn_idx + 1:<5} │ {label:<20} │ {usage_pct:>7.1f}% │ {median_score:>12.1f}"
                )
            else:
                print(f"{'':5} │ {label:<20} │ {usage_pct:>7.1f}% │ {median_score:>12.1f}")

        if turn_idx < NUMBER_OF_CATEGORIES - 1:
            print("─" * 90)

    print("=" * 90)


def print_score_distribution(total_rewards: NDArray[np.float64]) -> None:
    """Print score distribution table showing P(score >= n) for various thresholds.

    Args:
        total_rewards: Array of total scores from all games
    """
    n_games = len(total_rewards)
    thresholds = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500]

    print("\n" + "=" * 50)
    print("SCORE DISTRIBUTION")
    print("=" * 50)
    print(f"{'n':<10} │ {'P(score ≥ n)':>20}")
    print("─" * 50)

    for threshold in thresholds:
        count = np.sum(total_rewards >= threshold)
        probability = count / n_games

        # Format probability with appropriate precision
        if probability == 1.0:
            prob_str = "1.000000"
        elif probability >= 0.0001:  # noqa: PLR2004
            prob_str = f"{probability:.6f}"
        else:
            prob_str = f"{probability:.6e}"

        print(f"{threshold:<10} │ {prob_str:>20}")

    print("=" * 50)


def evaluate_model(checkpoint_path: str, num_games: int, baseline_path: str, alpha: float) -> None:
    """Evaluate trained model over multiple games."""
    model = load_model(checkpoint_path)
    baseline = load_baseline_stats(baseline_path)

    print(f"Running {num_games} games...")

    envs = SyncVectorEnv([create_env for _ in range(num_games)])
    batch_rewards, batch_scoresheets, category_usage = run_batch_games(envs, model, num_games)
    envs.close()

    stats = compute_statistics(batch_rewards, batch_scoresheets)
    print_results_table(stats, baseline, alpha)

    # Print additional statistics
    print_bonus_and_yahtzee_stats(batch_scoresheets)
    print_median_turn_by_category(category_usage)
    print_top_categories_by_turn(category_usage)
    print_score_distribution(batch_rewards)


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
