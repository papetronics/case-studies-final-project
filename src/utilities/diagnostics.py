"""Diagnostic metric computation utilities for RL training monitoring."""

import torch


def mean_std_per_mask(
    values: torch.Tensor, mask: torch.Tensor, prefix: str
) -> dict[str, torch.Tensor]:
    """Compute mean and std for values within a mask."""
    masked_values = values[mask]
    return {
        f"{prefix}_mean": masked_values.mean(),
        f"{prefix}_std": masked_values.std(unbiased=False),
    }


def bernoulli_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute Bernoulli entropy: -p*log(p) - (1-p)*log(1-p)."""
    p = probs.clamp(1e-6, 1 - 1e-6)
    return -torch.xlogy(p, p) - torch.xlogy(1 - p, 1 - p)


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence for Bernoulli: KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))."""
    eps = 1e-6
    p_safe = p.clamp(eps, 1 - eps)
    q_safe = q.clamp(eps, 1 - eps)
    return torch.xlogy(p_safe, p_safe / q_safe) + torch.xlogy(
        1 - p_safe, (1 - p_safe) / (1 - q_safe)
    )


def compute_advantage_stats(
    advantages: torch.Tensor, phases_flat: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute advantage statistics overall and per-phase.

    Healthy: adv_mean ~0 (by construction), adv_std nonzero (signal strength).
    Red flags: adv_std → 0 = tiny gradients; roll vs score std differs >10x = phase imbalance.
    """
    stats = {
        "adv_mean": advantages.mean(),
        "adv_std": advantages.std(unbiased=False),
    }
    # Per-phase stats
    stats.update(mean_std_per_mask(advantages, phases_flat == 0, "adv_roll"))
    stats.update(mean_std_per_mask(advantages, phases_flat != 0, "adv_score"))
    return stats


def compute_return_stats(episode_returns: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute return statistics per-episode.

    Healthy: high early (stochastic), stabilizes.
    Red flags: collapsing to tiny values = degenerate policy.
    """
    return {"ret_std": episode_returns.std(unbiased=False)}


def compute_critic_explained_variance(returns: torch.Tensor, v_ests: torch.Tensor) -> torch.Tensor:
    """Compute explained variance: EV = 1 - Var(G - V) / Var(G).

    Healthy: 0.3-0.7 and trending up.
    Red flags: stuck at 0 (not learning), <0 (harmful), wild swings (unstable).
    """
    with torch.no_grad():
        residuals = returns - v_ests.squeeze()
        return 1.0 - residuals.var(unbiased=False) / (returns.var(unbiased=False) + 1e-8)


def compute_entropy_stats(
    rolling_probs: torch.Tensor,
    scoring_probs: torch.Tensor,
    phases_flat: torch.Tensor,
    is_bernoulli: bool,
) -> dict[str, torch.Tensor]:
    """Compute entropy for both action heads.

    Healthy: roll ~2.0-2.4 early → ~1.0 later; score moderate, gradual decline.
    Red flags: roll <1.2 early = premature collapse; score near-zero = deterministic too soon.
    """
    # Rolling head: Bernoulli entropy per-die, sum to get per-step
    if is_bernoulli:
        roll_ent = bernoulli_entropy(rolling_probs).sum(dim=1)
    else:
        roll_ent = torch.distributions.Categorical(probs=rolling_probs).entropy()

    # Scoring head: categorical entropy
    score_ent = torch.distributions.Categorical(probs=scoring_probs).entropy()

    return {
        "entropy_roll": roll_ent[phases_flat == 0].mean(),
        "entropy_score": score_ent[phases_flat != 0].mean(),
    }


def compute_action_concentration(
    scoring_probs: torch.Tensor,
    rolling_probs: torch.Tensor,
    phases_flat: torch.Tensor,
    is_bernoulli: bool,
) -> dict[str, torch.Tensor]:
    """Compute top-k probability mass (early collapse detector).

    Healthy: top1 starts ~0.2-0.4, climbs slowly; top3 >0.6 early.
    Red flags: top1 >0.7 in first 10-20% of training = over-confident, collapsed.
    """
    result = {
        "score_top1": scoring_probs.max(dim=1).values.mean(),
        "score_top3": scoring_probs.topk(3, dim=1).values.sum(dim=1).mean(),
    }

    # Compute rolling concentration based on representation
    roll_mask = phases_flat == 0
    if roll_mask.any():
        if is_bernoulli:
            # For Bernoulli: each die is independent with prob p_i
            # Top1: average of max(p_i, 1-p_i) across all dice (how confident per die)
            # Top3: average of top-3 max(p_i, 1-p_i) (concentration in most confident dice)
            roll_probs_masked = rolling_probs[roll_mask]  # (n_roll_steps, 5)
            max_probs_per_die = torch.maximum(
                roll_probs_masked, 1 - roll_probs_masked
            )  # (n_roll_steps, 5)
            result["roll_top1"] = max_probs_per_die.mean()  # Average confidence per die
            result["roll_top3"] = max_probs_per_die.topk(
                3, dim=1
            ).values.mean()  # Average of top-3 most confident dice
        else:
            # For Categorical: standard top-1 and top-3 probability mass
            roll_probs_masked = rolling_probs[roll_mask]  # (n_roll_steps, 32)
            result["roll_top1"] = roll_probs_masked.max(dim=1).values.mean()
            result["roll_top3"] = roll_probs_masked.topk(3, dim=1).values.sum(dim=1).mean()
    else:
        result["roll_top1"] = torch.tensor(0.0)
        result["roll_top3"] = torch.tensor(0.0)

    return result


def compute_rolling_mask_diversity(
    rolling_actions_flat: torch.Tensor, phases_flat: torch.Tensor
) -> torch.Tensor | None:
    """Compute diversity of rolling mask patterns.

    Healthy: high and steady.
    Red flags: monotonic decline = converging to "always keep X" rut.
    """
    if rolling_actions_flat.numel() == 0:
        return None

    roll_mask = phases_flat == 0
    if not roll_mask.any():
        return None

    unique_patterns = torch.unique(rolling_actions_flat[roll_mask], dim=0).shape[0]
    total_patterns = roll_mask.sum()
    return torch.tensor(unique_patterns / total_patterns.clamp(min=1).item())


def compute_kl_divergence(
    rolling_probs: torch.Tensor,
    scoring_probs: torch.Tensor,
    phases_flat: torch.Tensor,
    prev_roll_p: torch.Tensor,
    prev_score_p: torch.Tensor,
) -> tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]:
    """Compute KL divergence to previous policy for both heads.

    Healthy: 0.002-0.02 (policy moving).
    Red flags: ≈0 = frozen; sudden spikes = too hot.
    """
    # KL divergence for Bernoulli rolling actions (product distribution)
    kl_ber_per_die = bernoulli_kl(rolling_probs, prev_roll_p)
    roll_mask = phases_flat == 0
    kl_roll = kl_ber_per_die.sum(dim=1)[roll_mask].mean() if roll_mask.any() else None

    # KL divergence for categorical scoring actions
    score_mask = phases_flat != 0
    if score_mask.any():
        eps = 1e-6
        p = scoring_probs[score_mask].clamp_min(eps)
        q = prev_score_p[score_mask].clamp_min(eps)
        kl_score = torch.sum(p * (p / q).log(), dim=1).mean()
    else:
        kl_score = None

    # Return stats and updated caches (detached)
    return (
        {"kl_roll": kl_roll, "kl_score": kl_score},
        rolling_probs.detach(),
        scoring_probs.detach(),
    )


def compute_phase_balance(phases_flat: torch.Tensor) -> torch.Tensor:
    """Compute fraction of rolling vs scoring steps.

    Healthy: ~0.67 (2 rolls + 1 score per turn).
    Red flags: extreme drift = bad logic.
    """
    return (phases_flat == 0).float().mean()


def compute_training_health_score(
    critic_ev_mean: float, entropy_roll_mean: float, entropy_score_mean: float
) -> float:
    """Compute overall training health score from critic and entropy metrics.

    Healthy: >0.4 overall.
    Components: critic_ev (0.3-0.7), entropy_roll (1.0-2.4), entropy_score (moderate).
    """
    critic_component = max(0, min(critic_ev_mean / 0.7, 1.0)) * 0.5
    roll_ent_component = max(0, min(entropy_roll_mean / 2.4, 1.0)) * 0.25
    score_ent_component = max(0, min(entropy_score_mean / 1.5, 1.0)) * 0.25
    return critic_component + roll_ent_component + score_ent_component
