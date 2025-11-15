"""Diagnostic metric computation utilities for RL training monitoring."""

import torch


def mean_std_per_mask(
    values: torch.Tensor, mask: torch.Tensor, prefix: str
) -> dict[str, torch.Tensor]:
    """Compute mean and std for values within a mask.

    Parameters
    ----------
    values : torch.Tensor
        Tensor of values to compute statistics on
    mask : torch.Tensor
        Boolean mask to select subset of values
    prefix : str
        Prefix for dictionary keys (e.g., "adv_roll" -> "adv_roll_mean", "adv_roll_std")

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "{prefix}_mean" and "{prefix}_std" keys
    """
    masked_values = values[mask]
    return {
        f"{prefix}_mean": masked_values.mean(),
        f"{prefix}_std": masked_values.std(unbiased=False),
    }


def bernoulli_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute Bernoulli entropy: -p*log(p) - (1-p)*log(1-p).

    Uses torch.xlogy for numerical stability (handles p=0 case).

    Parameters
    ----------
    probs : torch.Tensor
        Bernoulli probabilities in [0, 1]

    Returns
    -------
    torch.Tensor
        Entropy values (same shape as input)
    """
    p = probs.clamp(1e-6, 1 - 1e-6)
    return -torch.xlogy(p, p) - torch.xlogy(1 - p, 1 - p)


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence for Bernoulli: KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q)).

    Uses torch.xlogy for numerical stability.

    Parameters
    ----------
    p : torch.Tensor
        Current policy probabilities
    q : torch.Tensor
        Reference policy probabilities

    Returns
    -------
    torch.Tensor
        KL divergence values (same shape as input)
    """
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

    Parameters
    ----------
    advantages : torch.Tensor
        Advantage estimates for all steps
    phases_flat : torch.Tensor
        Phase indicators (0=rolling, non-zero=scoring)

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with overall and per-phase mean/std statistics
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

    Parameters
    ----------
    episode_returns : torch.Tensor
        Episode return values

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "ret_std" key
    """
    return {"ret_std": episode_returns.std(unbiased=False)}


def compute_critic_explained_variance(returns: torch.Tensor, v_ests: torch.Tensor) -> torch.Tensor:
    """Compute explained variance: EV = 1 - Var(G - V) / Var(G).

    Measures how well the value function predicts actual returns.
    - EV = 1.0: Perfect predictions
    - EV = 0.0: As good as constant baseline (mean)
    - EV < 0.0: Worse than mean (harmful critic)

    Parameters
    ----------
    returns : torch.Tensor
        Actual returns (targets)
    v_ests : torch.Tensor
        Value function estimates

    Returns
    -------
    torch.Tensor
        Scalar explained variance
    """
    with torch.no_grad():
        residuals = returns - v_ests.squeeze()
        return 1.0 - residuals.var(unbiased=False) / (returns.var(unbiased=False) + 1e-8)


def compute_entropy_stats(
    rolling_probs: torch.Tensor, scoring_probs: torch.Tensor, phases_flat: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute entropy for both action heads.

    Parameters
    ----------
    rolling_probs : torch.Tensor
        Bernoulli probabilities for rolling actions (batch_size, num_dice)
    scoring_probs : torch.Tensor
        Categorical probabilities for scoring actions (batch_size, num_categories)
    phases_flat : torch.Tensor
        Phase indicators (0=rolling, non-zero=scoring)

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "entropy_roll" and "entropy_score" keys
    """
    # Rolling head: Bernoulli entropy per-die, sum to get per-step
    roll_ent = bernoulli_entropy(rolling_probs).sum(dim=1)

    # Scoring head: categorical entropy
    score_ent = torch.distributions.Categorical(probs=scoring_probs).entropy()

    return {
        "entropy_roll": roll_ent[phases_flat == 0].mean(),
        "entropy_score": score_ent[phases_flat != 0].mean(),
    }


def compute_action_concentration(scoring_probs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute top-k probability mass (early collapse detector).

    Parameters
    ----------
    scoring_probs : torch.Tensor
        Categorical probabilities for scoring actions

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "score_top1" and "score_top3" keys
    """
    return {
        "score_top1": scoring_probs.max(dim=1).values.mean(),
        "score_top3": scoring_probs.topk(3, dim=1).values.sum(dim=1).mean(),
    }


def compute_rolling_mask_diversity(
    rolling_actions_flat: torch.Tensor, phases_flat: torch.Tensor
) -> torch.Tensor | None:
    """Compute diversity of rolling mask patterns.

    Parameters
    ----------
    rolling_actions_flat : torch.Tensor
        Rolling action masks
    phases_flat : torch.Tensor
        Phase indicators (0=rolling, non-zero=scoring)

    Returns
    -------
    torch.Tensor | None
        Fraction of unique patterns, or None if no rolling actions
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

    Parameters
    ----------
    rolling_probs : torch.Tensor
        Current rolling probabilities
    scoring_probs : torch.Tensor
        Current scoring probabilities
    phases_flat : torch.Tensor
        Phase indicators (0=rolling, non-zero=scoring)
    prev_roll_p : torch.Tensor
        Previous rolling probabilities
    prev_score_p : torch.Tensor
        Previous scoring probabilities

    Returns
    -------
    tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]
        KL stats dict, updated prev_roll_p, updated prev_score_p
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

    Parameters
    ----------
    phases_flat : torch.Tensor
        Phase indicators (0=rolling, non-zero=scoring)

    Returns
    -------
    torch.Tensor
        Fraction of rolling steps (should be ~0.67 for 2 rolls + 1 score)
    """
    return (phases_flat == 0).float().mean()
