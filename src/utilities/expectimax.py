from collections.abc import Generator
from itertools import product
from typing import cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .scoring_helper import get_all_scores_with_target

DiceType = tuple[int, int, int, int, int]
ActionType = tuple[int, int, int, int, int]


def all_dice_states() -> set[DiceType]:
    """Generate all possible dice states (5 dice with values 1-6)."""
    possible_dice = set()

    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                for l in range(1, 7):  # noqa: E741
                    for m in range(1, 7):
                        dice: list[int] = [i, j, k, l, m]
                        dice.sort()
                        possible_dice.add(cast("DiceType", tuple(dice)))
    return possible_dice


def all_dice_masks() -> Generator[ActionType, None, None]:
    """Generate all possible dice hold masks (5 dice, each can be held or not)."""
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                for l in [0, 1]:  # noqa: E741
                    for m in [0, 1]:
                        yield (i, j, k, l, m)


## prebuild a probability transition model dict[dice, dict[action, dict[next_dice, probability]]]
def build_transitions(
    dice_states: set[DiceType], dice_masks: list[ActionType]
) -> dict[DiceType, dict[ActionType, dict[DiceType, float]]]:
    """Build transition probabilities for all (dice, action) pairs.

    For each dice state and hold action, compute the probability of transitioning
    to each possible next state by considering all combinations of rerolled dice.
    """
    transitions: dict[DiceType, dict[ActionType, dict[DiceType, float]]] = {}

    for dice in dice_states:
        transitions[dice] = {}

        for action in dice_masks:
            # Count occurrences of each next state
            next_state_counts: dict[DiceType, int] = {}

            # Find which dice positions are NOT held (will be rerolled)
            non_held_positions = [i for i in range(5) if action[i] == 0]
            num_non_held = len(non_held_positions)

            if num_non_held == 0:
                # All dice are held, stay in same state
                next_state_counts[dice] = 1
            else:
                # Generate all possible combinations for non-held dice
                # Each non-held die can be 1-6
                for new_values in product(range(1, 7), repeat=num_non_held):
                    # Create next state by keeping held dice and replacing non-held
                    next_dice_list = list(dice)
                    for pos_idx, die_pos in enumerate(non_held_positions):
                        next_dice_list[die_pos] = new_values[pos_idx]

                    # Sort to get canonical representation
                    next_dice_list.sort()
                    next_dice_tuple: DiceType = (
                        next_dice_list[0],
                        next_dice_list[1],
                        next_dice_list[2],
                        next_dice_list[3],
                        next_dice_list[4],
                    )

                    # Increment count for this next state
                    if next_dice_tuple in next_state_counts:
                        next_state_counts[next_dice_tuple] += 1
                    else:
                        next_state_counts[next_dice_tuple] = 1

            # Normalize counts to probabilities
            total_count = sum(next_state_counts.values())
            transitions[dice][action] = {
                next_state: count / total_count for next_state, count in next_state_counts.items()
            }

    return transitions


def build_r2_cache(
    dice_states: set[DiceType], open_scores: NDArray[np.int_]
) -> dict[DiceType, tuple[int, int]]:
    """Build a cache of all possible dice states to their best score and category."""
    r2_cache: dict[DiceType, tuple[int, int]] = {}
    for dice in dice_states:
        if dice not in r2_cache:
            dice_array = np.array(dice, dtype=np.int_)

            scores, _, _ = get_all_scores_with_target(
                dice_array, open_scores=open_scores
            )  # todo implement yahtzee check

            # Mask closed categories by setting them to -1 for argmax selection
            scores_masked = scores.copy()
            scores_masked[open_scores == 0] = -1

            best_category: int = int(scores_masked.argmax())
            best_score: int = scores[best_category]
            r2_cache[dice] = (best_category, best_score)

    return r2_cache


def build_r1_cache(
    dice_states: set[DiceType],
    dice_masks: list[ActionType],
    transitions: dict[DiceType, dict[ActionType, dict[DiceType, float]]],
    r2_cache: dict[DiceType, tuple[int, int]],
) -> dict[DiceType, tuple[ActionType, float]]:
    """Build a cache for all possible dice states at depth 1."""
    r1_cache: dict[DiceType, tuple[ActionType, float]] = {}
    for dice in dice_states:
        if dice in r1_cache:
            continue

        action_values: NDArray[np.float32] = np.zeros(len(dice_masks), dtype=np.float32)

        for i, action in enumerate(dice_masks):
            # Use transition probabilities to compute expected value
            expected_value = 0.0
            transition_probs = transitions[dice][action]

            for next_dice, probability in transition_probs.items():
                _, best_score = r2_cache[next_dice]
                expected_value += probability * best_score

            action_values[i] = expected_value

        # max
        best_action_i = int(action_values.argmax())
        best_action: ActionType = dice_masks[best_action_i]
        best_action_value: float = action_values[best_action_i]
        r1_cache[dice] = (best_action, best_action_value)

    return r1_cache


def build_r0_cache(
    dice: DiceType,
    dice_masks: list[ActionType],
    transitions: dict[DiceType, dict[ActionType, dict[DiceType, float]]],
    r1_cache: dict[DiceType, tuple[ActionType, float]],
) -> tuple[ActionType, float]:
    """Build the best action and value for a specific dice state at depth 0."""
    action_values: NDArray[np.float32] = np.zeros(len(dice_masks), dtype=np.float32)

    for i, action in enumerate(dice_masks):
        # Use transition probabilities to compute expected value
        expected_value = 0.0
        transition_probs = transitions[dice][action]

        for next_dice, probability in transition_probs.items():
            _, best_value = r1_cache[next_dice]
            expected_value += probability * best_value

        action_values[i] = expected_value

    # max
    best_action_i = int(action_values.argmax())
    best_action: ActionType = dice_masks[best_action_i]
    best_action_value: float = action_values[best_action_i]

    return (best_action, best_action_value)


def expectimax_dp(
    dice: DiceType,
    roll_count: int,
    r0_result: tuple[ActionType, float] | None,
    r1_cache: dict[DiceType, tuple[ActionType, float]],
    r2_cache: dict[DiceType, tuple[int, int]],
) -> tuple[ActionType | int, float]:
    """Perform expectimax search using precomputed caches."""
    # roll_count: 0, 1, 2
    # depth = 0: first roll
    # depth = 1: second roll
    # depth = 2: scoring (terminal, return max possible score)

    sorted_dice = sorted(dice)
    state_key: DiceType = cast("DiceType", tuple(sorted_dice))

    if roll_count == 0:
        if r0_result is None:
            raise RuntimeError("r0_result not initialized")  # noqa: TRY003
        best_action, expected_value = r0_result
        return best_action, expected_value
    elif roll_count == 1:
        best_action, expected_value = r1_cache[state_key]
        return best_action, expected_value
    elif roll_count == 2:  # noqa: PLR2004
        best_category, best_score = r2_cache[state_key]
        return best_category, float(best_score)
    else:
        raise ValueError("roll_count must be 0, 1, or 2.")  # noqa: TRY003


class ExpectimaxAgent:
    """Agent that uses expectimax search with precomputed transitions."""

    def __init__(self) -> None:
        """Initialize the agent by precomputing dice states, masks, and transitions."""
        self.dice_states = all_dice_states()
        self.dice_masks = list(all_dice_masks())
        self.transitions = build_transitions(self.dice_states, self.dice_masks)

        # Global cache for r1 and r2 caches keyed by scoresheet mask
        # This allows reuse across multiple games for common scoresheet states
        self.scoresheet_cache: dict[
            tuple[int, ...],
            tuple[dict[DiceType, tuple[ActionType, float]], dict[DiceType, tuple[int, int]]],
        ] = {}

        self.r0_result: tuple[ActionType, float] | None = None
        self.r1_cache: dict[DiceType, tuple[ActionType, float]] | None = None
        self.r2_cache: dict[DiceType, tuple[int, int]] | None = None

    def init_round(self, dice: DiceType, scoresheet: NDArray[np.int_]) -> None:
        """Initialize caches for a new round based on the current dice and scoresheet.

        Args:
            dice: Current dice state at the start of the round
            scoresheet: Array indicating which categories are still open (1) or filled (0)
        """
        # Convert scoresheet to tuple for use as dict key
        scoresheet_key = tuple(scoresheet.tolist())

        # Check if we've already computed r1 and r2 caches for this scoresheet
        if scoresheet_key in self.scoresheet_cache:
            self.r1_cache, self.r2_cache = self.scoresheet_cache[scoresheet_key]
        else:
            # Compute and cache r1 and r2 for this scoresheet
            self.r2_cache = build_r2_cache(self.dice_states, scoresheet)
            self.r1_cache = build_r1_cache(
                self.dice_states, self.dice_masks, self.transitions, self.r2_cache
            )
            self.scoresheet_cache[scoresheet_key] = (self.r1_cache, self.r2_cache)

        # Always compute r0 for the specific dice state
        self.r0_result = build_r0_cache(dice, self.dice_masks, self.transitions, self.r1_cache)

    def select_action(self, dice: DiceType, roll_count: int) -> tuple[ActionType | int, float]:
        """Select the best action for the current dice state and roll count.

        Args:
            dice: Current dice state
            roll_count: Number of rolls made so far (0, 1, or 2)

        Returns
        -------
            Tuple of (best_action, expected_value)
            - If roll_count < 2: best_action is an ActionType (hold mask)
            - If roll_count == 2: best_action is an int (category to score)
        """
        if self.r0_result is None or self.r1_cache is None or self.r2_cache is None:
            raise RuntimeError("Caches not initialized. Call init_round() first.")  # noqa: TRY003

        return expectimax_dp(dice, roll_count, self.r0_result, self.r1_cache, self.r2_cache)


if __name__ == "__main__":
    import gymnasium as gym  # noqa: I001
    from environments.full_yahtzee_env import YahtzeeEnv, RollingAction, ScoringAction

    agent = ExpectimaxAgent()
    env: YahtzeeEnv = cast("YahtzeeEnv", gym.make("FullYahtzee-v1"))

    num_games = 1000
    all_rewards = []

    for _ in tqdm(range(num_games)):
        obs, _ = env.reset()
        dice_tuple = tuple(sorted(obs["dice"]))
        agent.init_round(dice_tuple, obs["score_sheet_available_mask"])

        total_reward = 0.0
        terminated = False

        while not terminated:
            dice_tuple = tuple(sorted(obs["dice"]))
            action, exp_value = agent.select_action(dice_tuple, obs["rolls_used"])

            if isinstance(action, tuple):
                hold_mask: np.ndarray = np.array(action, dtype=np.bool)
                r_act: RollingAction = {"hold_mask": hold_mask}
                obs, reward, terminated, _, _ = env.step(r_act)
            else:
                s_act: ScoringAction = {"score_category": action}
                obs, reward, terminated, _, _ = env.step(s_act)

            total_reward += float(reward)

            if not terminated and obs["phase"] == 0 and obs["rolls_used"] == 0:
                dice_tuple = tuple(sorted(obs["dice"]))
                agent.init_round(dice_tuple, obs["score_sheet_available_mask"])

        all_rewards.append(total_reward)

    rewards_array = np.array(all_rewards)
    print(f"Mean score: {rewards_array.mean():.2f}")
    print(f"Std dev: {rewards_array.std():.2f}")
