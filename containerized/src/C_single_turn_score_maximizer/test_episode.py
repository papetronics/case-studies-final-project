import gymnasium as gym
import torch
import src.C_single_turn_score_maximizer.yahtzee_env
from src.C_single_turn_score_maximizer.model import TurnScoreMaximizer
from src.C_single_turn_score_maximizer.trainer import SingleTurnScoreMaximizerREINFORCETrainer
from src.return_calculators import ReturnCalculator, MonteCarloReturnCalculator
from src.scoring_helper import ScoreCategory, get_all_scores

def main(
    checkpoint_path: str | None = None,
    model: TurnScoreMaximizer | None = None
):
    env = gym.make("Yahtzee-v1")

    # load the model from checkpoint path
    if checkpoint_path is not None:
        # Load the Lightning trainer from checkpoint
        trainer = SingleTurnScoreMaximizerREINFORCETrainer.load_from_checkpoint(checkpoint_path)
        model = trainer.policy_net
    else:
        model = model if model else TurnScoreMaximizer()
    
    model.eval()
    run_episode(env, model)

def run_episode(
    env: gym.Env,
    model: TurnScoreMaximizer
):
    obs, _ = env.reset()

    print_observation(obs)

    total_reward = 0.0

    with torch.no_grad():
        while True:
            actions, _ = model.sample_observation(obs)
            hold_action_tensor, scoring_action_tensor = actions

            action = {
                'hold_mask': hold_action_tensor.cpu().numpy().astype(bool),
                'score_category': scoring_action_tensor.cpu().item()
            }

            if obs['phase'] == 0:
                print_rolling_action(obs, hold_action_tensor)
            else:
                print_scoring_action(scoring_action_tensor)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)

            print_observation(obs)
            print(f"Reward: {reward}\n")

            if done:
                print(f"Episode finished. Total Reward: {total_reward}")
                break
            if truncated:
                print(f"Episode truncated. Total Reward: {total_reward}")
                break


def print_observation(
    observation: dict
):
    print("Observation:")
    print(f"  Dice: {observation['dice']}")
    print(f"  Rolls Used: {observation['rolls_used']}")
    print(f"  Phase: {observation['phase']}")

    possible_score_values, _ = get_all_scores(observation['dice'], observation['score_sheet_available_mask'])

    # iterate over score sheet available mask, if 1, print the friendly label
    print("  Score Sheet Available Categories:")
    for i, available in enumerate(observation['score_sheet_available_mask']):
        if available:
            print(f"    - {ScoreCategory.LABELS[i]} ({possible_score_values[i]})")

    print(f"  Score Sheet Available Mask: {observation['score_sheet_available_mask']}")

def print_rolling_action(
    observation: dict,
    hold_action_tensor: torch.Tensor
):
    hold_mask = hold_action_tensor.cpu().numpy().astype(bool)
    print("Rolling Action:")
    for i in range(len(observation['dice'])):
        action_str = "Re-roll" if hold_mask[i] else "Hold"
        print(f"  Die {i+1} (Value: {observation['dice'][i]}): {action_str}")

def print_scoring_action(
    scoring_action_tensor: torch.Tensor
):
    score_category = scoring_action_tensor.cpu().item()
    print("Scoring Action:")
    print(f"  Selected Category: {ScoreCategory.LABELS[int(score_category)]}")