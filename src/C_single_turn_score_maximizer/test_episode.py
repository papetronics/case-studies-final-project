import gymnasium as gym
import torch
import os
import sys
import src.environment.full_yahtzee_env
from src.C_single_turn_score_maximizer.model import TurnScoreMaximizer
from src.C_single_turn_score_maximizer.trainer import SingleTurnScoreMaximizerREINFORCETrainer
from src.utilities.return_calculators import ReturnCalculator, MonteCarloReturnCalculator
from src.utilities.scoring_helper import ScoreCategory, get_all_scores

def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def wait_for_enter():
    """Wait for user to press enter"""
    input("\nPress Enter to continue...")

def main(
    checkpoint_path: str | None = None,
    model: TurnScoreMaximizer | None = None,
    interactive: bool = True,
):
    env = gym.make("FullYahtzee-v1")

    # load the model from checkpoint path
    if checkpoint_path is not None:
        # Load the Lightning trainer from checkpoint
        trainer = SingleTurnScoreMaximizerREINFORCETrainer.load_from_checkpoint(checkpoint_path)
        model = trainer.policy_net
    else:
        model = model if model else TurnScoreMaximizer()
    
    model.eval()
    run_episode(env, model, interactive)

def run_episode(
    env: gym.Env,
    model: TurnScoreMaximizer,
    interactive: bool = True,
):
    obs, _ = env.reset()
    original_dice = obs['dice'].copy()
    kept_dice = []

    clear_screen()
    print_game_state(obs, original_dice, kept_dice)
    if interactive:
        wait_for_enter()

    total_reward = 0.0

    with torch.no_grad():
        while True:
            actions, _ = model.sample_observation(obs)
            hold_action_tensor, scoring_action_tensor = actions

            action = {
                'hold_mask': hold_action_tensor.cpu().numpy().astype(bool),
                'score_category': scoring_action_tensor.cpu().item()
            }

            # Track dice state for display
            if obs['phase'] == 0:
                # Rolling phase - update kept dice
                kept_dice = obs['dice'][~hold_action_tensor.cpu().numpy().astype(bool)].tolist()
                print_action_description(obs, hold_action_tensor, None)
            else:
                # Scoring phase
                print_action_description(obs, None, scoring_action_tensor)

            if interactive:
                wait_for_enter()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)

            # Update original dice for new turn if we just scored
            if obs['rolls_used'] == 0 and obs['phase'] == 0:
                original_dice = obs['dice'].copy()
                kept_dice = []

            clear_screen()
            print_game_state(obs, original_dice, kept_dice)
            
            reward_value = float(reward)
            if reward_value > 0:
                print(f"\n🎯 Scored: {int(reward_value)} points!")

            if done:
                print(f"\n🏁 Game Over! Total Score: {int(total_reward)}")
                break
            if truncated:
                print(f"\n⏰ Game Truncated! Total Score: {int(total_reward)}")
                break

            if interactive:
                wait_for_enter()


def print_scorecard(observation: dict):
    """Print a pretty scorecard with current scores"""
    print("=" * 50)
    print("                   SCORECARD")
    print("=" * 50)
    
    # Upper section
    print("UPPER SECTION:")
    print("-" * 25)
    upper_total = 0
    for i in range(6):
        label = ScoreCategory.LABELS[i]
        if observation['score_sheet_available_mask'][i] == 0:  # Filled
            score = observation['score_sheet'][i]
            upper_total += score
            print(f"{label:15s}: [{score:2d}]")
        else:  # Available
            print(f"{label:15s}: [  ]")
    
    print("-" * 25)
    print(f"Upper Total     : [{upper_total:2d}]")
    bonus = 35 if upper_total >= 63 else 0
    print(f"Bonus (63+)     : [{bonus:2d}]")
    print(f"Upper w/Bonus   : [{upper_total + bonus:2d}]")
    
    # Lower section
    print("\nLOWER SECTION:")
    print("-" * 25)
    lower_total = 0
    for i in range(6, 13):
        label = ScoreCategory.LABELS[i]
        if observation['score_sheet_available_mask'][i] == 0:  # Filled
            score = observation['score_sheet'][i]
            lower_total += score
            print(f"{label:15s}: [{score:2d}]")
        else:  # Available
            print(f"{label:15s}: [  ]")
    
    print("-" * 25)
    print(f"Lower Total     : [{lower_total:2d}]")
    
    grand_total = upper_total + bonus + lower_total
    print("-" * 25)
    print(f"GRAND TOTAL     : [{grand_total:2d}]")
    print("=" * 50)

def print_dice_state(observation: dict, original_dice: list, kept_dice: list):
    """Print current dice state"""
    print("\nDICE:")
    print("-" * 20)
    
    if observation['rolls_used'] > 0:
        print(f"Original: {' '.join(map(str, original_dice))}")
        if kept_dice:
            print(f"Kept    : {' '.join(map(str, kept_dice))}")
        else:
            print("Kept    : (none)")
    
    print(f"Current : {' '.join(map(str, observation['dice']))}")
    print(f"Rolls used: {observation['rolls_used']}/2")
    
    if observation['phase'] == 0:
        print("Phase: Rolling")
    else:
        print("Phase: Scoring")

def print_available_scores(observation: dict):
    """Print available scoring categories with potential scores"""
    if observation['phase'] == 1:  # Only show in scoring phase
        possible_scores, _ = get_all_scores(observation['dice'], observation['score_sheet_available_mask'])
        
        available_categories = []
        for i, available in enumerate(observation['score_sheet_available_mask']):
            if available:
                available_categories.append((ScoreCategory.LABELS[i], possible_scores[i]))
        
        if available_categories:
            print("\nAVAILABLE SCORING OPTIONS:")
            print("-" * 30)
            for label, score in available_categories:
                print(f"{label:15s}: {score:2d} points")

def print_game_state(observation: dict, original_dice: list, kept_dice: list):
    """Print the complete game state"""
    print_scorecard(observation)
    print_dice_state(observation, original_dice, kept_dice)
    print_available_scores(observation)

def print_action_description(observation: dict, hold_action_tensor: torch.Tensor | None = None, scoring_action_tensor: torch.Tensor | None = None):
    """Print what action the model is taking"""
    print("\n" + "=" * 50)
    if hold_action_tensor is not None:
        # Rolling action
        hold_mask = hold_action_tensor.cpu().numpy().astype(bool)
        reroll_dice = []
        keep_dice = []
        
        for i, die_value in enumerate(observation['dice']):
            if hold_mask[i]:  # This die will be re-rolled
                reroll_dice.append(f"Die {i+1}({die_value})")
            else:  # This die will be kept
                keep_dice.append(f"Die {i+1}({die_value})")
        
        print("🎲 MODEL ACTION: Rolling dice")
        if keep_dice:
            print(f"   Keeping: {', '.join(keep_dice)}")
        if reroll_dice:
            print(f"   Re-rolling: {', '.join(reroll_dice)}")
    
    elif scoring_action_tensor is not None:
        # Scoring action
        category = scoring_action_tensor.cpu().item()
        category_name = ScoreCategory.LABELS[int(category)]
        possible_scores, _ = get_all_scores(observation['dice'], observation['score_sheet_available_mask'])
        score = possible_scores[int(category)]
        
        print("🎯 MODEL ACTION: Scoring")
        print(f"   Category: {category_name}")
        print(f"   Points: {score}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()