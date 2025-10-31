#!/usr/bin/env python
import os
import torch
import wandb

import gymnasium as gym
import yahtzee_gym
from dice_sum_maximizer import DiceSumMaximizer

def maybe_init_wandb():
    """
    Only initialize wandb if running inside a wandb launch agent or job.
    """
    is_launch = os.getenv("WANDB_JOB_NAME") or os.getenv("WANDB_RUN_ID")
    if is_launch:
        print("✅ Detected W&B launch agent context.")
        run = wandb.init()
    else:
        print("⚡ No W&B job context — skipping wandb.init() to avoid polluting real runs.")
        run = None
    return run

def train(model, env, num_steps=1000):
    """
    Training loop for the Yahtzee agent.
    """
    observation, info = env.reset()
    print(f"Obs: {observation}")

    for _ in range(num_steps):
        # Get probabilities from neural network
        probs = model.forward(observation)
        
        # Sample binary mask from probabilities
        action = torch.bernoulli(probs.cpu()).numpy().astype(int)
        
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Obs: {observation}")
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            observation, info = env.reset()
            print(f"Obs: {observation}")

def main():
    run = maybe_init_wandb()

    print("CUDA available:", torch.cuda.is_available())
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected!")

    # Initialize the neural network and environment
    model = DiceSumMaximizer(hidden_size=64)
    env = gym.make('Yahtzee-v0')
    
    # Run training
    train(model, env, num_steps=1000)
    
    env.close()

    if run is not None:
        # Safely access config only if W&B is initialized
        print("Sweep test param:", run.config.get("test"))
        run.finish()
    else:
        print("Sweep test param: [W&B not initialized]")

if __name__ == "__main__":
    main()
