#!/usr/bin/env python
import os
import torch
import wandb
import argparse

import gymnasium as gym
import yahtzee_gym
import pytorch_lightning as L
from src.A_dice_maximizer.trainer import REINFORCEWithBaselineTrainer
from src.return_calculators import MonteCarloReturnCalculator
from src.dummy_dataset import DummyDataset

def maybe_init_wandb():
    """
    Only initialize wandb if running inside a wandb launch agent or job.
    """
    is_launch = os.getenv("WANDB_JOB_NAME") or os.getenv("WANDB_RUN_ID")
    if is_launch:
        print("✅ Detected W&B launch agent context.")
        run = wandb.init()
        return run
    else:
        print("⚡ No W&B job context — skipping wandb.init() to avoid polluting real runs.")
        return None

def main():
    # Initialize wandb first to check for config
    wandb_run = maybe_init_wandb()
    
    # Set up argument parser with defaults
    parser = argparse.ArgumentParser(description='Yahtzee Monte Carlo Training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--episodes-per-batch', type=int, default=32,
                        help='Episodes per training batch')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden layer size')
    
    args = parser.parse_args()
    
    # Get hyperparameters from wandb config if available, otherwise use argparse
    if wandb_run is not None:
        epochs = wandb_run.config.get('epochs', args.epochs)
        episodes_per_batch = wandb_run.config.get('episodes_per_batch', args.episodes_per_batch)
        learning_rate = wandb_run.config.get('learning_rate', args.learning_rate)
        hidden_size = wandb_run.config.get('hidden_size', args.hidden_size)
        use_wandb = True
    else:
        epochs = args.epochs
        episodes_per_batch = args.episodes_per_batch
        learning_rate = args.learning_rate
        hidden_size = args.hidden_size
        use_wandb = False

    print("CUDA available:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected!")

    print(f"\n=== Starting Monte Carlo Training ===")
    print(f"Epochs: {epochs}")
    print(f"Episodes per batch: {episodes_per_batch}")
    print(f"Learning rate: {learning_rate}")
    print(f"Hidden size: {hidden_size}")
    
    # Create return calculator and model
    return_calculator = MonteCarloReturnCalculator()
    model = REINFORCEWithBaselineTrainer(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        episodes_per_batch=episodes_per_batch,
        return_calculator=return_calculator,
    )
    
    # Configure logger
    if use_wandb:
        logger = L.pytorch_lightning.loggers.WandbLogger(
            project="yahtzee-mc",
            name="monte-carlo-training"
        )
    else:
        # Ensure log directory exists
        log_dir = "/workspace/logs"
        os.makedirs(log_dir, exist_ok=True)
        logger = L.pytorch_lightning.loggers.TensorBoardLogger(log_dir, name="yahtzee-reinforce")
        
    # Create trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        accelerator='auto',  # Will use GPU if available
        devices='auto',
    )
    
    # Create dummy dataloader (required by Lightning but not used)
    dummy_dataset = DummyDataset(size=1000)
    dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1, num_workers=0)
    
    # Train
    trainer.fit(model, dataloader)
    
    print("Training completed!")
    
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
