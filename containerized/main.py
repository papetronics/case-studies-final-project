#!/usr/bin/env python
import os
import torch
import wandb
import argparse

import pytorch_lightning as L
from src.A_dice_maximizer.trainer import REINFORCEWithBaselineTrainer

from src.B_supervised_scorer.trainer import SupervisedScorerTrainer

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
    parser = argparse.ArgumentParser(description='Yahtzee RL')
    parser.add_argument('--scenario', type=str, default='supervised_scorer', choices=['dice_maximizer', 'supervised_scorer'],
                        help='Scenario to run')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--episodes-per-batch', type=int, default=32,
                        help='Episodes per training batch')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--num-hidden', type=int, default=1,
                        help='Number of hidden layers (for supervised scorer)')
    parser.add_argument('--dataset-size', type=int, default=10000,
                        help='Dataset size (for supervised scorer)')
    parser.add_argument('--log-dir', type=str, default='./logs')
    
    args = parser.parse_args()
    
    # Get hyperparameters from wandb config if available, otherwise use argparse
    if wandb_run is not None:
        epochs = wandb_run.config.get('epochs', args.epochs)
        episodes_per_batch = wandb_run.config.get('episodes_per_batch', args.episodes_per_batch)
        learning_rate = wandb_run.config.get('learning_rate', args.learning_rate)
        hidden_size = wandb_run.config.get('hidden_size', args.hidden_size)
        log_dir = wandb_run.config.get('log_dir', args.log_dir)
        num_hidden = wandb_run.config.get('num_hidden', args.num_hidden)
        dataset_size = wandb_run.config.get('dataset_size', args.dataset_size)
        use_wandb = True
    else:
        epochs = args.epochs
        episodes_per_batch = args.episodes_per_batch
        learning_rate = args.learning_rate
        hidden_size = args.hidden_size
        use_wandb = False
        num_hidden = args.num_hidden
        dataset_size = args.dataset_size
        log_dir = args.log_dir

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

    # Configure logger
    if use_wandb:
        logger = L.pytorch_lightning.loggers.WandbLogger(
            project=f"yahtzee-{args.scenario}",
            name="monte-carlo-training"
        )
    else:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        logger = L.pytorch_lightning.loggers.TensorBoardLogger(log_dir, name=f"yahtzee-reinforce-{args.scenario}")
    
    if args.scenario == 'dice_maximizer':
        dice_maximizer_main(
            epochs=epochs,
            episodes_per_batch=episodes_per_batch,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            logger=logger
        )
    elif args.scenario == 'supervised_scorer':
        supervised_scorer_main(
            epochs=epochs,
            episodes_per_batch=episodes_per_batch,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            logger=logger,
            num_hidden=num_hidden,
            dataset_size=dataset_size
        )
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")
    
    print("Training completed!")
    
    if wandb_run is not None:
        wandb_run.finish()

def supervised_scorer_main(
    epochs: int,
    episodes_per_batch: int,
    learning_rate: float,
    hidden_size: int,
    logger: L.loggers.Logger,
    num_hidden: int,
    dataset_size: int,
):
    # Create model
    model = SupervisedScorerTrainer(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        batch_size=episodes_per_batch,
        dataset_size=dataset_size,
        num_hidden=num_hidden
    )
        
    # Create trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        accelerator='auto',  # Will use GPU if available
        devices='auto',
    )
    
    # Train
    trainer.fit(model)


def dice_maximizer_main(
    epochs: int,
    episodes_per_batch: int,
    learning_rate: float,
    hidden_size: int,
    logger: L.loggers.Logger,
):
    # Create return calculator and model
    return_calculator = MonteCarloReturnCalculator()
    model = REINFORCEWithBaselineTrainer(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        episodes_per_batch=episodes_per_batch,
        return_calculator=return_calculator,
    )
        
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

if __name__ == "__main__":
    main()
