#!/usr/bin/env python
import os
import torch
import wandb
import argparse

import pytorch_lightning as L
from src.A_dice_maximizer.trainer import REINFORCEWithBaselineTrainer
from src.B_supervised_scorer.trainer import SupervisedScorerTrainer
from src.C_single_turn_score_maximizer.trainer import SingleTurnScoreMaximizerREINFORCETrainer

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
    
    # Set up argument Parser with defaults
    parser = argparse.ArgumentParser(description='Yahtzee RL')
    parser.add_argument('--scenario', type=str, default='supervised_scorer',
                        choices=['dice_maximizer', 'supervised_scorer', 'single_turn_score_maximizer', 'test_single_turn_rl'],
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
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to model checkpoint for evaluation') 
    parser.add_argument('--activation-function', type=str, default='GELU',
                        choices=['ReLU', 'GELU', 'CELU', 'PReLU', 'ELU', 'Tanh', 
                                'LeakyReLU', 'Softplus', 'Softsign', 
                                'Mish', 'Swish', 'SeLU'],
                        help='Activation function to use in the model')

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
        activation_function = wandb_run.config.get('activation_function', args.activation_function)
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
        activation_function = args.activation_function

    print("\n=== Hyperparameters ===")
    print(f"Scenario: {args.scenario}")
    print(f"Epochs: {epochs}")
    print(f"Episodes per batch: {episodes_per_batch}")
    print(f"Learning rate: {learning_rate}")
    print(f"Hidden size: {hidden_size}")
    print(f"Num hidden layers: {num_hidden}")
    print(f"Dataset size: {dataset_size}")
    print(f"Activation function: {activation_function}")

    print("\n=== System Info ===")
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
            name="monte-carlo-training",
            log_model=True
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
    elif args.scenario == 'single_turn_score_maximizer':
        single_turn_score_maximizer_main(
            epochs=epochs,
            episodes_per_batch=episodes_per_batch,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            logger=logger,
            num_hidden=num_hidden,
            dropout_rate=0.1,
            dataset_size=dataset_size,
            activation_function=activation_function
        )
    elif args.scenario == 'test_single_turn_rl':
        from src.C_single_turn_score_maximizer.test_episode import main as test_episode_main
        test_episode_main(checkpoint_path=args.checkpoint_path)
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")
    
    print("Training completed!")
    
    if wandb_run is not None:
        wandb_run.finish()

def single_turn_score_maximizer_main(
    epochs: int,
    episodes_per_batch: int,
    learning_rate: float,
    hidden_size: int,
    logger: L.loggers.Logger,
    num_hidden: int,
    dropout_rate: float,
    dataset_size: int,
    activation_function: str,
):
    # Create return calculator and model
    return_calculator = MonteCarloReturnCalculator()
    model = SingleTurnScoreMaximizerREINFORCETrainer(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        episodes_per_batch=episodes_per_batch,
        return_calculator=return_calculator,
        num_hidden=num_hidden,
        dropout_rate=dropout_rate,
        activation_function=activation_function
    )
        
    # Create trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        accelerator='auto',  # Will use GPU if available
        devices='auto',
        check_val_every_n_epoch=1,  # Run validation every epoch
        #num_sanity_val_steps=0,  # Disable sanity checking
    )
    
    # Create dummy dataloader (required by Lightning but not used)
    dummy_dataset = DummyDataset(size=dataset_size//episodes_per_batch)
    train_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1, num_workers=15)
    
    # Create dummy validation dataloader
    val_dummy_dataset = DummyDataset(size=1)  # Just one batch for validation
    val_dataloader = torch.utils.data.DataLoader(val_dummy_dataset, batch_size=1, num_workers=15)
    
    # Train with validation
    trainer.fit(model, train_dataloader, val_dataloader)

    from src.C_single_turn_score_maximizer.test_episode import main as test_episode_main
    test_episode_main(model=model.policy_net, interactive=False)

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
