#!/usr/bin/env python
import pytorch_lightning as lightning
import torch

from A_dice_maximizer.trainer import REINFORCEWithBaselineTrainer
from utilities.dummy_dataset import DummyDataset
from utilities.initialize import ConfigParam, finish, initialize
from utilities.return_calculators import MonteCarloReturnCalculator


def main() -> None:
    """Run training for Yahtzee dice maximization."""
    # Define configuration schema
    config_params = [
        ConfigParam("epochs", int, 50, "Number of training epochs"),
        ConfigParam(
            "episodes_per_batch",
            int,
            32,
            "Episodes per training batch",
            display_name="Episodes per batch",
        ),
        ConfigParam("learning_rate", float, 1e-3, "Learning rate", display_name="Learning rate"),
        ConfigParam("hidden_size", int, 64, "Hidden layer size", display_name="Hidden size"),
    ]

    # Initialize project with configuration
    wandb_run, config, logger = initialize(
        scenario_name="dice_maximizer",
        config_params=config_params,
        description="Yahtzee Dice Maximizer RL",
        logger_name="monte-carlo-training",
    )

    # Extract config values for easy access
    epochs = config["epochs"]
    episodes_per_batch = config["episodes_per_batch"]
    learning_rate = config["learning_rate"]
    hidden_size = config["hidden_size"]

    # Create return calculator and model
    return_calculator = MonteCarloReturnCalculator()
    model = REINFORCEWithBaselineTrainer(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        episodes_per_batch=episodes_per_batch,
        return_calculator=return_calculator,
    )

    # Create trainer
    trainer = lightning.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        accelerator="auto",  # Will use GPU if available
        devices="auto",
    )

    # Create dummy dataloader (required by Lightning but not used)
    dummy_dataset = DummyDataset(size=1000)
    dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1, num_workers=0)

    # Train
    trainer.fit(model, dataloader)

    finish(wandb_run)


if __name__ == "__main__":
    main()
