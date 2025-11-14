#!/usr/bin/env python
import logging
import os

import pytorch_lightning as lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from C_single_turn_score_maximizer import test_episode
from C_single_turn_score_maximizer.self_play_dataset import SelfPlayDataset
from C_single_turn_score_maximizer.trainer import SingleTurnScoreMaximizerREINFORCETrainer
from utilities.initialize import ConfigParam, finish, initialize
from utilities.return_calculators import MonteCarloReturnCalculator

log = logging.getLogger(__name__)

CKPT_DIR: str = "/opt/ml/checkpoints"  # SageMaker restores this from S3 on restart

# 1 game = 13 turns, 1 turn = 3 steps
TURNS_PER_GAME: int = 13


class InvalidBatchConfigurationError(ValueError):
    """Exception raised when batch configuration is invalid."""

    def __init__(self, games_per_batch: int, games_per_epoch: int) -> None:
        batches_per_epoch = games_per_epoch / games_per_batch
        super().__init__(
            f"games_per_batch ({games_per_batch}) must divide games_per_epoch ({games_per_epoch}) evenly. "
            f"Current configuration would result in {batches_per_epoch} batches per epoch."
        )


def main() -> None:
    """Run training or testing for single-turn Yahtzee score maximization."""
    # Define configuration schema
    config_params = [
        ConfigParam("mode", str, "train", "Mode to run (train or test)", choices=["train", "test"]),
        ConfigParam("epochs", int, 500, "Number of training epochs"),
        ConfigParam(
            "games_per_epoch",
            int,
            100,
            "Number of complete Yahtzee games per epoch",
            display_name="Games per epoch",
        ),
        ConfigParam(
            "games_per_batch",
            int,
            4,
            "Number of complete Yahtzee games per batch (must divide games_per_epoch evenly)",
            display_name="Games per batch",
        ),
        ConfigParam("learning_rate", float, 3e-4, "Learning rate", display_name="Learning rate"),
        ConfigParam("hidden_size", int, 384, "Hidden layer size", display_name="Hidden size"),
        ConfigParam(
            "num_hidden", int, 3, "Number of hidden layers", display_name="Num hidden layers"
        ),
        ConfigParam(
            "checkpoint_path",
            str,
            None,
            "Path to model checkpoint for evaluation",
            display_name="Checkpoint path",
        ),
        ConfigParam(
            "activation_function",
            str,
            "Swish",
            "Activation function to use in the model",
            choices=[
                "ReLU",
                "GELU",
                "CELU",
                "PReLU",
                "ELU",
                "Tanh",
                "LeakyReLU",
                "Softplus",
                "Softsign",
                "Mish",
                "Swish",
                "SeLU",
            ],
            display_name="Activation function",
        ),
        ConfigParam(
            "min_lr_ratio",
            float,
            0.1,
            "Ratio of minimum learning rate to initial learning rate (for cosine annealing)",
            display_name="Min LR ratio",
        ),
        ConfigParam(
            "gamma_max",
            float,
            1.0,
            "Discount factor for reward calculation (max, end)",
            display_name="Discount factor",
        ),
        ConfigParam(
            "gamma_min",
            float,
            0.9,
            "Discount factor for reward calculation (min, start)",
            display_name="Discount factor",
        ),
        ConfigParam(
            "dropout_rate",
            float,
            0.1,
            "Dropout rate for the model",
            display_name="Dropout rate",
        ),
        ConfigParam(
            "gradient_clip_val",
            float,
            0.5,
            "Gradient clipping value",
            display_name="Gradient clip value",
        ),
    ]

    # Initialize project with configuration
    wandb_run, config, logger = initialize(
        scenario_name="single_turn_score_maximizer",
        config_params=config_params,
        description="Yahtzee Single Turn Score Maximizer RL",
        logger_name="rl-training",
    )

    # Extract config values for easy access
    mode = config["mode"]
    epochs = config["epochs"]
    games_per_epoch = config["games_per_epoch"]
    games_per_batch = config["games_per_batch"]
    learning_rate = config["learning_rate"]
    hidden_size = config["hidden_size"]
    num_hidden = config["num_hidden"]
    checkpoint_path = config["checkpoint_path"]
    activation_function = config["activation_function"]
    min_lr_ratio = config["min_lr_ratio"]
    gamma_min = config["gamma_min"]
    gamma_max = config["gamma_max"]
    dropout_rate = config["dropout_rate"]
    gradient_clip_val = config["gradient_clip_val"]

    # Calculate dataset_size and batch_size from games parameters

    # Validate that games_per_batch divides games_per_epoch evenly
    if games_per_epoch % games_per_batch != 0:
        raise InvalidBatchConfigurationError(games_per_batch, games_per_epoch)

    dataset_size = games_per_epoch * TURNS_PER_GAME
    batch_size = games_per_batch * TURNS_PER_GAME
    batches_per_epoch = games_per_epoch // games_per_batch

    log.info(f"Configuration: {games_per_epoch} games/epoch, {games_per_batch} games/batch")
    log.info(f"Calculated: dataset_size={dataset_size} turns, batch_size={batch_size} turns")
    log.info(
        f"Training: {batches_per_epoch} batches/epoch, {batches_per_epoch * epochs} total updates"
    )

    if mode == "test":
        # Test mode
        test_episode.main(checkpoint_path=checkpoint_path)
    else:
        # Create return calculator and model
        return_calculator = MonteCarloReturnCalculator()
        model = SingleTurnScoreMaximizerREINFORCETrainer(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            return_calculator=return_calculator,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
            max_epochs=epochs,
            min_lr_ratio=min_lr_ratio,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

        run_scope = os.getenv("WANDB_RUN_ID") or "local-run"

        checkpoint_dir = os.path.join(CKPT_DIR, run_scope)
        last = os.path.join(checkpoint_dir, "last.ckpt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_cb = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,  # keep only last.ckpt (rolling)
            save_top_k=0,  # do NOT keep k-best; disables metric-based saves
            every_n_epochs=1,  # save at end of every training epoch
            save_on_train_epoch_end=True,
        )

        # Create trainer
        trainer = lightning.Trainer(
            max_epochs=epochs,
            logger=logger,
            enable_checkpointing=True,
            log_every_n_steps=1,
            accelerator="auto",  # Will use GPU if available
            devices="auto",
            check_val_every_n_epoch=1,  # Run validation every epoch
            callbacks=[ckpt_cb],
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

        # Create self-play dataset that collects episodes using the policy
        train_dataset = SelfPlayDataset(
            policy_net=model.policy_net,
            return_calculator=return_calculator,
            size=dataset_size,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=0
        )

        # Create validation dataset (just one batch)
        val_dataset = SelfPlayDataset(
            policy_net=model.policy_net,
            return_calculator=return_calculator,
            size=batch_size,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=0
        )

        # Train with validation
        trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_path=last if os.path.exists(last) else None,
        )

        # Run a test episode after training
        test_episode.main(model=model.policy_net, interactive=False)

    print("Process completed!")

    finish(wandb_run)


if __name__ == "__main__":
    main()
