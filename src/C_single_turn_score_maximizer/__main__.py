#!/usr/bin/env python
import logging
import os

import pytorch_lightning as lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from C_single_turn_score_maximizer import test_episode
from C_single_turn_score_maximizer.self_play_dataset import SelfPlayDataset
from C_single_turn_score_maximizer.trainer import SingleTurnScoreMaximizerREINFORCETrainer
from utilities.dummy_dataset import DummyDataset
from utilities.initialize import ConfigParam, finish, initialize
from utilities.return_calculators import MonteCarloReturnCalculator

log = logging.getLogger(__name__)

CKPT_DIR: str = "/opt/ml/checkpoints"  # SageMaker restores this from S3 on restart

# 1 game = 13 turns, 1 turn = 3 steps
TURNS_PER_GAME: int = 13


class InvalidTrainingConfigurationError(ValueError):
    """Exception raised when training configuration is invalid."""


class GamesPerEpochNotDivisibleError(InvalidTrainingConfigurationError):
    """Exception raised when games_per_epoch doesn't divide evenly by games_per_batch."""

    def __init__(
        self, total_train_games: int, epochs: int, games_per_epoch: int, games_per_batch: int
    ) -> None:
        super().__init__(
            f"total_train_games ({total_train_games}) // epochs ({epochs}) = {games_per_epoch} games/epoch, "
            f"which does not divide evenly by games_per_batch ({games_per_batch}). "
            f"Please adjust parameters so that (total_train_games // epochs) % games_per_batch == 0."
        )


class BatchSizeTooLargeError(InvalidTrainingConfigurationError):
    """Exception raised when games_per_batch is larger than games_per_epoch."""

    def __init__(self, games_per_batch: int, games_per_epoch: int) -> None:
        super().__init__(
            f"games_per_batch ({games_per_batch}) is larger than games_per_epoch ({games_per_epoch}). "
            f"Either reduce games_per_batch, reduce epochs, or increase total_train_games."
        )


def main() -> None:  # noqa: PLR0915
    """Run training or testing for single-turn Yahtzee score maximization."""
    # Define configuration schema
    config_params = [
        ConfigParam("mode", str, "train", "Mode to run (train or test)", choices=["train", "test"]),
        ConfigParam("epochs", int, 500, "Number of training epochs"),
        ConfigParam(
            "total_train_games",
            int,
            260000,
            "Total number of Yahtzee games to train on",
            display_name="Total train games",
        ),
        ConfigParam(
            "games_per_batch",
            int,
            26,
            "Number of complete Yahtzee games per batch",
            display_name="Games per batch",
        ),
        ConfigParam("learning_rate", float, 0.0005, "Learning rate", display_name="Learning rate"),
        ConfigParam("hidden_size", int, 384, "Hidden layer size", display_name="Hidden size"),
        ConfigParam(
            "num_hidden", int, 4, "Number of hidden layers", display_name="Num hidden layers"
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
            0.01,
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
            0.0,
            "Maximum gradient norm for gradient clipping",
            display_name="Gradient clip value",
        ),
        ConfigParam(
            "entropy_coeff_start",
            float,
            0.05,
            "Starting coefficient for entropy regularization",
            display_name="Entropy coeff start",
        ),
        ConfigParam(
            "entropy_coeff_end",
            float,
            0.0,
            "Ending coefficient for entropy regularization",
            display_name="Entropy coeff end",
        ),
        ConfigParam(
            "entropy_anneal_percentage",
            float,
            0.4,
            "Percentage of training epochs over which to anneal entropy coefficient",
            display_name="Entropy anneal percentage",
        ),
        ConfigParam(
            "critic_coeff",
            float,
            0.05,
            "Coefficient for the critic loss term",
            display_name="Critic coefficient",
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
    total_train_games = config["total_train_games"]
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
    entropy_coeff_start = config["entropy_coeff_start"]
    entropy_coeff_end = config["entropy_coeff_end"]
    entropy_anneal_percentage = config["entropy_anneal_percentage"]
    critic_coeff = config["critic_coeff"]

    # Calculate games_per_epoch from total_train_games and epochs
    games_per_epoch = total_train_games // epochs

    # Validate that games_per_epoch divides evenly by games_per_batch
    if games_per_epoch % games_per_batch != 0:
        raise GamesPerEpochNotDivisibleError(
            total_train_games, epochs, games_per_epoch, games_per_batch
        )

    # Validate we have at least one batch per epoch
    if games_per_epoch < games_per_batch:
        raise BatchSizeTooLargeError(games_per_batch, games_per_epoch)

    # Calculate derived training metrics
    # Since we now run full games (not single turns), batch_size is just games_per_batch
    batch_size = games_per_batch
    updates_per_epoch = games_per_epoch // games_per_batch
    total_updates = updates_per_epoch * epochs
    games_per_update = games_per_batch
    games_per_epoch_actual = games_per_epoch  # Since we validate exact division
    total_games_actual = games_per_epoch * epochs

    # Log training configuration table
    config_table = [
        "=" * 50,
        "TRAINING INFORMATION",
        f"Total Games:       {total_games_actual:,}",
        f"Total Epochs:      {epochs:,}",
        f"Total Updates:     {total_updates:,}",
        f"Updates / Epoch:   {updates_per_epoch:,}",
        f"Games / Update:    {games_per_update:,}",
        f"Games / Epoch:     {games_per_epoch_actual:,}",
        "=" * 50,
    ]

    for line in config_table:
        print(line)
        log.info(line)

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
            epochs=epochs,
            min_lr_ratio=min_lr_ratio,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            entropy_coeff_start=entropy_coeff_start,
            entropy_coeff_end=entropy_coeff_end,
            entropy_anneal_epochs=int(entropy_anneal_percentage * epochs),
            critic_coeff=critic_coeff,
        )

        # Save hyperparameters explicitly
        model.save_hyperparameters(
            {
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "num_hidden": num_hidden,
                "dropout_rate": dropout_rate,
                "activation_function": activation_function,
                "epochs": epochs,
                "min_lr_ratio": min_lr_ratio,
                "gamma_max": gamma_max,
                "gamma_min": gamma_min,
                "total_train_games": total_train_games,
                "games_per_batch": games_per_batch,
                "total_games_actual": total_games_actual,
                "total_updates": total_updates,
                "updates_per_epoch": updates_per_epoch,
                "games_per_update": games_per_update,
                "games_per_epoch": games_per_epoch_actual,
                "gradient_clip_val": gradient_clip_val,
                "entropy_coeff_start": entropy_coeff_start,
                "entropy_coeff_end": entropy_coeff_end,
                "entropy_anneal_percentage": entropy_anneal_percentage,
                "entropy_anneal_epochs": int(entropy_anneal_percentage * epochs),
                "critic_coeff": critic_coeff,
            }
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
            check_val_every_n_epoch=5,  # Run validation every 5 epochs
            callbacks=[ckpt_cb],
            gradient_clip_val=gradient_clip_val,  # or 1.0, etc.
            gradient_clip_algorithm="norm",  # L2 norm clipping
        )

        # Create self-play dataset that collects episodes using the policy
        # Dataset now handles batching internally with parallel environments
        train_dataset = SelfPlayDataset(
            policy_net=model.policy_net,
            return_calculator=return_calculator,
            size=updates_per_epoch,  # Number of batches per epoch
            batch_size=batch_size,  # Number of parallel episodes per batch
        )
        # DataLoader batch_size=1 with passthrough collate since dataset already returns full batches
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, num_workers=0, collate_fn=lambda x: x[0]
        )

        # Create validation dataset (dummy since validation_step does its own game simulations)
        val_dataset = DummyDataset(size=1)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)

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
