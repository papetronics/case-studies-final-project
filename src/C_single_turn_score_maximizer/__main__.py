#!/usr/bin/env python
import os

import pytorch_lightning as lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from C_single_turn_score_maximizer import test_episode
from C_single_turn_score_maximizer.trainer import SingleTurnScoreMaximizerREINFORCETrainer
from utilities.dummy_dataset import DummyDataset
from utilities.initialize import ConfigParam, finish, initialize
from utilities.return_calculators import MonteCarloReturnCalculator

CKPT_DIR: str = "/opt/ml/checkpoints"  # SageMaker restores this from S3 on restart


def main() -> None:
    """Run training or testing for single-turn Yahtzee score maximization."""
    # Define configuration schema
    config_params = [
        ConfigParam("mode", str, "train", "Mode to run (train or test)", choices=["train", "test"]),
        ConfigParam("epochs", int, 500, "Number of training epochs"),
        ConfigParam(
            "episodes_per_batch",
            int,
            52,
            "Episodes per training batch",
            display_name="Episodes per batch",
        ),
        ConfigParam("learning_rate", float, 0.00075, "Learning rate", display_name="Learning rate"),
        ConfigParam("hidden_size", int, 384, "Hidden layer size", display_name="Hidden size"),
        ConfigParam(
            "num_hidden", int, 3, "Number of hidden layers", display_name="Num hidden layers"
        ),
        ConfigParam("dataset_size", int, 1300, "Dataset size", display_name="Dataset size"),
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
            0.0001,
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
    episodes_per_batch = config["episodes_per_batch"]
    learning_rate = config["learning_rate"]
    hidden_size = config["hidden_size"]
    num_hidden = config["num_hidden"]
    dataset_size = config["dataset_size"]
    checkpoint_path = config["checkpoint_path"]
    activation_function = config["activation_function"]
    min_lr_ratio = config["min_lr_ratio"]
    gamma_min = config["gamma_min"]
    gamma_max = config["gamma_max"]

    if mode == "test":
        # Test mode
        test_episode.main(checkpoint_path=checkpoint_path)
    else:
        # Create return calculator and model
        return_calculator = MonteCarloReturnCalculator()
        model = SingleTurnScoreMaximizerREINFORCETrainer(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            episodes_per_batch=episodes_per_batch,
            return_calculator=return_calculator,
            num_hidden=num_hidden,
            dropout_rate=0.1,
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
            save_last=True,
            save_top_k=0,
            every_n_train_steps=1000,  # or every_n_epochs=1
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
        )

        # Create dummy dataloader (required by Lightning but not used)
        dummy_dataset = DummyDataset(size=dataset_size // episodes_per_batch)
        train_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1, num_workers=15)

        # Create dummy validation dataloader
        val_dummy_dataset = DummyDataset(size=1)  # Just one batch for validation
        val_dataloader = torch.utils.data.DataLoader(
            val_dummy_dataset, batch_size=1, num_workers=15
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
