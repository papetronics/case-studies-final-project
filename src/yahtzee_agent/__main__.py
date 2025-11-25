#!/usr/bin/env python
import logging
import os
import secrets

import pytorch_lightning as lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from tabulate import tabulate

from utilities.dummy_dataset import DummyDataset
from utilities.initialize import ConfigParam, initialize
from yahtzee_agent import test_episode
from yahtzee_agent.features import FEATURE_REGISTRY, create_features
from yahtzee_agent.self_play_dataset import SelfPlayDataset
from yahtzee_agent.trainer import Algorithm, YahtzeeAgentTrainer

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


class MissingPhiFeaturesError(InvalidTrainingConfigurationError):
    """Exception raised when phi_features configuration is not specified."""

    def __init__(self, available_features: list[str]) -> None:
        features_str = ", ".join(f"'{f}'" for f in available_features)
        super().__init__(f"phi_features must be specified. Available features: {features_str}")


class PPOBatchNotDivisibleError(InvalidTrainingConfigurationError):
    """Exception raised when PPO batch size is not divisible by number of minibatches."""

    def __init__(
        self,
        ppo_games_per_minibatch: int,
        ppo_batch_size: int,
    ) -> None:
        super().__init__(
            f"ppo_batch_size ({ppo_batch_size}) must be divisible by ppo_games_per_minibatch ({ppo_games_per_minibatch}). "
            f"Current ratio: {ppo_batch_size}/{ppo_games_per_minibatch} = {ppo_batch_size/ppo_games_per_minibatch}"
        )


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training or testing for single-turn Yahtzee score maximization."""
    # Define configuration schema
    config_params = [
        ConfigParam("mode", str, "train", "Mode to run (train or test)", choices=["train", "test"]),
        ConfigParam(
            "game_scenario",
            str,
            "full_game",
            "Game scenario: full_game (39 steps) or single_turn (3 steps)",
            choices=["full_game", "single_turn"],
            display_name="Game scenario",
        ),
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
            "he_kaiming_initialization",
            bool,
            False,
            "Use He/Kaiming initialization for better behavior at high learning rates",
            display_name="He/Kaiming initialization",
        ),
        ConfigParam(
            "checkpoint_path",
            str,
            None,
            "Path to model checkpoint for evaluation",
            display_name="Checkpoint path",
        ),
        ConfigParam(
            "phi_features",
            str,
            "dice_onehot,dice_counts,rolls_used,phase,has_earned_yahtzee,available_categories,percent_progress_towards_bonus,potential_scoring_opportunities,game_progress",
            f"Comma-separated list of phi features to enable. Available: {', '.join(FEATURE_REGISTRY.keys())}",
            display_name="Phi features",
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
            "gamma_end",
            float,
            1.0,
            "Discount factor for reward calculation (max, end)",
            display_name="Discount factor",
        ),
        ConfigParam(
            "gamma_start",
            float,
            None,  # Will default based on game_scenario: 0.9 for single_turn, 1.0 for full_game
            "Discount factor for reward calculation (min, start)",
            display_name="Discount factor",
        ),
        ConfigParam(
            "gamma_anneal_period",
            float,
            0.5,
            "Fraction of training over which to anneal gamma from start to end",
            display_name="Gamma anneal period",
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
            "entropy_coeff_rolling_max",
            float,
            0.15,
            "Maximum entropy coefficient for rolling head",
            display_name="Entropy coeff rolling max",
        ),
        ConfigParam(
            "entropy_coeff_rolling_min",
            float,
            0.045,
            "Minimum entropy coefficient for rolling head",
            display_name="Entropy coeff rolling min",
        ),
        ConfigParam(
            "entropy_coeff_scoring_max",
            float,
            0.3,
            "Maximum entropy coefficient for scoring head",
            display_name="Entropy coeff scoring max",
        ),
        ConfigParam(
            "entropy_coeff_scoring_min",
            float,
            0.006,
            "Minimum entropy coefficient for scoring head",
            display_name="Entropy coeff scoring min",
        ),
        ConfigParam(
            "entropy_hold_period",
            float,
            0.4,
            "Fraction of training to hold entropy at max before annealing",
            display_name="Entropy hold period",
        ),
        ConfigParam(
            "entropy_anneal_period",
            float,
            0.35,
            "Fraction of training over which to anneal entropy from max to min",
            display_name="Entropy anneal period",
        ),
        ConfigParam(
            "critic_coeff",
            float,
            0.05,
            "Coefficient for the critic loss term",
            display_name="Critic coefficient",
        ),
        ConfigParam(
            "rolling_action_representation",
            str,
            "bernoulli",
            "Representation for rolling actions: 'bernoulli' (5 independent binary) or 'categorical' (32 discrete masks)",
            choices=["bernoulli", "categorical"],
            display_name="Rolling action representation",
        ),
        ConfigParam(
            "algorithm",
            str,
            "reinforce",
            "Training algorithm: 'reinforce' (REINFORCE with Monte Carlo returns), 'a2c' (Advantage Actor-Critic with TD(0) bootstrapping), or 'ppo' (Proximal Policy Optimization)",
            choices=["reinforce", "a2c", "ppo"],
            display_name="Algorithm",
        ),
        ConfigParam(
            "gae_lambda",
            float,
            0.0,
            "GAE lambda parameter (for A2C algorithm only)",
        ),
        ConfigParam(
            "upper_score_regression_loss_weight",
            float,
            0.1,
            "Weight for the upper score regression loss term",
            display_name="Upper score regression loss weight",
        ),
        ConfigParam(
            "upper_score_shaping_weight",
            float,
            0.1,
            "Weight for the upper score regression shaping loss term",
            display_name="Upper score regression shaping weight",
        ),
        ConfigParam(
            "clip_epsilon",
            float,
            0.2,
            "PPO clipping parameter (epsilon) for clipped surrogate objective",
            display_name="Clip epsilon",
        ),
        ConfigParam(
            "ppo_games_per_minibatch",
            int,
            4,
            "Number of games per PPO minibatch (batch is split into multiple minibatches).",
            display_name="PPO games per minibatch",
        ),
        ConfigParam(
            "ppo_epochs",
            int,
            3,
            "Number of epochs to train over each PPO batch.",
            display_name="PPO epochs",
        ),
    ]

    # Initialize project with configuration
    config, logger = initialize(
        config_params=config_params,
        scenario_name="yahtzee_agent",
        description="Yahtzee Agent using Reinforcement Learning",
    )

    # Extract config values for easy access
    mode = config["mode"]
    game_scenario = config["game_scenario"]
    epochs = config["epochs"]
    total_train_games = config["total_train_games"]
    games_per_batch = config["games_per_batch"]
    learning_rate = config["learning_rate"]
    hidden_size = config["hidden_size"]
    num_hidden = config["num_hidden"]
    checkpoint_path = config["checkpoint_path"]
    phi_features_str = config["phi_features"]
    activation_function = config["activation_function"]
    min_lr_ratio = config["min_lr_ratio"]
    gamma_start = config["gamma_start"]
    gamma_end = config["gamma_end"]
    gamma_anneal_period = config["gamma_anneal_period"]
    dropout_rate = config["dropout_rate"]
    gradient_clip_val = config["gradient_clip_val"]
    entropy_coeff_rolling_max = config["entropy_coeff_rolling_max"]
    entropy_coeff_rolling_min = config["entropy_coeff_rolling_min"]
    entropy_coeff_scoring_max = config["entropy_coeff_scoring_max"]
    entropy_coeff_scoring_min = config["entropy_coeff_scoring_min"]
    entropy_hold_period = config["entropy_hold_period"]
    entropy_anneal_period = config["entropy_anneal_period"]
    critic_coeff = config["critic_coeff"]
    rolling_action_representation = config["rolling_action_representation"]
    algorithm = config["algorithm"]
    gae_lambda = config["gae_lambda"]
    upper_score_regression_loss_weight = config["upper_score_regression_loss_weight"]
    upper_score_shaping_weight = config["upper_score_shaping_weight"]
    clip_epsilon = config["clip_epsilon"]
    ppo_games_per_minibatch = config["ppo_games_per_minibatch"]
    ppo_epochs = config["ppo_epochs"]

    torch.set_float32_matmul_precision("medium")

    # Parse phi features from comma-separated string
    if phi_features_str and phi_features_str.strip():
        feature_names = [name.strip() for name in phi_features_str.split(",") if name.strip()]
        phi_features = create_features(feature_names)
        log.info(f"Enabled phi features: {[f.name for f in phi_features]}")
    else:
        raise MissingPhiFeaturesError(list(FEATURE_REGISTRY.keys()))

    # Set gamma_start default based on game_scenario if not explicitly provided
    if gamma_start is None:
        gamma_start = 0.9 if game_scenario == "single_turn" else 1.0
        log.info(f"Setting gamma_start={gamma_start} based on game_scenario={game_scenario}")

    # Calculate derived values based on game scenario
    if game_scenario == "single_turn":
        num_steps_per_episode = 3  # One turn: roll, roll, score
        stagger_environments = True  # Distribute envs across turns 0-12 to avoid temporal bias
        batch_size_multiplier = TURNS_PER_GAME  # Each game contributes 13 single-turn episodes
        log.info("Game scenario: single_turn (3 steps per episode, staggered environments)")
    else:  # full_game
        num_steps_per_episode = 39  # Full game: 13 turns * 3 steps per turn
        stagger_environments = False  # All games start from turn 0
        batch_size_multiplier = 1  # Each game is one full episode
        log.info("Game scenario: full_game (39 steps per episode, no staggering)")

    # Calculate games_per_epoch from total_train_games and epochs
    games_per_epoch = total_train_games // epochs

    # check that PPO minibatches evenly divide the batch size
    if algorithm == "ppo":
        ppo_batch_size = games_per_batch * num_steps_per_episode
        ppo_minibatch_size = ppo_games_per_minibatch * num_steps_per_episode
        if ppo_batch_size % ppo_minibatch_size != 0:
            raise PPOBatchNotDivisibleError(ppo_games_per_minibatch, games_per_batch)

    # Validate that games_per_epoch divides evenly by games_per_batch
    if games_per_epoch % games_per_batch != 0:
        raise GamesPerEpochNotDivisibleError(
            total_train_games, epochs, games_per_epoch, games_per_batch
        )

    # Validate we have at least one batch per epoch
    if games_per_epoch < games_per_batch:
        raise BatchSizeTooLargeError(games_per_batch, games_per_epoch)

    # Calculate derived training metrics
    # batch_size is the number of parallel environments
    # In single_turn: games_per_batch * 13 (one env per turn per game)
    # In full_game: games_per_batch (one env per game)
    batch_size = games_per_batch * batch_size_multiplier
    updates_per_epoch = games_per_epoch // games_per_batch
    total_updates = updates_per_epoch * epochs
    games_per_epoch_actual = games_per_epoch  # Since we validate exact division
    total_games_actual = games_per_epoch * epochs

    # Log training configuration table
    scenario_name = "SINGLE TURN" if game_scenario == "single_turn" else "FULL GAME"
    print(f"\n{'=' * 50}")
    print(f"{scenario_name:^50}")
    print(f"{'=' * 50}")
    print("\nTRAINING INFORMATION")

    config_table = [
        ["Total Games", f"{total_games_actual:,}"],
        ["Total Epochs", f"{epochs:,}"],
        ["Total Updates", f"{total_updates:,}"],
        ["Updates / Epoch", f"{updates_per_epoch:,}"],
        ["Games / Update", f"{games_per_batch:,}"],
        ["Games / Epoch", f"{games_per_epoch_actual:,}"],
    ]

    table_str = tabulate(config_table, headers=["Metric", "Value"], tablefmt="github")
    print(table_str)
    for line in table_str.split("\n"):
        log.info(line)

    if mode == "test":
        # Test mode
        test_episode.main(checkpoint_path=checkpoint_path)
    else:
        if algorithm == "a2c":
            algorithm = Algorithm.A2C
            log.info("Using A2C (Advantage Actor-Critic) with TD(0) bootstrapping")
        elif algorithm == "ppo":
            algorithm = Algorithm.PPO
            log.info("Using PPO (Proximal Policy Optimization) with clipped surrogate objective")
        else:  # reinforce
            algorithm = Algorithm.REINFORCE
            log.info("Using REINFORCE with Monte Carlo returns")

        he_kaiming_initialization = config.get("he_kaiming_initialization", False)
        model = YahtzeeAgentTrainer(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            algorithm=algorithm,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
            epochs=epochs,
            min_lr_ratio=min_lr_ratio,
            gamma_start=gamma_start,
            gamma_end=gamma_end,
            gamma_anneal_period=gamma_anneal_period,
            entropy_coeff_rolling_max=entropy_coeff_rolling_max,
            entropy_coeff_rolling_min=entropy_coeff_rolling_min,
            entropy_coeff_scoring_max=entropy_coeff_scoring_max,
            entropy_coeff_scoring_min=entropy_coeff_scoring_min,
            entropy_hold_period=entropy_hold_period,
            entropy_anneal_period=entropy_anneal_period,
            critic_coeff=critic_coeff,
            num_steps_per_episode=num_steps_per_episode,
            features=phi_features,
            rolling_action_representation=rolling_action_representation,
            he_kaiming_initialization=he_kaiming_initialization,
            gae_lambda=gae_lambda,
            upper_score_regression_loss_weight=upper_score_regression_loss_weight,
            upper_score_shaping_weight=upper_score_shaping_weight,
            clip_epsilon=clip_epsilon,
            ppo_games_per_minibatch=ppo_games_per_minibatch,
            ppo_epochs=ppo_epochs,
            gradient_clip_val=gradient_clip_val,
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
                "gamma_end": gamma_end,
                "gamma_start": gamma_start,
                "gamma_anneal_period": gamma_anneal_period,
                "total_train_games": total_train_games,
                "games_per_batch": games_per_batch,
                "gradient_clip_val": gradient_clip_val,
                "entropy_coeff_rolling_max": entropy_coeff_rolling_max,
                "entropy_coeff_rolling_min": entropy_coeff_rolling_min,
                "entropy_coeff_scoring_max": entropy_coeff_scoring_max,
                "entropy_coeff_scoring_min": entropy_coeff_scoring_min,
                "entropy_hold_period": entropy_hold_period,
                "entropy_anneal_period": entropy_anneal_period,
                "critic_coeff": critic_coeff,
                "game_scenario": game_scenario,
                "phi_features": phi_features_str,
                "rolling_action_representation": rolling_action_representation,
                "algorithm": algorithm,
                "gae_lambda": gae_lambda,
                "upper_score_regression_loss_weight": upper_score_regression_loss_weight,
                "upper_score_shaping_weight": upper_score_shaping_weight,
                "clip_epsilon": clip_epsilon,
                "ppo_games_per_minibatch": ppo_games_per_minibatch,
                "ppo_epochs": ppo_epochs,
            }
        )

        # Log entropy schedule info
        entropy_hold_epochs = int(entropy_hold_period * epochs)
        entropy_anneal_epochs = int(entropy_anneal_period * epochs)
        model.log("stat/entropy_hold_epochs", entropy_hold_epochs, prog_bar=False)
        model.log("stat/entropy_anneal_epochs", entropy_anneal_epochs, prog_bar=False)
        model.log("stat/updates_per_epoch", updates_per_epoch, prog_bar=False)
        model.log("stat/total_games_actual", total_games_actual, prog_bar=False)
        model.log("stat/total_updates", total_updates, prog_bar=False)
        model.log("stat/games_per_epoch", games_per_epoch_actual, prog_bar=False)

        run_scope = os.getenv("WANDB_RUN_ID") or secrets.token_hex(4)

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
            gradient_clip_val=gradient_clip_val
            if algorithm != Algorithm.PPO
            else None,  # or 1.0, etc.
            gradient_clip_algorithm="norm"
            if algorithm != Algorithm.PPO
            else None,  # L2 norm clipping
        )

        # Create self-play dataset that collects episodes using the policy
        # Dataset handles batching internally with parallel environments
        # batch_size = number of parallel environments (games_per_batch * multiplier)
        # num_steps_per_episode = 3 for single_turn, 39 for full_game
        train_dataset = SelfPlayDataset(
            policy_net=model.policy_net,
            size=updates_per_epoch,  # Number of batches per epoch
            batch_size=batch_size,  # Number of parallel environments
            num_steps_per_episode=num_steps_per_episode,
            stagger_environments=stagger_environments,
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


if __name__ == "__main__":
    main()
