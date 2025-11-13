import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers import logger as lightning_logger

import wandb


@dataclass
class ConfigParam:
    """Configuration parameter definition."""

    name: str
    type: type
    default: Any
    help: str
    choices: list[Any] | None = None
    display_name: str | None = None  # For pretty printing

    def get_display_name(self) -> str:
        """Get the display name for printing."""
        return self.display_name or self.name.replace("_", " ").replace("-", " ").title()


def initialize(  # noqa: C901, PLR0912, PLR0915
    scenario_name: str,
    config_params: list[ConfigParam],
    description: str | None = None,
    wandb_project_prefix: str = "yahtzee",
    logger_name: str | None = None,
) -> tuple[Any, dict[str, Any], Any]:
    """
    Initialize the project with configuration management, wandb setup, logger setup, and system info.

    Args:
        scenario_name: Name of the scenario (for logging)
        config_params: List of ConfigParam objects defining the configuration schema
        description: Description for the argument parser
        wandb_project_prefix: Prefix for wandb project name (default: "yahtzee")
        logger_name: Name for the logger (default: based on scenario)

    Returns
    -------
        tuple: (wandb_run, config_dict, logger)
    """
    # Initialize wandb if running inside a wandb launch agent or job
    is_launch = os.getenv("WANDB_JOB_NAME") or os.getenv("WANDB_RUN_ID")
    if is_launch:
        print("✅ Detected W&B launch agent context.")
        wandb_run = wandb.init()
    else:
        print("⚡ No W&B job context — skipping wandb.init() to avoid polluting real runs.")
        wandb_run = None

    # Log system information
    print("\n=== System Info ===")
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected!")

    # Add log_dir parameter if not already present
    param_names = [param.name for param in config_params]
    if "log_dir" not in param_names:
        config_params = config_params + [  # noqa: RUF005
            ConfigParam(
                "log_dir", str, "./logs", "Directory for logs", display_name="Log directory"
            )
        ]

    # Set up argument parser
    parser = argparse.ArgumentParser(description=description or f"Yahtzee {scenario_name}")

    # Add arguments dynamically
    for param in config_params:
        arg_name = f"--{param.name.replace('_', '-')}"
        kwargs: dict[str, Any] = {"help": param.help}

        # Handle optional parameters (default None)
        if param.default is None:
            kwargs["default"] = None
            kwargs["nargs"] = "?"  # Make it optional
            if param.type is not str:
                kwargs["type"] = param.type
        else:
            kwargs["type"] = param.type
            kwargs["default"] = param.default

        if param.choices:
            kwargs["choices"] = param.choices
        parser.add_argument(arg_name, **kwargs)

    args = parser.parse_args()

    # Build configuration dictionary from wandb config (if available) or argparse
    config = {}
    use_wandb = wandb_run is not None

    for param in config_params:
        if wandb_run is not None:
            config[param.name] = wandb_run.config.get(
                param.name, getattr(args, param.name.replace("-", "_"))
            )
        else:
            config[param.name] = getattr(args, param.name.replace("-", "_"))

    # Print configuration
    print("\n=== Hyperparameters ===")
    print(f"Scenario: {scenario_name}")
    for param in config_params:
        display_name = param.get_display_name()
        value = config[param.name]
        print(f"{display_name}: {value}")

    # Set up logger
    logger: lightning_logger.Logger
    if use_wandb:
        logger = WandbLogger(
            project=f"{wandb_project_prefix}-{scenario_name}",
            name=logger_name or f"{scenario_name}-training",
            log_model=True,
            experiment=wandb_run,  # Pass existing run to maintain step synchronization
        )
    else:
        # Get log_dir from config, fallback to default if not specified
        log_dir = config.get("log_dir", "./logs")
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        logger = TensorBoardLogger(
            log_dir, name=f"{wandb_project_prefix}-reinforce-{scenario_name}"
        )

    print(f"\n=== Starting {scenario_name} ===")

    return wandb_run, config, logger


def finish(wandb_run) -> None:  # noqa: ANN001
    """Finalize the training session."""
    print("Training completed!")

    if wandb_run is not None:
        wandb_run.finish()
