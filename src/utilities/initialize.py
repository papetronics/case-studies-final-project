import argparse
import os
from dataclasses import dataclass
from typing import Any, cast

import torch
from pytorch_lightning.loggers import (
    TensorBoardLogger,
    WandbLogger,
)
from pytorch_lightning.loggers import (
    logger as lightning_logger,
)
from tabulate import tabulate

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


def initialize(
    scenario_name: str,
    config_params: list[ConfigParam],
    description: str,
    wandb_project_prefix: str = "yahtzee",
    logger_name: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Initialize the project with configuration management, wandb logger setup, and system info."""
    run_id = os.getenv("WANDB_RUN_ID") or None
    use_wandb = run_id is not None

    logger: lightning_logger.Logger

    if use_wandb:
        print(f"✅ Detected W&B launch agent context. (run_id={run_id})")
        logger = WandbLogger(
            project=f"{wandb_project_prefix}-{scenario_name}",
            name=logger_name or f"{scenario_name}-training",
            log_model=True,
            id=run_id,
            resume="allow",
        )

    config = get_hyperparameters(
        config_params, description, cast("WandbLogger", logger).experiment if use_wandb else None
    )

    if not use_wandb:
        print("⚡ No W&B job context — using TensorBoardLogger.")
        log_dir = config.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        logger = TensorBoardLogger(
            log_dir, name=f"{wandb_project_prefix}-reinforce-{scenario_name}"
        )

    # Log system information

    print_cuda_info()
    print_hyperparameters(config, config_params)

    return config, logger


def get_hyperparameters(
    config_params: list[ConfigParam], description: str, wandb_run: wandb.Run | None
) -> dict[str, Any]:
    """Extract hyperparameters from W&B and CLI."""
    param_names = [param.name for param in config_params]
    if "log_dir" not in param_names:
        config_params = config_params + [  # noqa: RUF005
            ConfigParam(
                "log_dir", str, "./logs", "Directory for logs", display_name="Log directory"
            )
        ]

    parser = argparse.ArgumentParser(description=description)

    for param in config_params:
        arg_name = f"--{param.name.replace('_', '-')}"
        kwargs: dict[str, Any] = {"help": param.help}

        if param.default is None:
            kwargs["default"] = None
            kwargs["nargs"] = "?"
            if param.type is not str:
                kwargs["type"] = param.type
        else:
            kwargs["type"] = param.type
            kwargs["default"] = param.default

        if param.choices:
            kwargs["choices"] = param.choices
        parser.add_argument(arg_name, **kwargs)

    args = parser.parse_args()

    config = {}

    for param in config_params:
        if wandb_run is not None:
            config[param.name] = wandb_run.config.get(
                param.name, getattr(args, param.name.replace("-", "_"))
            )
        else:
            config[param.name] = getattr(args, param.name.replace("-", "_"))

    return config


def print_hyperparameters(config: dict[str, Any], config_params: list[ConfigParam]) -> None:
    """Display parsed hyperparameters in a compact table."""
    print("\n=== Hyperparameters ===")
    rows: list[list[str]] = []
    for param in config_params:
        display_name = param.get_display_name()
        value = config[param.name]
        rows.append([display_name, str(value)])
    if rows:
        print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="github"))


def print_cuda_info() -> None:
    """Summarize CUDA availability and detected GPU devices."""
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    rows: list[list[str]] = [
        ["CUDA available", str(cuda_available)],
        ["Using device", str(device)],
    ]
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        rows.append(["GPU count", str(gpu_count)])
        for index in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(index)
            rows.append([f"GPU {index} name", gpu_name])
    else:
        rows.append(["GPU status", "No GPU detected"])
    print(tabulate(rows, headers=["Property", "Value"], tablefmt="github"))
