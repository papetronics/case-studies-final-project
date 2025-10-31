#!/usr/bin/env python
import os
import torch
import wandb

def maybe_init_wandb():
    """
    Only initialize wandb if running inside a wandb launch agent or job.
    """
    is_launch = os.getenv("WANDB_JOB_NAME") or os.getenv("WANDB_RUN_ID")
    if is_launch:
        print("✅ Detected W&B launch agent context.")
        run = wandb.init()
    else:
        print("⚡ No W&B job context — skipping wandb.init() to avoid polluting real runs.")
        run = None
    return run

def main():
    run = maybe_init_wandb()

    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected!")

    if run is not None:
        # Safely access config only if W&B is initialized
        print("Sweep test param:", run.config.get("test"))
        run.finish()
    else:
        print("Sweep test param: [W&B not initialized]")

if __name__ == "__main__":
    main()
