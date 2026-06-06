import os
import yaml
import wandb
from setup import Config
import platform


def _extra_wandb_tags_from_env() -> list[str]:
    return [
        tag.strip()
        for tag in os.getenv("WANDB_TAGS", "").split(",")
        if tag.strip()
    ]


def setup_methods_run(job_type, wandb_project_name="source-detection"):
    # create wandb run for tracking
    os_tag = platform.system().lower()
    tags = [f"job:{job_type}", f"os:{os_tag}", *_extra_wandb_tags_from_env()]
    wandb.init(
        project=wandb_project_name,
        tags=tags,
        job_type=job_type,
        settings=wandb.Settings(
            show_errors=True,  # Show error messages in the W&B App
            silent=False,      # Disable all W&B console output
            show_warnings=True,# Show warning messages in the W&B App
            show_info=True     # Show info messages in the W&B App
        )
    )
