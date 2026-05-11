import yaml
import wandb
from setup import Config
import platform



def setup_methods_run(job_type, wandb_project_name="source-detection"):
    # create wandb run for tracking
    os_tag = platform.system().lower()
    tags = [f"job:{job_type}", f"os:{os_tag}"]
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
