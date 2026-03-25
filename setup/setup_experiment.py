import yaml
import wandb
from pathlib import Path
from setup import Config
import platform


def setup_tsir_run(cfg_path, wandb_project_name="source-detection"):
    with open(cfg_path) as f:
        config_data = yaml.safe_load(f)

    # if it's an empirical network, load some info about it
    if config_data["nwk"]["type"] == "empirical":
        with open("nwk/" + config_data["nwk"]["name"] + ".yml") as f:
            config_data["nwk"].update(yaml.safe_load(f))

    # create wandb run for tracking
    job_type = "tsir"
    os_tag = platform.system().lower()
    tags = [f"job:{job_type}", f"os:{os_tag}"]
    wandb.init(
        project=wandb_project_name,
        config=config_data,
        tags=tags,
        job_type=job_type,
        settings=wandb.Settings(
            show_errors=True,  # Show error messages in the W&B App
            silent=False,      # Disable all W&B console output
            show_warnings=True,# Show warning messages in the W&B App
            show_info=True     # Show info messages in the W&B App
        )
    )

    # create config object with relevant parameters
    cfg = Config(config_data)
    return cfg
