"""
Constants, configuration resolution and experiment sweeps.
"""

# %% Imports
from copy import deepcopy
from pathlib import Path

import yaml


# %% Paths
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent


# %% Image-related constants
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"
}

# Image configurations according to SLIP repo
STANDARD_IMAGE_SIZE = (224, 224)
SLIP_NORM_MEAN = [0.485, 0.456, 0.406]
SLIP_NORM_STD = [0.229, 0.224, 0.225]


# %% Runtime settings
BATCH_SIZES = {"small": 32, "base": 16, "large": 8}
CHUNK_SIZE = 5000
MAX_ATTEMPTS = 3

DEBUG = False
def message(*args, **kwargs):
    """Print diagnostic messages only in debug mode."""
    if DEBUG:
        print(*args, **kwargs)

# %% Load experiment configuration
def _load_yaml(filename: str) -> dict:
    """Load a YAML file from the config directory."""
    path = CONFIG_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_EXPERIMENT_CONFIG = _load_yaml("experiment.yml")

LAYOUTS = _EXPERIMENT_CONFIG["layouts"]
TRIAL_DESIGNS = _EXPERIMENT_CONFIG["trial_designs"]
METRICS = _EXPERIMENT_CONFIG["metrics"]
PROTOCOL_PRESETS = _EXPERIMENT_CONFIG["protocol_presets"]
EXPERIMENT_SWEEPS = _EXPERIMENT_CONFIG["experiment_sweeps"]

ACTIVE_SWEEP = _EXPERIMENT_CONFIG["run"]["sweep"]

def resolve_protocol(protocol_id: str) -> dict:
    """Resolve a protocol preset into its component configurations."""
    protocol = PROTOCOL_PRESETS[protocol_id]

    layout_id = protocol["layout"]
    trial_design_id = protocol["trial_design"]
    metrics_id = protocol["metrics"]

    return {
        "protocol_id": protocol_id,

        "layout_id": layout_id,
        "layout": deepcopy(LAYOUTS[layout_id]),

        "trial_design_id": trial_design_id,
        "trial_design": deepcopy(TRIAL_DESIGNS[trial_design_id]),

        "metrics_id": metrics_id,
        "metrics": deepcopy(METRICS[metrics_id]),
    }

def sweep_experiments(sweep_id: str):
    """Yield all experiments defined by a sweep."""
    sweep = EXPERIMENT_SWEEPS[sweep_id]
    protocol = resolve_protocol(sweep["protocol"])

    for model_id in sweep["models"]:
        for imageset_id, subsets in sweep["imagesets"].items():
            for subset in subsets:

                experiment_id = (
                    f"{model_id}__{imageset_id}__{subset}"
                )

                yield {
                    "experiment_id": experiment_id,
                    "sweep_id": sweep_id,

                    "model": model_id,
                    "imageset": imageset_id,
                    "subset": subset,

                    **deepcopy(protocol),
                }

def save_config_snapshot(config: dict, path):
    """Save the configuration for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )