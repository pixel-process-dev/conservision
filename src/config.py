"""
src/config.py
=============
Load and validate experiment + model configs. Single source of truth for
classes, paths, and shared constants.
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Derived constants (always in sync with experiment config)
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_experiment(config_path: str | Path | None = None) -> dict:
    """Load the experiment-level config."""
    path = Path(config_path) if config_path else _CONFIG_DIR / "experiment.json"
    return load_json(path)


def load_model_config(config_path: str | Path) -> dict:
    """Load a model config and merge with experiment defaults."""
    exp = load_experiment()
    model = load_json(config_path)

    # Attach experiment-level fields the model config will need
    model.setdefault("classes", exp["classes"])
    model.setdefault("num_classes", len(exp["classes"]))
    model.setdefault("imagenet_mean", exp["imagenet_mean"])
    model.setdefault("imagenet_std", exp["imagenet_std"])
    model.setdefault("val_fold", exp["val_fold"])
    model.setdefault("n_folds", exp["n_folds"])
    model.setdefault("num_workers", exp["num_workers"])

    return model


def load_augmentation(aug_name: str) -> dict:
    """Load a named augmentation preset."""
    aug_cfg = load_json(_CONFIG_DIR / "augmentation.json")
    if aug_name not in aug_cfg:
        raise ValueError(f"Unknown augmentation preset: {aug_name!r}. "
                         f"Available: {list(aug_cfg.keys())}")
    return aug_cfg[aug_name]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_classes(exp: dict | None = None) -> list[str]:
    if exp is None:
        exp = load_experiment()
    return exp["classes"]


def cls2idx(classes: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


def idx2cls(classes: list[str]) -> dict[int, str]:
    return {i: c for i, c in enumerate(classes)}
