"""
src/evaluation.py
=================
Evaluation, metrics, predictions export, and visualization.

Used by training, ensemble, and submission scripts.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss,
)


# ---------------------------------------------------------------------------
# Predictions export (wide-format probabilities)
# ---------------------------------------------------------------------------

def build_predictions_df(
    df: pd.DataFrame,
    probs: np.ndarray,
    classes: list[str],
    split_name: str = "val",
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Build a wide-format predictions DataFrame.

    Columns: id, split, <class>_prob (one per class), pred_class, confidence
    If include_labels: also true_class, correct

    Args:
        df: DataFrame with at least 'original_id' (and 'label' if include_labels).
            For crop-level data, will aggregate to image-level.
        probs: (N, C) probability matrix aligned with df rows.
        classes: ordered class names.
        split_name: 'train', 'val', or 'test'.
        include_labels: whether to include ground truth columns.
    """
    work = df.copy()
    for i, cls in enumerate(classes):
        work[f"{cls}_prob"] = probs[:, i]
    work["max_prob"] = probs.max(axis=1)

    prob_cols = [f"{cls}_prob" for cls in classes]

    # Aggregate crops -> images (confidence-weighted mean)
    rows = []
    for img_id, group in work.groupby("original_id"):
        if len(group) == 1:
            avg_probs = group[prob_cols].values[0]
        else:
            weights = group["max_prob"].values
            weights = weights / weights.sum()
            avg_probs = (group[prob_cols].values * weights[:, None]).sum(axis=0)

        pred_idx = avg_probs.argmax()
        row = {
            "id": img_id,
            "split": split_name,
            "pred_class": classes[pred_idx],
            "confidence": avg_probs[pred_idx],
            "n_crops": len(group),
        }
        for j, cls in enumerate(classes):
            row[f"{cls}_prob"] = round(float(avg_probs[j]), 6)

        if include_labels and "label" in group.columns:
            row["true_class"] = group["label"].iloc[0]
            row["correct"] = row["true_class"] == row["pred_class"]

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(pred_df: pd.DataFrame, classes: list[str]) -> dict:
    """
    Compute accuracy, F1 macro, per-class F1, and log_loss from a
    predictions DataFrame (must have true_class, pred_class, and *_prob cols).
    """
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_true = pred_df["true_class"].map(cls2idx).values
    y_pred = pred_df["pred_class"].map(cls2idx).values

    prob_cols = [f"{c}_prob" for c in classes]
    y_prob = pred_df[prob_cols].values

    # Clip for log_loss stability
    y_prob_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    y_prob_clipped = y_prob_clipped / y_prob_clipped.sum(axis=1, keepdims=True)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    logloss = log_loss(y_true, y_prob_clipped, labels=list(range(len(classes))))

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "log_loss": round(logloss, 4),
        "f1_per_class": {classes[i]: round(f1_per[i], 4) for i in range(len(classes))},
    }


def print_classification_report(pred_df: pd.DataFrame, classes: list[str]) -> str:
    """Print and return sklearn classification_report string."""
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_true = pred_df["true_class"].map(cls2idx).values
    y_pred = pred_df["pred_class"].map(cls2idx).values
    report = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0)
    print(report)
    return report


def compute_confusion_matrix(pred_df: pd.DataFrame,
                             classes: list[str]) -> np.ndarray:
    """Compute confusion matrix from predictions DataFrame."""
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_true = pred_df["true_class"].map(cls2idx).values
    y_pred = pred_df["pred_class"].map(cls2idx).values
    return confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, epochs_frozen: int,
                         save_path: Path):
    """Plot loss and accuracy training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_x = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs_x, history["train_loss"], label="train")
    ax1.plot(epochs_x, history["val_loss"], label="val")
    ax1.axvline(x=epochs_frozen + 0.5, color="gray", ls="--", alpha=0.5,
                label="unfreeze")
    ax1.set(title="Loss", xlabel="Epoch")
    ax1.legend()

    ax2.plot(epochs_x, history["train_acc"], label="train")
    ax2.plot(epochs_x, history["val_acc"], label="val")
    ax2.axvline(x=epochs_frozen + 0.5, color="gray", ls="--", alpha=0.5)
    ax2.set(title="Accuracy", xlabel="Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curves -> {save_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: list[str],
                          save_path: Path, title: str = "Confusion Matrix"):
    """Plot annotated confusion matrix heatmap."""
    n = len(classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([c[:8] for c in classes], rotation=45, ha="right")
    ax.set_yticklabels([c[:8] for c in classes])

    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=8)

    ax.set(xlabel="Predicted", ylabel="True", title=title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix -> {save_path}")


# ---------------------------------------------------------------------------
# Save standard model output directory
# ---------------------------------------------------------------------------

def save_model_outputs(
    output_dir: Path,
    pred_df: pd.DataFrame,
    classes: list[str],
    config: dict,
    history: dict | None = None,
    epochs_frozen: int = 0,
    report_str: str | None = None,
    script_path: str | None = None,
):
    """
    Save the standard set of model outputs to output_dir:
      - training_config.json
      - classification_report.txt
      - confusion_matrix.png
      - training_curves.png (if history provided)
      - predictions_val.csv (or predictions_train.csv)
      - Copy of training script (if script_path provided)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    if report_str is None:
        report_str = print_classification_report(pred_df, classes)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)

    # Confusion matrix
    cm = compute_confusion_matrix(pred_df, classes)
    plot_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")

    # Training curves
    if history:
        plot_training_curves(history, epochs_frozen,
                             output_dir / "training_curves.png")

    # Predictions CSV
    split = pred_df["split"].iloc[0] if "split" in pred_df.columns else "val"
    pred_path = output_dir / f"predictions_{split}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions -> {pred_path}")

    # Metrics into config
    if "true_class" in pred_df.columns:
        metrics = compute_metrics(pred_df, classes)
        config.update(metrics)
        config["confusion_matrix"] = cm.tolist()

    # Model checksum
    model_path = output_dir / "best_model.pt"
    if model_path.exists():
        config["model_checksum_md5"] = _model_checksum(model_path)

    # Config JSON
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config -> {config_path}")

    # Script copy
    if script_path:
        import shutil
        shutil.copy2(script_path, output_dir / Path(script_path).name)


def _model_checksum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
