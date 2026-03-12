"""
scripts/05_predict.py
=====================
Generate fold-level validation predictions with raw logits and probabilities.

Produces per-fold CSVs containing log-probabilities (for downstream temperature
scaling) and softmax probabilities, aggregated from crops to per-image using
confidence-weighted mean (matching the submission pipeline).

Output per fold_dir:
    val_predictions.csv       — standard inference (logits + probas + meta)
    val_predictions_tta.csv   — TTA inference (optional, same columns)

Usage:
    # ── Single fold ──────────────────────────────────────────────
    python scripts/05_predict.py \
        --fold_dir models/dinov2_.1_folds/fold_0 \
        --crop_dir data/competition/crops_10 \
        --data_dir data/competition

    # ── All folds in one model ───────────────────────────────────
    python scripts/05_predict.py \
        --model_dir models/dinov2_.1_folds \
        --crop_dir data/competition/crops_10 \
        --data_dir data/competition

    # ── All models (batch) ───────────────────────────────────────
    python scripts/05_predict.py \
        --models_dir models/ \
        --crop_map configs/crop_map.json \
        --data_dir data/competition

    # ── With TTA ─────────────────────────────────────────────────
    python scripts/05_predict.py \
        --model_dir models/dinov2_.1_folds \
        --crop_dir data/competition/crops_10 \
        --data_dir data/competition \
        --tta --n_tta 5

crop_map.json format (threshold → crop directory):
    {
        ".1":   "data/competition/crops_10",
        ".05":  "data/competition/crops_05",
        "full": "data/competition/crops_full"
    }
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, f1_score, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    load_augmentation, get_classes, cls2idx as make_cls2idx,
)
from src.transforms import build_val_transform, build_tta_transforms
from src.dataset import CropInferenceDataset
from src.data import load_train_data, get_site_split
from src.inference import load_trained_model

CLASSES = get_classes()
NUM_CLASSES = len(CLASSES)
CLS2IDX = make_cls2idx(CLASSES)
SKIP_DIRS = {"archive", "__pycache__", ".git", ".ipynb_checkpoints"}


# ═════════════════════════════════════════════════════════════════════════════
# Inference — logit extraction
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_logits(model, loader, device):
    """Forward pass returning raw logits as (N, C) numpy array.

    CropInferenceDataset yields (image_tensor, row_index). Row alignment
    is guaranteed by shuffle=False.
    """
    model.eval()
    all_logits = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        all_logits.append(logits.cpu())

    logits_cat = torch.cat(all_logits, dim=0)

    # Safety: detect if model returns softmax probabilities
    sample_sum = logits_cat[0].sum().item()
    if abs(sample_sum - 1.0) < 0.01:
        print("  ⚠ Model output sums to ~1 — applying log() to recover "
              "log-probabilities.")
        logits_cat = torch.log(logits_cat.clamp(min=1e-7))

    return logits_cat.numpy()


def predict_logits_tta(model, df_val, crop_train_dir, img_size,
                       batch_size, num_workers, device, mean, std,
                       n_tta=5):
    """Run TTA inference, averaging logits across augmented passes.

    Averaging in logit space (before softmax) preserves full dynamic
    range for downstream temperature scaling.
    """
    tta_cfg = load_augmentation("tta")
    passes = tta_cfg["passes"][:n_tta]
    tta_transforms = build_tta_transforms(passes, img_size, mean, std)

    all_logits = None
    for i, tfm in enumerate(tta_transforms):
        print(f"    TTA pass {i+1}/{len(tta_transforms)}")
        dataset = CropInferenceDataset(df_val, crop_train_dir,
                                       transform=tfm, img_size=img_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
        logits = predict_logits(model, loader, device)

        if all_logits is None:
            all_logits = logits
        else:
            all_logits += logits

    return all_logits / len(tta_transforms)


# ═════════════════════════════════════════════════════════════════════════════
# Crop → image aggregation (matches build_predictions_df / submission)
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_to_images(df_val, logits, classes):
    """Aggregate crop-level logits to per-image predictions.

    Aggregation uses confidence-weighted mean on probabilities, matching
    the submission pipeline in build_predictions_df. Per-image "logits"
    are stored as log(aggregated_probas) for temperature scaling:
        softmax(log(p) / T)  ≡  p^(1/T) / Σ p^(1/T)

    Returns DataFrame with columns:
        id, site, n_crops, true_label, pred_label, confidence,
        {class}_logit × C, {class}_prob × C
    """
    probas = F.softmax(torch.tensor(logits, dtype=torch.float32),
                       dim=1).numpy()
    prob_cols = [f"{c}_prob" for c in classes]

    work = df_val.copy()
    for i, c in enumerate(classes):
        work[f"{c}_prob"] = probas[:, i]
    work["max_prob"] = probas.max(axis=1)

    rows = []
    for img_id, group in work.groupby("original_id"):
        if len(group) == 1:
            avg_probs = group[prob_cols].values[0]
        else:
            # Confidence-weighted mean (matches build_predictions_df)
            weights = group["max_prob"].values
            weights = weights / weights.sum()
            avg_probs = (group[prob_cols].values
                         * weights[:, None]).sum(axis=0)

        pred_idx = avg_probs.argmax()

        # Log-probabilities as logits for temperature scaling
        avg_probs_clipped = np.clip(avg_probs, 1e-7, 1.0)
        log_probs = np.log(avg_probs_clipped)

        row = {
            "id": img_id,
            "site": (group["site"].iloc[0]
                     if "site" in group.columns else ""),
            "n_crops": len(group),
            "true_label": group["label"].iloc[0],
            "pred_label": classes[pred_idx],
            "confidence": round(float(avg_probs[pred_idx]), 6),
        }
        for i, c in enumerate(classes):
            row[f"{c}_logit"] = round(float(log_probs[i]), 6)
        for i, c in enumerate(classes):
            row[f"{c}_prob"] = round(float(avg_probs[i]), 6)

        rows.append(row)

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(df_out, classes):
    """Compute log loss, F1 macro, and accuracy from output DataFrame."""
    true_idx = df_out["true_label"].map(CLS2IDX).values
    prob_cols = [f"{c}_prob" for c in classes]
    probas = df_out[prob_cols].values

    # Clip for stability (matches evaluation.py)
    probas_clipped = np.clip(probas, 1e-7, 1.0 - 1e-7)
    probas_clipped = (probas_clipped
                      / probas_clipped.sum(axis=1, keepdims=True))
    pred_idx = probas.argmax(axis=1)

    return {
        "log_loss": log_loss(true_idx, probas_clipped,
                             labels=list(range(len(classes)))),
        "f1_macro": f1_score(true_idx, pred_idx, average="macro",
                             zero_division=0),
        "accuracy": accuracy_score(true_idx, pred_idx),
    }


def print_metrics(metrics, prefix=""):
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    print(f"    {prefix}{', '.join(parts)}")


# ═════════════════════════════════════════════════════════════════════════════
# Core: fold-level prediction
# ═════════════════════════════════════════════════════════════════════════════

def predict_fold(fold_dir, crop_dir, data_dir, device,
                 tta=False, n_tta=5):
    """Generate val predictions for a single fold.

    Loads model + config from fold_dir, rebuilds the exact val split and
    transforms used during training, runs inference, aggregates crops →
    images, computes metrics, and saves CSV.

    Returns metrics dict (or None on skip).
    """
    fold_dir = Path(fold_dir)
    crop_dir = Path(crop_dir)
    data_dir = Path(data_dir)

    if not (fold_dir / "best_model.pt").exists():
        print(f"  SKIP {fold_dir} — no best_model.pt")
        return None

    # ── Load model + config ──────────────────────────────────────────────
    model, config = load_trained_model(fold_dir, device, NUM_CLASSES)

    img_size = config["img_size"]
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)
    val_fold = config.get("val_fold", 0)
    n_folds = config.get("n_folds", 5)
    backbone = config.get("backbone", "unknown")
    aug_name = config.get("augmentation", "standard")

    mean = config.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = config.get("imagenet_std", [0.229, 0.224, 0.225])
    if isinstance(mean, str):
        mean = json.loads(mean)
    if isinstance(std, str):
        std = json.loads(std)

    # ── Load data + rebuild val split (must match training) ──────────────
    df = load_train_data(data_dir, crop_dir, CLASSES, CLS2IDX)
    _, df_val = get_site_split(df, val_fold, n_folds)
    crop_train_dir = str(crop_dir / "train")

    n_images = df_val["original_id"].nunique()
    print(f"  fold_{val_fold} | {len(df_val)} crops ({n_images} images) | "
          f"{backbone} | img={img_size} | aug={aug_name}")

    # ── Build val transforms (exact match to training eval) ──────────────
    aug_cfg = load_augmentation(aug_name)
    val_tfm = build_val_transform(aug_cfg, img_size, mean, std)

    # ── Standard predictions ─────────────────────────────────────────────
    t0 = time.time()
    dataset = CropInferenceDataset(df_val, crop_train_dir,
                                   transform=val_tfm, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    logits = predict_logits(model, loader, device)
    df_out = aggregate_to_images(df_val, logits, CLASSES)
    df_out.insert(1, "fold", val_fold)

    out_path = fold_dir / "inf_val_predictions.csv"
    df_out.to_csv(out_path, index=False)
    metrics = compute_metrics(df_out, CLASSES)
    print_metrics(metrics)
    print(f"    → {out_path.name} "
          f"({len(df_out)} images, {time.time() - t0:.0f}s)")

    # ── TTA predictions ──────────────────────────────────────────────────
    if tta:
        t0 = time.time()
        logits_tta = predict_logits_tta(
            model, df_val, crop_train_dir, img_size, batch_size,
            num_workers, device, mean, std, n_tta)
        df_out_tta = aggregate_to_images(df_val, logits_tta, CLASSES)
        df_out_tta.insert(1, "fold", val_fold)

        tta_path = fold_dir / "inf_val_predictions_tta.csv"
        df_out_tta.to_csv(tta_path, index=False)
        metrics_tta = compute_metrics(df_out_tta, CLASSES)
        delta_ll = metrics["log_loss"] - metrics_tta["log_loss"]
        print_metrics(metrics_tta, prefix="TTA: ")
        print(f"    TTA Δlog_loss: {delta_ll:+.4f}")
        print(f"    → {tta_path.name} "
              f"({len(df_out_tta)} images, {time.time() - t0:.0f}s)")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Batch helpers: discovery + threshold resolution
# ═════════════════════════════════════════════════════════════════════════════

def discover_folds(model_dir):
    """Find fold_N directories, sorted by fold number."""
    model_dir = Path(model_dir)
    folds = sorted(
        [f for f in model_dir.glob("fold_*")
         if f.is_dir() and (f / "best_model.pt").exists()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    return folds


def infer_threshold(model_dir_name, fold_dir=None):
    """Infer detection threshold from training config or directory name.

    Returns canonical threshold string: '.1', '.05', or 'full'.
    """
    # Try training config first
    if fold_dir is not None:
        cfg_path = Path(fold_dir) / "training_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            for key in ("detection_threshold", "detection_thresh",
                        "crop_threshold"):
                if key in cfg:
                    return _normalize_threshold(str(cfg[key]))

    # Fall back to directory name parsing
    name = model_dir_name.lower()
    if "_full" in name:
        return "full"
    # Check .05/05 before .1 to avoid partial match
    if "_.05" in name or "_05_" in name or "_05" in name:
        return ".05"
    if "_.1_" in name or "_.1" in name:
        return ".1"
    # Catch patterns like "eva02_1_folds"
    if re.search(r'[._]1(?:_folds)?$', name):
        return ".1"

    return None


def _normalize_threshold(t):
    t = t.strip().lower()
    mapping = {
        ".1": ".1", "0.1": ".1", "10": ".1",
        ".05": ".05", "0.05": ".05", "05": ".05",
        "full": "full", "none": "full",
    }
    return mapping.get(t, t)


def load_crop_map(crop_map_path):
    """Load and normalize crop_map.json."""
    with open(crop_map_path) as f:
        raw = json.load(f)
    return {_normalize_threshold(k): v for k, v in raw.items()}


def resolve_crop_dir(model_dir_name, fold_dir, crop_map):
    """Resolve crop directory via threshold inference + crop_map."""
    thresh = infer_threshold(model_dir_name, fold_dir)
    if thresh is None:
        return None, None
    crop_dir = crop_map.get(thresh)
    return crop_dir, thresh


# ═════════════════════════════════════════════════════════════════════════════
# Batch runners
# ═════════════════════════════════════════════════════════════════════════════

def predict_model(model_dir, crop_dir, data_dir, device,
                  tta=False, n_tta=5):
    """Run prediction on all folds within a model directory."""
    model_dir = Path(model_dir)
    fold_dirs = discover_folds(model_dir)

    if not fold_dirs:
        print(f"  No fold directories found in {model_dir}")
        return []

    print(f"\n{'═'*60}")
    print(f"Model: {model_dir.name}  ({len(fold_dirs)} folds)")
    print(f"Crops: {crop_dir}")
    print(f"{'═'*60}")

    all_metrics = []
    for fd in fold_dirs:
        m = predict_fold(fd, crop_dir, data_dir, device,
                         tta=tta, n_tta=n_tta)
        if m is not None:
            all_metrics.append(m)

    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics])
               for k in all_metrics[0]}
        std_vals = {k: np.std([m[k] for m in all_metrics])
                    for k in all_metrics[0]}
        print(f"  ── {model_dir.name} mean across {len(all_metrics)} folds ──")
        parts = [f"{k}={avg[k]:.4f} ±{std_vals[k]:.4f}" for k in avg]
        print(f"    {', '.join(parts)}")

    return all_metrics


def predict_all(models_dir, crop_map, data_dir, device,
                tta=False, n_tta=5):
    """Run prediction on all models in a models directory."""
    models_dir = Path(models_dir)
    model_dirs = sorted([
        d for d in models_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    ])

    if not model_dirs:
        print(f"No model directories found in {models_dir}")
        return

    print(f"Found {len(model_dirs)} model directories in {models_dir}\n")

    # ── Pre-flight: resolve all crop dirs before running ─────────────────
    plan = []
    for md in model_dirs:
        folds = discover_folds(md)
        first_fold = folds[0] if folds else None
        crop_dir, thresh = resolve_crop_dir(md.name, first_fold, crop_map)

        if crop_dir is None:
            print(f"  ⚠ {md.name}: cannot infer threshold. Skipping.")
            continue
        if not Path(crop_dir).exists():
            print(f"  ⚠ {md.name}: crop_dir '{crop_dir}' not found. Skipping.")
            continue
        if not folds:
            print(f"  ⚠ {md.name}: no fold dirs with best_model.pt. Skipping.")
            continue

        plan.append((md, crop_dir, thresh))
        print(f"  ✓ {md.name} → thresh={thresh} → {crop_dir} "
              f"({len(folds)} folds)")

    print(f"\nRunning {len(plan)} models...\n")

    summary = []
    for md, crop_dir, thresh in plan:
        metrics_list = predict_model(md, crop_dir, data_dir, device,
                                     tta=tta, n_tta=n_tta)
        if metrics_list:
            avg = {k: np.mean([m[k] for m in metrics_list])
                   for k in metrics_list[0]}
            avg["model"] = md.name
            avg["threshold"] = thresh
            summary.append(avg)

    # ── Final summary table ──────────────────────────────────────────────
    if summary:
        print(f"\n{'═'*60}")
        print("Summary (mean across folds)")
        print(f"{'═'*60}")
        header = (f"  {'model':<35} {'thresh':>6} "
                  f"{'log_loss':>9} {'f1':>7} {'acc':>7}")
        print(header)
        print(f"  {'─'*35} {'─'*6} {'─'*9} {'─'*7} {'─'*7}")
        for s in sorted(summary, key=lambda x: x["log_loss"]):
            print(f"  {s['model']:<35} {s['threshold']:>6} "
                  f"{s['log_loss']:>9.4f} "
                  f"{s['f1_macro']:>7.4f} "
                  f"{s['accuracy']:>7.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate fold-level validation predictions "
                    "(logits + probas, per-image)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode selection (mutually exclusive) ──
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold_dir", type=Path,
                      help="Single fold directory "
                           "(contains best_model.pt)")
    mode.add_argument("--model_dir", type=Path,
                      help="Model directory "
                           "(contains fold_0/ ... fold_4/)")
    mode.add_argument("--models_dir", type=Path,
                      help="Root models/ directory "
                           "(batch over all models)")

    # ── Data paths ──
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Competition data root "
                             "(has train_labels.csv, train_features.csv)")
    parser.add_argument("--crop_dir", type=Path, default=None,
                        help="Crop directory "
                             "(required for --fold_dir / --model_dir)")
    parser.add_argument("--crop_map", type=Path, default=None,
                        help="JSON: threshold → crop_dir "
                             "(required for --models_dir)")

    # ── TTA options ──
    parser.add_argument("--tta", action="store_true",
                        help="Also generate TTA predictions")
    parser.add_argument("--n_tta", type=int, default=5,
                        help="Number of TTA passes (default: 5)")

    args = parser.parse_args()

    # ── Validate ──
    if args.fold_dir or args.model_dir:
        if args.crop_dir is None:
            parser.error(
                "--crop_dir required with --fold_dir / --model_dir")
    if args.models_dir:
        if args.crop_map is None:
            parser.error("--crop_map required with --models_dir")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes ({NUM_CLASSES}): {CLASSES}")
    print(f"TTA: {'yes' if args.tta else 'no'}"
          + (f" ({args.n_tta} passes)" if args.tta else ""))
    print()

    # ── Dispatch ──
    if args.fold_dir:
        predict_fold(args.fold_dir, args.crop_dir, args.data_dir,
                     device, tta=args.tta, n_tta=args.n_tta)

    elif args.model_dir:
        predict_model(args.model_dir, args.crop_dir, args.data_dir,
                      device, tta=args.tta, n_tta=args.n_tta)

    elif args.models_dir:
        crop_map = load_crop_map(args.crop_map)
        predict_all(args.models_dir, crop_map, args.data_dir,
                    device, tta=args.tta, n_tta=args.n_tta)


if __name__ == "__main__":
    main()
