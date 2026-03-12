"""
scripts/06_aggregate_folds.py
=============================
Aggregate fold-level val_predictions.csv into a single model-level OOF file.

Each fold's val_predictions.csv covers a disjoint subset of images.
Concatenating all 5 gives full training-set coverage with unbiased
(out-of-fold) predictions — exactly what the ensemble step needs.

Output per model_dir:
    oof_predictions.csv   — all folds concatenated (same columns as fold files)
    oof_summary.json      — model + per-fold metrics

Usage:
    python scripts/06_aggregate_folds.py --model_dir models/dinov2_.1_folds
    python scripts/06_aggregate_folds.py --models_dir models/
    python scripts/06_aggregate_folds.py --models_dir models/ \
        --pred_file val_predictions_tta.csv --out_prefix oof_tta
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, f1_score, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_classes, cls2idx as make_cls2idx

CLASSES = get_classes()
CLS2IDX = make_cls2idx(CLASSES)
SKIP_DIRS = {"archive", "__pycache__", ".git", ".ipynb_checkpoints"}


def compute_metrics(df, classes=CLASSES):
    """Log loss, F1 macro, accuracy, per-class F1."""
    true_idx = df["true_label"].map(CLS2IDX).values
    prob_cols = [f"{c}_prob" for c in classes]
    probas = np.clip(df[prob_cols].values, 1e-7, 1.0 - 1e-7)
    probas = probas / probas.sum(axis=1, keepdims=True)
    pred_idx = df[prob_cols].values.argmax(axis=1)

    f1_per = f1_score(true_idx, pred_idx, average=None, zero_division=0)
    return {
        "log_loss": round(float(log_loss(
            true_idx, probas, labels=list(range(len(classes))))), 6),
        "f1_macro": round(float(f1_score(
            true_idx, pred_idx, average="macro", zero_division=0)), 6),
        "accuracy": round(float(accuracy_score(true_idx, pred_idx)), 6),
        "f1_per_class": {c: round(float(f1_per[i]), 4)
                         for i, c in enumerate(classes)},
    }


def aggregate_model(model_dir, pred_file="val_predictions.csv",
                    out_prefix="oof"):
    """Concatenate fold predictions, compute metrics, save OOF file."""
    model_dir = Path(model_dir)
    fold_dirs = sorted(
        [f for f in model_dir.glob("fold_*")
         if f.is_dir() and (f / pred_file).exists()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not fold_dirs:
        print(f"  SKIP {model_dir.name} — no folds with {pred_file}")
        return None

    dfs = []
    fold_metrics = {}
    for fd in fold_dirs:
        df = pd.read_csv(fd / pred_file)
        fold_num = int(fd.name.split("_")[-1])
        dfs.append(df)
        m = compute_metrics(df)
        fold_metrics[f"fold_{fold_num}"] = m
        print(f"    fold_{fold_num}: {len(df):>5} images  "
              f"ll={m['log_loss']:.4f}  f1={m['f1_macro']:.4f}  "
              f"acc={m['accuracy']:.4f}")

    oof = pd.concat(dfs, ignore_index=True)
    dupes = oof["id"].duplicated().sum()
    if dupes > 0:
        print(f"  ⚠ {dupes} duplicate ids — possible fold overlap")

    oof_metrics = compute_metrics(oof)

    # Read model metadata from first fold config
    meta = {}
    cfg_path = fold_dirs[0] / "training_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        meta = {k: cfg.get(k, "") for k in
                ("backbone", "img_size", "augmentation")}

    summary = {
        "model": model_dir.name,
        **meta,
        "n_folds": len(fold_dirs),
        "n_images": len(oof),
        "metrics": oof_metrics,
        "fold_metrics": fold_metrics,
    }

    oof_path = model_dir / f"{out_prefix}_predictions.csv"
    oof.to_csv(oof_path, index=False)
    summary_path = model_dir / f"{out_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  OOF: {len(oof)} images  ll={oof_metrics['log_loss']:.4f}  "
          f"f1={oof_metrics['f1_macro']:.4f}  acc={oof_metrics['accuracy']:.4f}")
    print(f"  → {oof_path.name}, {summary_path.name}")
    return summary


def aggregate_all(models_dir, pred_file, out_prefix):
    models_dir = Path(models_dir)
    dirs = sorted(d for d in models_dir.iterdir()
                  if d.is_dir() and d.name not in SKIP_DIRS)

    summaries = []
    for md in dirs:
        has = any((md / f"fold_{i}" / pred_file).exists() for i in range(10))
        if not has:
            continue
        print(f"\n{md.name}")
        print(f"{'─'*55}")
        s = aggregate_model(md, pred_file, out_prefix)
        if s:
            summaries.append(s)

    if summaries:
        print(f"\n\n{'═'*65}")
        print(f"{'model':<35} {'images':>6} {'log_loss':>9} "
              f"{'f1':>7} {'acc':>7}")
        print(f"{'─'*35} {'─'*6} {'─'*9} {'─'*7} {'─'*7}")
        for s in sorted(summaries, key=lambda x: x["metrics"]["log_loss"]):
            m = s["metrics"]
            print(f"{s['model']:<35} {s['n_images']:>6} "
                  f"{m['log_loss']:>9.4f} {m['f1_macro']:>7.4f} "
                  f"{m['accuracy']:>7.4f}")

        out = models_dir / f"all_{out_prefix}_summary.json"
        with open(out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\n→ {out}")


def main():
    p = argparse.ArgumentParser(
        description="Aggregate fold predictions into model-level OOF")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model_dir", type=Path)
    mode.add_argument("--models_dir", type=Path)
    p.add_argument("--pred_file", default="val_predictions.csv")
    p.add_argument("--out_prefix", default="oof")
    args = p.parse_args()

    if args.model_dir:
        print(f"{args.model_dir.name}\n{'─'*55}")
        aggregate_model(args.model_dir, args.pred_file, args.out_prefix)
    else:
        aggregate_all(args.models_dir, args.pred_file, args.out_prefix)


if __name__ == "__main__":
    main()
