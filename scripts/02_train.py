"""
03_train.py
===========
Unified config-driven training script for the Conser-vision competition.

Replaces the individual 03_train_classifier.py, 03_train_dinov2.py, etc.
All model-specific settings live in configs/model_*.json files.

Training pipeline:
  1. Load experiment + model config
  2. Build data loaders with config-specified augmentation
  3. Phase 1: frozen backbone, train head only
  4. Phase 2: unfreeze all, fine-tune with differential LR
  5. Early stopping on val loss
  6. Final evaluation on best model
  7. Save standardized outputs to models/<model_name>/

Standardized outputs per model:
  - best_model.pt              (model weights)
  - training_config.json       (full config + metrics + history)
  - classification_report.txt  (sklearn text report)
  - confusion_matrix.png       (annotated heatmap)
  - training_curves.png        (loss + accuracy curves)
  - predictions_val.csv        (wide-format probabilities per image)
  - predictions_test.csv       (wide-format probabilities for test set)
  - <script copy>              (this script)

Usage:
    python 03_train.py --model_config configs/model_dinov2_v1.json --data_dir data/competition
    python 03_train.py --model_config configs/model_effnetv2s.json --data_dir data/competition
    python 03_train.py --model_config configs/model_dinov2_v2.json --data_dir data/competition --quick

Hardware target: RTX 2060 6GB (batch sizes and img_size set per config)
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_model_config, load_augmentation, cls2idx, idx2cls
from src.transforms import build_train_transform, build_val_transform
from src.dataset import CropDataset, CropInferenceDataset
from src.data import load_train_data, get_site_split, load_test_crop_metadata
from src.training import (
    compute_class_weights, cosine_warmup_scheduler,
    freeze_backbone, unfreeze_all, get_param_groups,
    train_epoch, val_epoch, build_batch_aug,
)
from src.evaluation import (
    build_predictions_df, save_model_outputs, print_classification_report,
)
from src.inference import predict_batch
from src import mlflow_utils


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(model_config_path: str, data_dir: Path, output_dir: Path | None = None,
          run_name: str | None = None, quick: bool = False,
          predict_test: bool = True, val_fold: int | None = None,
          crop_dir_override: Path | None = None):

    cfg = load_model_config(model_config_path)

    # CLI overrides (for k-fold CV without editing config files)
    if val_fold is not None:
        cfg["val_fold"] = val_fold

    classes = cfg["classes"]
    _cls2idx = cls2idx(classes)
    _idx2cls = idx2cls(classes)
    num_classes = cfg["num_classes"]

    # Resolve paths
    crop_dir = crop_dir_override or (data_dir / "full_images")
    if output_dir is None:
        output_dir = data_dir / "models" / cfg["model_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode overrides
    if quick:
        cfg["epochs_frozen"] = 1
        cfg["epochs_unfrozen"] = 2
        cfg["patience"] = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {cfg['model_name']} ({cfg['backbone']})")
    print(f"Image size: {cfg['img_size']}, Batch size: {cfg['batch_size']}")

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_train_data(data_dir, crop_dir, classes, _cls2idx)
    crop_train_dir = str(crop_dir / "train")

    df_train, df_val = get_site_split(df, cfg["val_fold"], cfg["n_folds"])

    if quick:
        df_train = df_train.sample(min(500, len(df_train)), random_state=42)
        df_val = df_val.sample(min(200, len(df_val)), random_state=42)
        print(f"QUICK MODE: train={len(df_train)}, val={len(df_val)}")

    # ── Transforms ────────────────────────────────────────────────────────
    aug_cfg = load_augmentation(cfg["augmentation"])
    mean, std = cfg["imagenet_mean"], cfg["imagenet_std"]
    train_tfm = build_train_transform(aug_cfg, cfg["img_size"], mean, std)
    val_tfm = build_val_transform(aug_cfg, cfg["img_size"], mean, std)

    train_loader = DataLoader(
        CropDataset(df_train, crop_train_dir, train_tfm, cfg["img_size"]),
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        CropDataset(df_val, crop_train_dir, val_tfm, cfg["img_size"]),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    create_kwargs = {
        "pretrained": True,
        "num_classes": num_classes,
        **cfg.get("timm_create_kwargs", {}),
    }
    if cfg.get("drop_rate", 0) > 0:
        create_kwargs["drop_rate"] = cfg["drop_rate"]

    model = timm.create_model(cfg["backbone"], **create_kwargs)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    cfg["total_params"] = total_params

    cw = compute_class_weights(df_train["label_idx"].tolist(), num_classes, device)
    criterion = nn.CrossEntropyLoss(
        weight=cw, label_smoothing=cfg.get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Batch-level augmentation (mixup/cutmix) — None if not in aug config
    batch_aug = build_batch_aug(aug_cfg, num_classes)

    cfg["class_weights"] = {classes[i]: round(cw[i].item(), 4) for i in range(num_classes)}
    cfg["train_samples"] = len(df_train)
    cfg["val_samples"] = len(df_val)

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_active = mlflow_utils.setup_run(
        "conservision",
        run_name or f"{cfg['model_name']}_{time.strftime('%m%d_%H%M')}",
        cfg,
    )

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0
    total_epochs = cfg["epochs_frozen"] + cfg["epochs_unfrozen"]
    head_kw = cfg["head_keyword"]
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Frozen backbone — head only
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Phase 1: head only ({cfg['epochs_frozen']} epochs)")
    print(f"{'─'*60}")

    freeze_backbone(model, head_kw)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total_params:,} "
          f"({trainable/total_params*100:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg["epochs_frozen"] * steps_per_epoch)

    for epoch in range(cfg["epochs_frozen"]):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, criterion, optimizer,
                             scheduler, scaler, device, cfg["grad_clip_norm"],
                             batch_aug=batch_aug, class_weights=cw)
        vl, va, _, _, _ = val_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["lr"].append(lr)

        mlflow_utils.log_metrics(
            {"train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va, "lr": lr},
            step=epoch + 1)

        marker = ""
        if vl < best_val_loss:
            best_val_loss, best_epoch = vl, epoch + 1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            marker = " ✓"

        print(f"  [{epoch+1}/{total_epochs}] "
              f"loss={tl:.4f}/{vl:.4f}  acc={ta:.3f}/{va:.3f}  "
              f"lr={lr:.6f}  {dt:.0f}s{marker}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Full fine-tune with differential LR
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Phase 2: full fine-tune ({cfg['epochs_unfrozen']} epochs)")
    print(f"{'─'*60}")

    unfreeze_all(model)
    print(f"Trainable: {total_params:,} / {total_params:,} (100%)")

    param_groups = get_param_groups(
        model, head_kw, cfg["lr_backbone"],
        cfg["lr_head"] * 0.1, cfg["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    scheduler = cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg["epochs_unfrozen"] * steps_per_epoch)
    no_improve = 0

    for epoch in range(cfg["epochs_unfrozen"]):
        global_epoch = cfg["epochs_frozen"] + epoch + 1
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, criterion, optimizer,
                             scheduler, scaler, device, cfg["grad_clip_norm"],
                             batch_aug=batch_aug, class_weights=cw)
        vl, va, vp, vl_labels, vprobs = val_epoch(
            model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["lr"].append(lr)

        mlflow_utils.log_metrics(
            {"train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va, "lr": lr},
            step=global_epoch)

        marker = ""
        if vl < best_val_loss:
            best_val_loss, best_epoch = vl, global_epoch
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            marker = " ✓"
        else:
            no_improve += 1

        print(f"  [{global_epoch}/{total_epochs}] "
              f"loss={tl:.4f}/{vl:.4f}  acc={ta:.3f}/{va:.3f}  "
              f"lr={lr:.6f}  {dt:.0f}s{marker}")

        if no_improve >= cfg["patience"]:
            print(f"  Early stopping (no improvement for {cfg['patience']} epochs)")
            break

    training_time = time.time() - t_start

    # ══════════════════════════════════════════════════════════════════════
    # Final evaluation with best model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Final eval — best model from epoch {best_epoch}")
    print(f"{'─'*60}")

    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", weights_only=True))
    vl, va, vp, vl_labels, vprobs = val_epoch(
        model, val_loader, criterion, device)

    # Build standardized predictions DataFrame
    val_pred_df = build_predictions_df(
        df_val, vprobs, classes, split_name="val", include_labels=True)

    print(f"\nVal Accuracy: {va:.4f} ({va*100:.1f}%)")
    report_str = print_classification_report(val_pred_df, classes)

    # ── Save all outputs ──────────────────────────────────────────────────
    cfg["best_epoch"] = best_epoch
    cfg["best_val_loss"] = round(best_val_loss, 6)
    cfg["training_time_seconds"] = round(training_time, 1)
    cfg["history"] = history

    save_model_outputs(
        output_dir=output_dir,
        pred_df=val_pred_df,
        classes=classes,
        config=cfg,
        history=history,
        epochs_frozen=cfg["epochs_frozen"],
        report_str=report_str,
        script_path=__file__,
    )

    # MLflow final
    if mlflow_active:
        from src.evaluation import compute_metrics
        metrics = compute_metrics(val_pred_df, classes)
        mlflow_utils.log_metrics({
            "best_val_acc": metrics["accuracy"],
            "best_val_loss": best_val_loss,
            "val_f1_macro": metrics["f1_macro"],
            "val_log_loss": metrics["log_loss"],
            "training_time_s": training_time,
        })
        for cls, f1 in metrics["f1_per_class"].items():
            mlflow_utils.log_metrics({f"f1_{cls}": f1})
        for art in (output_dir).iterdir():
            if art.is_file():
                mlflow_utils.log_artifact(str(art))
        mlflow_utils.end_run()

    # ══════════════════════════════════════════════════════════════════════
    # Test set predictions (no labels, just probabilities for ensembling)
    # ══════════════════════════════════════════════════════════════════════
    if predict_test:
        print(f"\n{'─'*60}")
        print("Generating test predictions")
        print(f"{'─'*60}")

        test_meta = load_test_crop_metadata(crop_dir)
        if len(test_meta) > 0:
            crop_test_dir = str(crop_dir / "test")

            # Need a label_idx column for CropInferenceDataset compatibility
            test_meta = test_meta.copy()
            test_meta["label_idx"] = 0  # dummy

            test_dataset = CropInferenceDataset(
                test_meta, crop_test_dir, val_tfm, cfg["img_size"])
            test_loader = DataLoader(
                test_dataset, batch_size=cfg["batch_size"], shuffle=False,
                num_workers=cfg["num_workers"], pin_memory=True)

            test_probs = predict_batch(model, test_loader, device)

            test_pred_df = build_predictions_df(
                test_meta, test_probs, classes,
                split_name="test", include_labels=False)

            test_pred_path = output_dir / "predictions_test.csv"
            test_pred_df.to_csv(test_pred_path, index=False)
            print(f"Test predictions -> {test_pred_path} ({len(test_pred_df)} images)")
        else:
            print("No test crops found, skipping test predictions.")

    print(f"\nTraining time: {training_time/60:.1f} min")
    print(f"Done! Model outputs -> {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classifier from a model config file")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to model config JSON (e.g., configs/model_dinov2_v1.json)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--run_name", type=str, default=None,
                        help="MLflow run name")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: minimal epochs for testing")
    parser.add_argument("--no_test", action="store_true",
                        help="Skip test set predictions")
    parser.add_argument("--val_fold", type=int, default=None,
                        help="Override val fold (0-4) for k-fold CV")
    parser.add_argument("--crop_dir", type=str, default=None,
                        help="Override crop directory (for different detection thresholds)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    crop_dir = Path(args.crop_dir).resolve() if args.crop_dir else None

    train(
        model_config_path=args.model_config,
        data_dir=data_dir,
        output_dir=output_dir,
        run_name=args.run_name,
        quick=args.quick,
        predict_test=not args.no_test,
        val_fold=args.val_fold,
        crop_dir_override=crop_dir,
    )
