"""
scripts/02_train_ovr.py
=======================
One-vs-rest binary training: target_class vs everything else.

Reuses the full training pipeline (same backbone, augmentation, transforms,
data loading, MegaDetector crops) but with binary labels. The model learns
to distinguish a single target class from all others.

The output probability P(target_class) can be blended into the multi-class
ensemble to improve calibration for hard classes.

Config changes vs multi-class:
    "ovr_target": "blank"       ← new: which class to isolate
    "num_classes": 2            ← override (automatic if omitted)

Everything else (backbone, img_size, augmentation, etc.) works identically.

Usage:
    python scripts/02_train_ovr.py \
        --model_config configs/ovr_blank_dinov2.json \
        --data_dir data/competition \
        --crop_dir data/competition/crops_.1 \
        --val_fold 0

    # Works with 03_kfold_runner.py — just add to train_jobs.json:
    # {"label": "ovr_blank_dinov2", "config": "configs/ovr_blank_dinov2.json",
    #  "crop_dir": "data/competition/crops_.1"}
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

from src.config import (
    load_model_config, load_augmentation,
    cls2idx as make_cls2idx, idx2cls as make_idx2cls, get_classes,
)
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


# ═════════════════════════════════════════════════════════════════════════════
# OVR label remapping
# ═════════════════════════════════════════════════════════════════════════════

def remap_to_binary(df, target_class, original_classes):
    """Remap multi-class labels to binary: target_class=1, rest=0.

    Modifies df in-place: overwrites 'label' and 'label_idx' columns.
    Returns the binary class list ['rest', target_class].
    """
    binary_classes = ["rest", target_class]
    df["label"] = df["label"].apply(
        lambda x: target_class if x == target_class else "rest")
    df["label_idx"] = (df["label"] == target_class).astype(int)

    n_target = (df["label_idx"] == 1).sum()
    n_rest = (df["label_idx"] == 0).sum()
    ratio = n_rest / max(n_target, 1)
    print(f"\n  OVR: {target_class} vs rest")
    print(f"  Target: {n_target} ({n_target/len(df)*100:.1f}%)")
    print(f"  Rest:   {n_rest} ({n_rest/len(df)*100:.1f}%)")
    print(f"  Ratio:  1:{ratio:.1f}")

    return binary_classes


# ═════════════════════════════════════════════════════════════════════════════
# Main training function
# ═════════════════════════════════════════════════════════════════════════════

def train(model_config_path: str, data_dir: Path, output_dir: Path | None = None,
          run_name: str | None = None, quick: bool = False,
          predict_test: bool = True, val_fold: int | None = None,
          crop_dir_override: Path | None = None):

    cfg = load_model_config(model_config_path)

    # ── OVR config ───────────────────────────────────────────────────────
    target_class = cfg.get("ovr_target")
    if target_class is None:
        raise ValueError("Config must include 'ovr_target' field "
                         "(e.g., 'blank', 'leopard', 'rodent')")

    original_classes = get_classes()
    if target_class not in original_classes:
        raise ValueError(f"ovr_target '{target_class}' not in classes: "
                         f"{original_classes}")

    # Override to binary
    binary_classes = ["rest", target_class]
    cfg["classes"] = binary_classes
    cfg["num_classes"] = 2

    if val_fold is not None:
        cfg["val_fold"] = val_fold

    _cls2idx = make_cls2idx(binary_classes)
    _idx2cls = make_idx2cls(binary_classes)
    num_classes = 2

    # Resolve paths
    crop_dir = crop_dir_override or (data_dir / "full_images")
    if output_dir is None:
        output_dir = data_dir / "models" / cfg["model_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        cfg["epochs_frozen"] = 1
        cfg["epochs_unfrozen"] = 2
        cfg["patience"] = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {cfg['model_name']} ({cfg['backbone']})")
    print(f"OVR target: {target_class}")
    print(f"Image size: {cfg['img_size']}, Batch size: {cfg['batch_size']}")

    # ── Data: load with original classes, then remap ─────────────────────
    orig_cls2idx = make_cls2idx(original_classes)
    df = load_train_data(data_dir, crop_dir, original_classes, orig_cls2idx)
    crop_train_dir = str(crop_dir / "train")

    # Split BEFORE remapping (preserves site-aware fold structure)
    df_train, df_val = get_site_split(df, cfg["val_fold"], cfg["n_folds"])

    # Remap to binary
    print("\n── Train split ──")
    remap_to_binary(df_train, target_class, original_classes)
    print("\n── Val split ──")
    remap_to_binary(df_val, target_class, original_classes)

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

    # ── Model (2-class head) ─────────────────────────────────────────────
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

    batch_aug = build_batch_aug(aug_cfg, num_classes)

    cfg["class_weights"] = {binary_classes[i]: round(cw[i].item(), 4)
                            for i in range(num_classes)}
    cfg["train_samples"] = len(df_train)
    cfg["val_samples"] = len(df_val)
    cfg["ovr_target"] = target_class
    cfg["original_classes"] = original_classes

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_active = mlflow_utils.setup_run(
        "conservision_ovr",
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
            {"train_loss": tl, "train_acc": ta,
             "val_loss": vl, "val_acc": va, "lr": lr},
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
            {"train_loss": tl, "train_acc": ta,
             "val_loss": vl, "val_acc": va, "lr": lr},
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
            print(f"  Early stopping (no improvement for "
                  f"{cfg['patience']} epochs)")
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

    val_pred_df = build_predictions_df(
        df_val, vprobs, binary_classes,
        split_name="val", include_labels=True)

    print(f"\nVal Accuracy: {va:.4f} ({va*100:.1f}%)")
    report_str = print_classification_report(val_pred_df, binary_classes)

    # ── Additional OVR-specific metrics ──────────────────────────────────
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, average_precision_score,
    )
    binary_true = (val_pred_df["true_class"] == target_class).astype(int)
    target_prob_col = f"{target_class}_prob"
    if target_prob_col in val_pred_df.columns:
        target_probs = val_pred_df[target_prob_col].values
        auc = roc_auc_score(binary_true, target_probs)
        ap = average_precision_score(binary_true, target_probs)
        print(f"\n  OVR metrics for '{target_class}':")
        print(f"    ROC AUC:            {auc:.4f}")
        print(f"    Average Precision:  {ap:.4f}")
        cfg["ovr_roc_auc"] = round(auc, 4)
        cfg["ovr_avg_precision"] = round(ap, 4)

    # ── Save all outputs ──────────────────────────────────────────────────
    cfg["best_epoch"] = best_epoch
    cfg["best_val_loss"] = round(best_val_loss, 6)
    cfg["training_time_seconds"] = round(training_time, 1)
    cfg["history"] = history

    save_model_outputs(
        output_dir=output_dir,
        pred_df=val_pred_df,
        classes=binary_classes,
        config=cfg,
        history=history,
        epochs_frozen=cfg["epochs_frozen"],
        report_str=report_str,
        script_path=__file__,
    )

    # MLflow final
    if mlflow_active:
        from src.evaluation import compute_metrics
        metrics = compute_metrics(val_pred_df, binary_classes)
        mlflow_utils.log_metrics({
            "best_val_acc": metrics["accuracy"],
            "best_val_loss": best_val_loss,
            "val_f1_macro": metrics["f1_macro"],
            "val_log_loss": metrics["log_loss"],
            "training_time_s": training_time,
        })
        if "ovr_roc_auc" in cfg:
            mlflow_utils.log_metrics({
                "ovr_roc_auc": cfg["ovr_roc_auc"],
                "ovr_avg_precision": cfg["ovr_avg_precision"],
            })
        for art in output_dir.iterdir():
            if art.is_file():
                mlflow_utils.log_artifact(str(art))
        mlflow_utils.end_run()

    # ══════════════════════════════════════════════════════════════════════
    # Test set predictions
    # ══════════════════════════════════════════════════════════════════════
    if predict_test:
        print(f"\n{'─'*60}")
        print("Generating test predictions")
        print(f"{'─'*60}")

        test_meta = load_test_crop_metadata(crop_dir)
        if len(test_meta) > 0:
            crop_test_dir = str(crop_dir / "test")
            test_meta = test_meta.copy()
            test_meta["label_idx"] = 0

            test_dataset = CropInferenceDataset(
                test_meta, crop_test_dir, val_tfm, cfg["img_size"])
            test_loader = DataLoader(
                test_dataset, batch_size=cfg["batch_size"], shuffle=False,
                num_workers=cfg["num_workers"], pin_memory=True)

            test_probs = predict_batch(model, test_loader, device)

            test_pred_df = build_predictions_df(
                test_meta, test_probs, binary_classes,
                split_name="test", include_labels=False)

            test_pred_path = output_dir / "predictions_test.csv"
            test_pred_df.to_csv(test_pred_path, index=False)
            print(f"Test predictions -> {test_pred_path} "
                  f"({len(test_pred_df)} images)")
        else:
            print("No test crops found, skipping.")

    print(f"\nTraining time: {training_time/60:.1f} min")
    print(f"Done! Model outputs -> {output_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI — identical interface to 02_train.py
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OVR (one-vs-rest) binary classifier")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--val_fold", type=int, default=None)
    parser.add_argument("--crop_dir", type=str, default=None)
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
