"""
src/inference.py
================
Model inference: single-pass, TTA, and multi-crop aggregation.

Used by training (val predictions), predict/submit, and ensemble scripts.
"""

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CropInferenceDataset
from .transforms import build_val_transform, build_tta_transforms
from .config import load_augmentation


@torch.no_grad()
def predict_batch(model, loader, device) -> np.ndarray:
    """Run inference on a DataLoader. Returns (N, C) probability matrix."""
    model.eval()
    all_probs = []

    for images, _ in tqdm(loader, desc="  Predicting", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def predict_with_tta(model, df, crop_dir, img_size, n_tta,
                     batch_size, num_workers, device,
                     mean, std) -> np.ndarray:
    """Run TTA: average predictions across augmented views."""
    aug_cfg = load_augmentation("tta")
    tta_tfms = build_tta_transforms(
        aug_cfg["passes"][:n_tta], img_size, mean, std)
    all_probs = None

    for i, tfm in enumerate(tta_tfms):
        print(f"  TTA pass {i + 1}/{len(tta_tfms)}")
        dataset = CropInferenceDataset(df, crop_dir, transform=tfm,
                                       img_size=img_size)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        probs = predict_batch(model, loader, device)

        if all_probs is None:
            all_probs = probs
        else:
            all_probs += probs

    return all_probs / len(tta_tfms)


def load_trained_model(model_dir, device, num_classes: int):
    """Load a trained model from its output directory."""
    import json
    config_path = model_dir / "training_config.json"
    with open(config_path) as f:
        config = json.load(f)

    backbone = config["backbone"]
    img_size = config["img_size"]

    create_kwargs = {"pretrained": False, "num_classes": num_classes}
    # ViT models need img_size override
    if "dinov2" in backbone or "vit" in backbone.lower():
        create_kwargs["img_size"] = img_size

    model = timm.create_model(backbone, **create_kwargs)
    model.load_state_dict(
        torch.load(model_dir / "best_model.pt",
                    weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    return model, config


def get_model_probabilities(model_dir, df_val, crop_dir, device,
                            num_classes: int,
                            mean: list, std: list) -> tuple[dict, dict]:
    """
    Load a trained model and compute per-image probability dicts.
    Returns (image_probs_dict, config_dict).
    """
    model, config = load_trained_model(model_dir, device, num_classes)
    img_size = config["img_size"]
    batch_size = config.get("batch_size", 16)

    aug_cfg = {"val": {"resize_factor": 1.14, "center_crop": True}}
    val_tfm = build_val_transform(aug_cfg, img_size, mean, std)

    dataset = CropInferenceDataset(
        df_val, str(crop_dir / "train"), transform=val_tfm, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    probs = predict_batch(model, loader, device)

    # Aggregate crops -> per-image (confidence-weighted mean)
    image_probs = {}
    for i in range(len(df_val)):
        img_id = df_val.iloc[i]["original_id"]
        conf = probs[i].max()
        if img_id not in image_probs:
            image_probs[img_id] = {"probs_sum": probs[i] * conf,
                                    "weight_sum": conf}
        else:
            image_probs[img_id]["probs_sum"] += probs[i] * conf
            image_probs[img_id]["weight_sum"] += conf

    result = {}
    for img_id, data in image_probs.items():
        result[img_id] = data["probs_sum"] / data["weight_sum"]

    print(f"  -> {len(result)} image predictions from {model_dir.name}")
    return result, config
