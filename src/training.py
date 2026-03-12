"""
src/training.py
===============
Training loop components: epoch runners, schedulers, freeze/unfreeze.
"""

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int], num_classes: int,
                          device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    counts = Counter(labels)
    total = sum(counts.values())
    w = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def cosine_warmup_scheduler(optimizer, warmup_steps: int,
                            total_steps: int):
    """Linear warmup then cosine decay to zero."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module, head_keyword: str = "head"):
    """Freeze all parameters except those whose name contains head_keyword."""
    for name, param in model.named_parameters():
        if head_keyword not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model: nn.Module, head_keyword: str,
                     lr_backbone: float, lr_head: float,
                     weight_decay: float):
    """Split model params into backbone/head groups with differential LR."""
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if head_keyword in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]


# ---------------------------------------------------------------------------
# Mixup / CutMix batch augmentation
# ---------------------------------------------------------------------------

class MixupCutmix:
    """
    Batch-level Mixup and CutMix augmentation.

    On each batch, randomly applies one of:
      - Mixup:  blends two images and their labels via a Beta-sampled lambda
      - CutMix: pastes a random patch from one image onto another, blends labels
                proportional to patch area
      - Neither: passes batch through unchanged

    Targets are converted to soft labels (one-hot with mixing) so the loss
    function must accept float targets — use soft_cross_entropy below.

    Config example in augmentation.json:
        "batch": {
            "mixup_alpha": 0.3,    # Beta distribution alpha for mixup
            "cutmix_alpha": 1.0,   # Beta distribution alpha for cutmix
            "mixup_prob": 0.5,     # probability of applying mixup
            "cutmix_prob": 0.5     # probability of applying cutmix
        }

    Probabilities are normalized: if both sum > 1, they're scaled so that
    the remaining probability mass is "do nothing".
    """

    def __init__(self, batch_cfg: dict, num_classes: int):
        self.mixup_alpha = batch_cfg.get("mixup_alpha", 0.3)
        self.cutmix_alpha = batch_cfg.get("cutmix_alpha", 1.0)
        self.num_classes = num_classes

        mp = batch_cfg.get("mixup_prob", 0.5)
        cp = batch_cfg.get("cutmix_prob", 0.5)
        total = mp + cp
        # Normalize so they define a proper distribution with "none" as remainder
        if total > 1.0:
            mp, cp = mp / total, cp / total
        self.mixup_prob = mp
        self.cutmix_prob = cp

    def __call__(self, imgs: torch.Tensor, tgts: torch.Tensor):
        """
        Args:
            imgs: (B, C, H, W) batch of images
            tgts: (B,) integer class indices

        Returns:
            mixed_imgs: (B, C, H, W)
            soft_tgts:  (B, num_classes) float target distribution
        """
        B = imgs.size(0)
        # Convert to one-hot
        soft_tgts = torch.zeros(B, self.num_classes, device=imgs.device)
        soft_tgts.scatter_(1, tgts.unsqueeze(1), 1.0)

        r = np.random.rand()
        if r < self.mixup_prob:
            return self._mixup(imgs, soft_tgts)
        elif r < self.mixup_prob + self.cutmix_prob:
            return self._cutmix(imgs, soft_tgts)
        else:
            return imgs, soft_tgts

    def _mixup(self, imgs, soft_tgts):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1.0 - lam)  # keep lam >= 0.5 so original dominates
        perm = torch.randperm(imgs.size(0), device=imgs.device)
        mixed = lam * imgs + (1.0 - lam) * imgs[perm]
        tgts_mixed = lam * soft_tgts + (1.0 - lam) * soft_tgts[perm]
        return mixed, tgts_mixed

    def _cutmix(self, imgs, soft_tgts):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        lam = max(lam, 1.0 - lam)
        B, C, H, W = imgs.shape
        perm = torch.randperm(B, device=imgs.device)

        # Random bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        mixed = imgs.clone()
        mixed[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]

        # Actual lambda based on pasted area
        area_ratio = (y2 - y1) * (x2 - x1) / (H * W)
        lam_actual = 1.0 - area_ratio
        tgts_mixed = lam_actual * soft_tgts + (1.0 - lam_actual) * soft_tgts[perm]
        return mixed, tgts_mixed


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor,
                       weight: torch.Tensor = None) -> torch.Tensor:
    """
    Cross-entropy loss for soft (float) targets.

    Standard CE with integer labels is a special case of this.
    Supports per-class weights for class imbalance.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    if weight is not None:
        # Weight each class's contribution
        log_probs = log_probs * weight.unsqueeze(0)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


def build_batch_aug(aug_cfg: dict, num_classes: int):
    """
    Build a MixupCutmix augmenter from augmentation config, or None if
    no batch augmentation is specified.
    """
    batch_cfg = aug_cfg.get("batch")
    if batch_cfg is None:
        return None
    return MixupCutmix(batch_cfg, num_classes)


# ---------------------------------------------------------------------------
# Epoch runners
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, scheduler,
                scaler, device, grad_clip_norm: float = 1.0,
                batch_aug: MixupCutmix = None,
                class_weights: torch.Tensor = None):
    """
    Run one training epoch. Returns (loss, accuracy).

    If batch_aug is provided, applies mixup/cutmix and uses soft CE loss
    instead of the standard criterion.
    """
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []

    for imgs, tgts in tqdm(loader, desc="  train", leave=False):
        imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)

        # Save original hard labels for accuracy tracking
        hard_tgts = tgts.clone()

        # Apply batch-level augmentation (mixup/cutmix)
        if batch_aug is not None:
            imgs, soft_tgts = batch_aug(imgs, tgts)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
            if batch_aug is not None:
                loss = soft_cross_entropy(logits, soft_tgts, class_weights)
            else:
                loss = criterion(logits, tgts)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        preds_all.extend(logits.detach().argmax(1).cpu().tolist())
        labels_all.extend(hard_tgts.cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    """
    Run one validation epoch.
    Returns (loss, accuracy, pred_indices, label_indices, prob_matrix).
    """
    model.eval()
    total_loss = 0.0
    preds_all, labels_all, probs_all = [], [], []

    for imgs, tgts in tqdm(loader, desc="  val  ", leave=False):
        imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, tgts)

        total_loss += loss.item() * imgs.size(0)
        preds_all.extend(logits.argmax(1).cpu().tolist())
        labels_all.extend(tgts.cpu().tolist())
        probs_all.append(torch.softmax(logits, 1).cpu().numpy())

    return (total_loss / len(loader.dataset),
            accuracy_score(labels_all, preds_all),
            preds_all, labels_all, np.concatenate(probs_all))
