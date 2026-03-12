"""
src/transforms.py
=================
Build torchvision transform pipelines from augmentation config dicts.
"""

from torchvision import transforms


def build_train_transform(aug_cfg: dict, img_size: int,
                          mean: list, std: list) -> transforms.Compose:
    """Build a training transform from an augmentation config dict."""
    t = aug_cfg["train"]
    ops = [
        transforms.RandomResizedCrop(
            img_size,
            scale=tuple(t["random_resized_crop_scale"]),
            ratio=tuple(t["random_resized_crop_ratio"]),
        ),
    ]
    if t.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())

    cj = t.get("color_jitter")
    if cj:
        ops.append(transforms.ColorJitter(**cj))

    gs_p = t.get("random_grayscale_p", 0)
    if gs_p > 0:
        ops.append(transforms.RandomGrayscale(p=gs_p))

    rot = t.get("rotation_degrees", 0)
    if rot > 0:
        ops.append(transforms.RandomRotation(rot))

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    re_p = t.get("random_erasing_p", 0)
    if re_p > 0:
        ops.append(transforms.RandomErasing(p=re_p))

    return transforms.Compose(ops)


def build_val_transform(aug_cfg: dict, img_size: int,
                        mean: list, std: list) -> transforms.Compose:
    """Build a validation/test transform from augmentation config."""
    v = aug_cfg["val"]
    resize_factor = v.get("resize_factor", 1.14)
    return transforms.Compose([
        transforms.Resize(int(img_size * resize_factor)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_tta_transforms(tta_passes: list, img_size: int,
                         mean: list, std: list) -> list[transforms.Compose]:
    """Build a list of TTA transform pipelines."""
    base_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    result = []
    for p in tta_passes:
        resize_factor = p.get("resize_factor", 1.14)
        ops = [
            transforms.Resize(int(img_size * resize_factor)),
            transforms.CenterCrop(img_size),
        ]
        if p.get("hflip", False):
            ops.append(transforms.RandomHorizontalFlip(p=1.0))
        ops.append(base_norm)
        result.append(transforms.Compose(ops))
    return result
