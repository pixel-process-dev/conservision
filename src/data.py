"""
src/data.py
===========
Data loading and split utilities shared across training and evaluation.
"""

import pandas as pd
from sklearn.model_selection import GroupKFold


def load_train_data(data_dir, crop_dir, classes: list[str],
                    cls2idx: dict) -> pd.DataFrame:
    """
    Join crop metadata + labels + site info into a training DataFrame.

    Returns DataFrame with columns including:
        original_id, crop_filename, label, label_idx, site, split, etc.
    """
    meta = pd.read_csv(crop_dir / "crop_metadata.csv")
    labels = pd.read_csv(data_dir / "train_labels.csv")
    features = pd.read_csv(data_dir / "train_features.csv")

    target_cols = [c for c in classes if c in labels.columns]
    labels["label"] = labels[target_cols].idxmax(axis=1)
    labels["label_idx"] = labels["label"].map(cls2idx)

    df = meta[meta["split"] == "train"].copy()
    df = df.merge(labels[["id", "label", "label_idx"]],
                  left_on="original_id", right_on="id", how="inner")
    df = df.merge(features[["id", "site"]],
                  left_on="original_id", right_on="id",
                  how="left", suffixes=("", "_f"))

    print(f"Training crops: {len(df)} from {df['original_id'].nunique()} images")
    print(f"Sites: {df['site'].nunique()}")
    for cls in classes:
        n = (df["label"] == cls).sum()
        print(f"  {cls}: {n} ({n / len(df) * 100:.1f}%)")

    return df


def get_site_split(df: pd.DataFrame, val_fold: int = 0,
                   n_folds: int = 5):
    """
    Site-aware GroupKFold split. Returns (df_train, df_val).
    """
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(df, df["label_idx"], df["site"]))
    train_idx, val_idx = splits[val_fold]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    print(f"\nFold {val_fold}: train={len(df_train)}, val={len(df_val)}")
    print(f"Val sites: {df_val['site'].nunique()} (overlap with train: "
          f"{len(set(df_val['site']) & set(df_train['site']))})")

    return df_train, df_val


def load_test_crop_metadata(crop_dir) -> pd.DataFrame:
    """Load test split from crop metadata."""
    meta = pd.read_csv(crop_dir / "crop_metadata.csv")
    return meta[meta["split"] == "test"].copy()
