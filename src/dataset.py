"""
src/dataset.py
==============
PyTorch Datasets for crop-based classification.
"""

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CropDataset(Dataset):
    """
    Loads cropped images for training/validation.
    Returns (image_tensor, label_idx).
    """

    def __init__(self, df: pd.DataFrame, crop_dir: str,
                 transform=None, img_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.crop_dir = crop_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.crop_dir, row["crop_filename"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size))
        if self.transform:
            img = self.transform(img)
        return img, row["label_idx"]


class CropInferenceDataset(Dataset):
    """
    Loads cropped images for inference (no labels needed).
    Returns (image_tensor, row_index).
    """

    def __init__(self, df: pd.DataFrame, crop_dir: str,
                 transform=None, img_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.crop_dir = crop_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.crop_dir, row["crop_filename"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size))
        if self.transform:
            img = self.transform(img)
        return img, idx
