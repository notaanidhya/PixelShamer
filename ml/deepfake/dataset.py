"""
ml/deepfake/dataset.py
======================
PyTorch Dataset for Face Forgery Detection.
Implements robust anti-overfitting augmentations (JPEG recompression, Gaussian blur, color jitter).
"""

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULT_IMG_SIZE = 288 # EfficientNet-B5 standard input resolution (260 for B2)

def get_train_transforms(img_size: int = DEFAULT_IMG_SIZE):
    """
    Augmentation pipeline designed to defend against dataset-specific compression artifacts.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        # Defend against JPEG compression memorization
        A.ImageCompression(quality_range=(55, 95), p=0.5),
        # Defend against optical edge sharpness memorization
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.MotionBlur(blur_limit=(3, 7), p=0.4),
        ], p=0.4),
        # Color & lighting variability
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        # ImageNet normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(img_size: int = DEFAULT_IMG_SIZE):
    """Deterministic validation/test preprocessing."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

class FaceForensicsDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, is_training: bool = True, img_size: int = DEFAULT_IMG_SIZE):
        self.df = pd.read_csv(manifest_csv)
        self.is_training = is_training
        self.img_size = img_size
        self.transforms = get_train_transforms(img_size=img_size) if is_training else get_val_transforms(img_size=img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["filepath"])

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot load image at path: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=image_rgb)
        tensor = augmented["image"]

        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        
        # Sub-type fallback for datasets without explicit difficulty labels
        if "sub_type" in row and pd.notna(row["sub_type"]):
            sub_type = str(row["sub_type"])
        else:
            sub_type = "fake" if float(row["label"]) > 0.5 else "real"

        return tensor, label, sub_type

