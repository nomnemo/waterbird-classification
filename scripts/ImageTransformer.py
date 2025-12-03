from typing import List

import cv2  # type: ignore
import albumentations as A  # type: ignore

"""
IMAGE TRANSFORMER UTILITIES

This module centralizes the Albumentations pipelines used for image preprocessing
and augmentation. The main public API is `make_transforms`, which dispatches to
separate helper functions for training-time and evaluation-time transforms.
"""

# ImageNet mean/std (what ViT/Swin expect in timm/timm pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _resize_and_pad(img_size: int, min_size: int) -> List[A.BasicTransform]:
    """
    Common resizing/padding steps used for both training and evaluation:
      - Resize so that the longest side is at least `min_size`.
      - Pad to ensure the image is at least img_size x img_size.
    """
    return [
        A.LongestMaxSize(max(img_size, min_size)),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_REFLECT_101),
    ]

def _normalize() -> A.Normalize:
    """Return the standard ImageNet normalization transform."""
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def build_train_transforms(img_size: int) -> A.Compose:
    """
    Build the training-time augmentation pipeline.

    Includes:
      - resize/pad to preserve context,
      - random resized crop,
      - flips, small rotations/shifts/scales,
      - color jitter and coarse dropout,
      - ImageNet normalization.
    """
    transforms = _resize_and_pad(img_size, min_size=256) + [
        # colony images are roughly rotation/flip invariant
        A.RandomResizedCrop(
            img_size,
            img_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.333),
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=20,
            p=0.35,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
            p=0.35,
        ),
        A.CoarseDropout(
            max_holes=2,
            max_height=int(img_size * 0.12),
            max_width=int(img_size * 0.12),
            min_holes=1,
            fill_value=0,
            p=0.3,
        ),
        _normalize(),
    ]
    return A.Compose(transforms)

def build_eval_transforms(img_size: int) -> A.Compose:
    """
    Build the evaluation-time (validation/test) pipeline.

    Uses deterministic preprocessing:
      - resize/pad,
      - center crop,
      - ImageNet normalization.
    """
    transforms = _resize_and_pad(img_size, min_size=img_size) + [
        A.CenterCrop(img_size, img_size),
        _normalize(),
    ]
    return A.Compose(transforms)

def get_transforms(img_size: int, train: bool) -> A.Compose:
    """
    Public factory for Albumentations transform pipelines.

    Args:
        img_size: target image size (square) expected by the model.
        train: if True, return the training augmentation pipeline;
               otherwise, return the evaluation pipeline.

    Returns:
        Albumentations `Compose` object.
    """
    if train:
        return build_train_transforms(img_size)
    return build_eval_transforms(img_size)