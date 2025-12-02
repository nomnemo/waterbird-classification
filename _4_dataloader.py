import os
from pathlib import Path
import csv
import random
from collections import Counter

import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

"""
4. DATA LOADER MODULE

This module defines a PyTorch Dataset and DataLoader setup for bird species classification.
It reads CSV files defining training, validation, and test splits, applies data augmentations,
and supports weighted sampling to address class imbalance.

The BirdDataset class expects CSV rows with at least:
    - crop_path       (path is relative to DATA_DIR or absolute)
    - species_name    (string label)
Optional but preserved: source_image, bbox_original, bbox_expanded, etc.

The build_loaders function constructs DataLoader objects for training, validation, and testing,
with configurable parameters such as input size, batch size, and whether to use a weighted sampler.
"""

# ---------- CONFIG ----------
DATA_DIR   = Path(r"C:/Users/Audub/Classification/data")  # where split_*.csv live
IMAGE_ROOT = Path(r"C:/Users/Audub/saahil_classification/data") # where cropped images live

TRAIN_CSV = DATA_DIR / "split_train.csv"
VAL_CSV   = DATA_DIR / "split_val.csv"
TEST_CSV  = DATA_DIR / "split_test.csv"

# inputs to the dataloader
INPUT_SIZE = 224            # 224 or 256 or 320
USE_SAMPLER = True          # WeightedRandomSampler to mitigate imbalance
BATCH_TRAIN = 32
BATCH_EVAL  = 128
NUM_WORKERS = 6 # why are we using 6 workers?
# This is a common choice for the number of workers,
# as it provides a good balance between CPU and I/O performance without overwhelming the system.
SEED = 42

# ImageNet mean/std (what ViT/Swin expect in timm)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
# ----------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_csv_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def build_label_map(rows, label_key="species_name"):
    classes = sorted({row[label_key] for row in rows})
    cls2id = {c:i for i,c in enumerate(classes)}
    return cls2id, classes

"""
Given an image size and whether it's for training, or evaluation,
it returns an Albumentations transform pipeline. If the transform is for training,
it includes data augmentations like: 
- random resized crop, flips, rotations, color jitter, and coarse dropout
- normalization with ImageNet mean/std

otherwise, it uses center cropping and normalization.
"""
def make_transforms(img_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose([
            # keep more context than a tight center crop
            A.LongestMaxSize(max(img_size, 256)),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_REFLECT_101),

            # colony images are roughly rotation/flip invariant
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), ratio=(0.75, 1.333), always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),      # small prob
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.35, border_mode=cv2.BORDER_REFLECT_101),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.35),
            A.CoarseDropout(max_holes=2, max_height=int(img_size*0.12), max_width=int(img_size*0.12),
                            min_holes=1, fill_value=0, p=0.3),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_REFLECT_101),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class BirdDataset(Dataset):
    """
    Expects CSV rows with at least:
      - crop_path       (path is relative to DATA_DIR or absolute)
      - species_name    (string label)
    Optional but preserved: source_image, bbox_original, bbox_expanded, etc.
    """
    def __init__(self, rows, cls2id, img_root: Path, transform):
        self.rows = rows
        self.cls2id = cls2id
        self.img_root = img_root
        self.tf = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        y = self.cls2id[row["species_name"]]

        p_raw = row["crop_path"].replace("\\", "/")
        p0 = Path(p_raw)

        if p0.is_absolute():
            full = p0
        else:
            # CSV is relative like 'crops/CLASS/file.jpg' → join to IMAGE_ROOT
            full = (self.img_root / p0).resolve()

        img = cv2.imread(full.as_posix())
        if img is None:
            # helpful one-time print for debugging
            if idx < 3:
                print(f"[warn] missing image -> {full.as_posix()}")
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        img = self.tf(image=img)["image"]
        x = torch.from_numpy(img.transpose(2,0,1)).float()

        return x, torch.tensor(y).long()

def build_loaders(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,
    input_size=INPUT_SIZE,
    use_sampler=USE_SAMPLER,
    batch_train=BATCH_TRAIN,
    batch_eval=BATCH_EVAL,
    num_workers=NUM_WORKERS,
    max_per_class=None,
):
    import pandas as pd

    MAX_PER_CLASS = max_per_class  # ~4k total if ~21 classes

    def cap_per_class(df, max_per_class):
        # Shuffle once, then take first K per class 
        df = df.sample(frac=1.0, random_state=42)
        return (
            df.groupby("species_name", group_keys=False)
            .head(max_per_class)
            .reset_index(drop=True) # drop=True means we don’t add the old index as a new column.
        )

    # 1) read CSVs as DataFrames
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    # 2) cap sizes
    train_df = cap_per_class(train_df, MAX_PER_CLASS)
    val_df   = cap_per_class(val_df,   max(1, MAX_PER_CLASS // 5))   # smaller val
    test_df  = cap_per_class(test_df,  max(1, MAX_PER_CLASS // 5))   # smaller test

    # 3) convert to list[dict] that BirdDataset expects
    train_rows = train_df.to_dict(orient="records")
    val_rows   = val_df.to_dict(orient="records")
    test_rows  = test_df.to_dict(orient="records")

    # 4) label map from TRAIN ONLY (capped)
    cls2id, classes = build_label_map(train_rows)

    # 5) transforms
    t_train = make_transforms(input_size, train=True)
    t_eval  = make_transforms(input_size, train=False)

    # 6) datasets
    img_root = IMAGE_ROOT  # e.g., C:/Users/Audub/saahil_classification/data
    ds_train = BirdDataset(train_rows, cls2id, img_root, t_train)
    ds_val   = BirdDataset(val_rows,   cls2id, img_root, t_eval)
    ds_test  = BirdDataset(test_rows,  cls2id, img_root, t_eval)

    # 7) sampler / class weights from CAPPED train set
    from collections import Counter
    class_counts = Counter([r["species_name"] for r in train_rows])

    sampler = None
    if use_sampler:
        weights = np.array([1.0 / class_counts[r["species_name"]] for r in train_rows], dtype=np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl_train = DataLoader(ds_train,
                            batch_size=batch_train,
                            shuffle=(sampler is None), # if the sampler is used, disable shuffle
                            sampler=sampler, 
                            num_workers=num_workers, # the number of subprocesses to use for data loading
                            pin_memory=True,
                            persistent_workers=(num_workers > 0))
    dl_val   = DataLoader(ds_val,
                            batch_size=batch_eval,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=(num_workers > 0))
    dl_test  = DataLoader(ds_test,
                            batch_size=batch_eval,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                          persistent_workers=(num_workers > 0))

    # 8) normalized class weights (optional for CE/Focal)
    cls_weights = np.array([1.0 / max(class_counts[c], 1) for c in classes], dtype=np.float32)
    cls_weights = cls_weights / cls_weights.mean()

    meta = {
        "classes": classes,
        "cls2id": cls2id,
        "class_counts": class_counts,
        "class_weights": torch.tensor(cls_weights, dtype=torch.float32),
        "sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
    }
    return dl_train, dl_val, dl_test, meta


if __name__ == "__main__":
    # smoke test
    tr, va, te, meta = build_loaders()
    print(f"Train size: {len(tr.dataset)}")
    print(f"Val size:   {len(va.dataset)}")
    print(f"Test size:  {len(te.dataset)}")
    print("Num classes:", len(meta["classes"]))
    print("Classes:", meta["classes"])

    xb, yb = next(iter(tr))
    print("train batch:", xb.shape, yb.shape)
    print("num classes:", len(meta["classes"]))