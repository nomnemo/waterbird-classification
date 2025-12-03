from pathlib import Path
import csv
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd

from scripts.image_transformer import get_transforms
from scripts.BirdDataset import BirdDataset

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
DATA_DIR   = Path(r"data")  # where split_*.csv live
IMAGE_ROOT = Path(r"data/crops") # where cropped images live
TRAIN_CSV = DATA_DIR / "split_train.csv"
VAL_CSV   = DATA_DIR / "split_val.csv"
TEST_CSV  = DATA_DIR / "split_test.csv"

# inputs to the dataloader
INPUT_SIZE = 224            # 224 or 256 or 320
USE_SAMPLER = True 
BATCH_TRAIN = 32            
BATCH_EVAL  = 128
NUM_WORKERS = 6 # DataLoader parallelization is on the CPU side # Start around num_workers = (#CPU cores) / 2
SEED = 42

def _build_label_map(rows: List[Dict], label_key: str = "species_name"):
    """
    Given a list of rows (dicts), and a label key (default: "species_name"),
    Build a consistent label map from label -> class index.
    """
    # get all unique labels
    classes: List[str] = sorted({row[label_key] for row in rows})

    # assign each class a unique index
    class2id = {class_name: i for i, class_name in enumerate(classes)}
    return class2id, classes

def _cap_per_class(df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
    """
    Cap (limit) the number of rows per class, without upsampling.

    Behavior:
        - If max_per_class is None: return df unchanged (no capping).
        - Otherwise:
            * Shuffle the DataFrame once.
            * For each species_name, keep at most max_per_class rows.
            * Classes with fewer than max_per_class examples keep all
            their rows (we do NOT create or duplicate samples).
    """
    if max_per_class is None:
        return df

    # Shuffle once, then take first K per class
    df = df.sample(frac=1.0, random_state=42)
    return (
        df.groupby("species_name", group_keys=False)
        .head(max_per_class)
        .reset_index(drop=True) # drop=True means we don’t add the old index as a new column.
    )
    
def set_up_data_loaders(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,
    input_size=INPUT_SIZE,
    use_sampler=USE_SAMPLER,
    batch_train=BATCH_TRAIN,
    batch_eval=BATCH_EVAL,
    num_workers=NUM_WORKERS,
    max_per_class=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build PyTorch DataLoaders for train/val/test splits.

    This helper:
      1) Reads split CSVs (train/val/test).
      2) Optionally caps the number of samples per class in each split.
      3) Builds a consistent label map from species_name -> class index
         based on the (possibly capped) training split.
      4) Creates BirdDataset instances with appropriate augmentations.
      5) Optionally uses a WeightedRandomSampler on the training set
         to mitigate class imbalance.

    Args:
        train_csv: path to CSV with training rows (default: DATA_DIR/split_train.csv).
        val_csv:   path to CSV with validation rows (default: DATA_DIR/split_val.csv).
        test_csv:  path to CSV with test rows (default: DATA_DIR/split_test.csv).
        input_size: image size (pixels, square) for model input; controls
                    resize/crop behavior inside the Albumentations pipeline.
        use_sampler: if True, use a WeightedRandomSampler for the training loader
                     based on inverse class frequency (helps balance long-tailed classes).
        batch_train: training batch size.
        batch_eval:  evaluation batch size for val/test.
        num_workers: number of worker processes used by each DataLoader to load
                     and augment images in parallel.
        max_per_class: if not None, cap the number of training samples per class
                       to this value. Val/test are capped to roughly max_per_class / 5
                       per class to keep them smaller while preserving class coverage.

    Returns:
        dl_train: DataLoader for training.
        dl_val:   DataLoader for validation.
        dl_test:  DataLoader for testing.
        meta:     dict with:
                  - "classes": list of class names (ordered by index)
                  - "cls2id": mapping from class name -> index
                  - "class_counts": Counter of training samples per class
                  - "class_weights": 1/frequency normalized array for use in loss
                  - "sizes": dict with sizes of train/val/test (after capping)
    """

    # 1) read CSVs as Pandas DataFrames
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    # 2) ================= CAP CLASS SIZES (if max_per_class is provided) =================
    train_df = _cap_per_class(train_df, max_per_class)
    if max_per_class is not None:
        cap_val_test = max(1, max_per_class // 5)   # smaller val/test per class
    else:
        cap_val_test = None
    val_df  = _cap_per_class(val_df,  cap_val_test)
    test_df = _cap_per_class(test_df, cap_val_test)

    # 3) convert to list[dict] that BirdDataset expects
    train_rows: List[Dict] = train_df.to_dict(orient="records")
    val_rows: List[Dict] = val_df.to_dict(orient="records")
    test_rows: List[Dict] = test_df.to_dict(orient="records")

    # 4)class label map derived from TRAINING SPLIT ONLY
    class2id, species = _build_label_map(train_rows)

    # 5) transforms
    train_transformer = get_transforms(input_size, train=True)
    eval_transformer = get_transforms(input_size, train=False)

    # 6) ================ create BirdDataset instances for each split ================
    #    these wrap the raw metadata rows and,
    #    when indexed, will load the image from disk, apply the appropriate
    #    transform (train/eval), and return (image_tensor, label_tensor) pairs.
    # TODO: learn what a tensor is and explain in the report
    ds_train = BirdDataset(train_rows, class2id, IMAGE_ROOT, train_transformer, missing_size=input_size)
    ds_val = BirdDataset(val_rows, class2id, IMAGE_ROOT, eval_transformer, missing_size=input_size)
    ds_test = BirdDataset(test_rows, class2id, IMAGE_ROOT, eval_transformer, missing_size=input_size)

    # 7) ============= sampler / class weights from CAPPED train set =============
    
    # counting how many from each class in the training set
    class_counts = Counter([r["species_name"] for r in train_rows])
    sampler = None
    if use_sampler:
        # Since our dataset is long-tailed (some classes have many more samples
        # than others), we can use a WeightedRandomSampler to balance the classes
        # during training. This sampler assigns a weight to each sample inversely
        # proportional to its class frequency, so that rarer classes are sampled more
        # frequently.
        
        # If we just sample uniformly from the dataset, common classes will dominate in most batches,
        # and the model will not learn to recognize rare classes well. We want the model to see the 
        # rare classes more often during training without actually duplicating any data.
        weights = np.array([1.0 / class_counts[r["species_name"]] for r in train_rows], dtype=np.float64)
        
        # WeightedRandomSampler draws length(weights) samples per epoch, essentially creating a new
        # balanced dataset each epoch by oversampling rare classes and undersampling common ones.
        # This helps with balanced learning, which then improves macro metrics like F1-score across all classes.
        # tradeoff: 
        # - some samples from common classes may be skipped in an epoch, so we don't use all their diversity in each epoch. 
        # - rare classes might slightly overfit since they are repeated more often.
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Using PyTorch DataLoader to handle batching, shuffling, and parallel loading.
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

    # 8) normalized class weights (optional for Cross Entropy/Focal loss [the standard loss for multi‑class classification])
    # precomputing a per-class weight vector that can be plugged into some loss functions to help with class imbalance.
    # class_counts[c] = how many training samples you have for class c (after any capping). 
    # raw_weight_c = 1.0 / class_counts[c] for each class. 
    cls_weights = np.array([1.0 / max(class_counts[c], 1) for c in species], dtype=np.float32)
    normalized_cls_weights = cls_weights / cls_weights.mean()

    meta = {
        "classes": species,
        "class2id": class2id,
        "class_counts": class_counts,
        "class_weights": torch.tensor(normalized_cls_weights, dtype=torch.float32),
        "sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
    }
    return dl_train, dl_val, dl_test, meta

if __name__ == "__main__":
    train_dl, val_dl, test_dl, meta = set_up_data_loaders()
    
    # sanity checks
    print(f"Train size: {len(train_dl.dataset)}")
    print(f"Val size:   {len(val_dl.dataset)}")
    print(f"Test size:  {len(test_dl.dataset)}")
    print("Num classes:", len(meta["classes"]))
    print("Classes:", meta["classes"])

    # xb = x batch = tensor of input images, shape roughly (batch_size, 3, H, W) (e.g., (32, 3, 224, 224)).
    # yb = y batch = tensor of labels, shape (batch_size,) (e.g., (32,)), with each entry an integer class index.
    xb, yb = next(iter(train_dl))
    # confirm shapes
    print("train batch:", xb.shape, yb.shape)
    print("num classes:", len(meta["classes"]))
