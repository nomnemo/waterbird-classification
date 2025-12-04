import json
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
from sklearn.model_selection import GroupShuffleSplit  # type: ignore

"""
3. MAKE GROUPS AND SPLITS

This module reads a metadata JSON file containing bird species image records
and creates training, validation, and test splits (default 80/10/10) while
ensuring that there is no leakage of images from the same source group
(e.g., same parent image). It can be imported as a library or run as a script
to write CSV splits.
"""

# === DEFAULT CONFIG ===
DEFAULT_JSON = Path("data/metadata_balanced_t100.json")
DEFAULT_TRAIN_CSV = DEFAULT_JSON.parent / "split_train.csv"
DEFAULT_VAL_CSV = DEFAULT_JSON.parent / "split_val.csv"
DEFAULT_TEST_CSV = DEFAULT_JSON.parent / "split_test.csv"

def _get_parent_from_source(source_path_name: str) -> str:
    """
    Given a source image path like 'images\102741 - 00001.jpg',
    extract the parent prefix '102741' to group related images.
    """
    # fname is the filename part. for example '102741 - 00001.jpg'
    fname = Path(source_path_name).name
    
    # extract the leading number before ' - ' using regex
    m = re.match(r"(\d+)\s*-\s*\d+\.(jpg|jpeg|png)$", fname, re.IGNORECASE)

    # return the leading number if matched, else the stem of the path
    # for example, the matched group is '102741'
    # if not matched, the stem is '102741 - 00001'
    return m.group(1) if m else Path(source_path_name).stem

def _write_csv_from_list_of_dicts(path: Path, rows: List[Dict]) -> None:
    """
    Write a list of dictionaries to a CSV file, ensuring the header includes all keys.
    """
    # build a header that is the union of all keys (handles 'species_name_original')
    header = set()
    for r in rows:
        header.update(r.keys())
    header = list(header)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

def _print_split_summary(info: Dict) -> None:
    """
    Print a short textual summary of the splits: sizes and top classes.
    """
    train_idx = info["train_idx"]
    val_idx = info["val_idx"]
    test_idx = info["test_idx"]
    parent_images_per_record = info["parent_images_per_record"]
    class_counts = info["class_counts"]

    # Count unique parent images per split.
    parent_to_indices = defaultdict(list)
    for i, parent in enumerate(parent_images_per_record):
        parent_to_indices[parent].append(i)

    print(
        "[info] parent-grouped sizes:\n"
        f"       train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    print(
        "[info] unique parents: "
        f"total={len(parent_to_indices)}, "
        f"train={len(set(parent_images_per_record[train_idx]))}, "
        f"val={len(set(parent_images_per_record[val_idx]))}, "
        f"test={len(set(parent_images_per_record[test_idx]))}"
    )

    def _top5(name: str, counter: Counter) -> None:
        top = ", ".join(f"{k}:{v}" for k, v in counter.most_common(5))
        total = sum(counter.values())
        print(f"[{name}] n={total} | top5: {top}")

    _top5("train", class_counts["train"])
    _top5("val  ", class_counts["val"])
    _top5("test ", class_counts["test"])

def create_grouped_splits(
    parent_images_per_record: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state_1: int = 42,
    random_state_2: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val/test splits with no data leakage between groups.
    
    Given parent group assignments for each record, this function ensures that
    all records from the same parent group stay together in one split.
    
    Args:
        parent_images_per_record: np.ndarray of shape (n_records,) where each element
                                  is the parent group ID for that record
        train_ratio: fraction of groups for training (default 0.8)
        val_ratio: fraction of groups for validation (default 0.1)
        test_ratio: fraction of groups for testing (default 0.1)
        random_state_1: random seed for first split (train vs holdout)
        random_state_2: random seed for second split (val vs test)
    
    Returns:
        (train_idx, val_idx, test_idx): numpy arrays of record indices
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Create index array for all records
    idx = np.arange(len(parent_images_per_record))
    
    # SPLIT 1: train vs holdout
    # holdout_ratio = val_ratio + test_ratio
    holdout_ratio = 1.0 - train_ratio
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=holdout_ratio, random_state=random_state_1
    )
    train_idx, hold_idx = next(gss1.split(idx, groups=parent_images_per_record))
    
    # SPLIT 2: val vs test (from the holdout set)
    # Among holdout, test gets: test_ratio / holdout_ratio of the holdout
    val_test_split_ratio = test_ratio / holdout_ratio
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_test_split_ratio, random_state=random_state_2
    )
    val_rel, test_rel = next(gss2.split(hold_idx, groups=parent_images_per_record[hold_idx]))
    val_idx = hold_idx[val_rel]
    test_idx = hold_idx[test_rel]
    
    return train_idx, val_idx, test_idx

def get_data_splits(
    json_path: Path = DEFAULT_JSON,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state_1: int = 42,
    random_state_2: int = 123,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Load metadata from JSON and create grouped train/val/test splits.

    Args:
        json_path: Path to the metadata JSON file.
        train_ratio: Fraction of groups assigned to training split.
        val_ratio: Fraction of groups assigned to validation split.
        test_ratio: Fraction of groups assigned to test split.
        random_state_1: Seed for the initial train vs holdout split.
        random_state_2: Seed for the validation vs test split.

    Returns:
        train_records: List of record dicts for the training split.
        val_records:   List of record dicts for the validation split.
        test_records:  List of record dicts for the test split.
        info:          Dictionary with split metadata, including:
                       - "train_idx", "val_idx", "test_idx": numpy arrays of indices
                       - "parent_images_per_record": numpy array of parent IDs
                       - "class_counts": Counter of species in each split
    """
    print(f"[info] reading {json_path}")
    records: List[Dict] = json.loads(json_path.read_text(encoding="utf-8"))

    # Derive parent image ID for each record.
    parent_ids: List[str] = [
        _get_parent_from_source(r.get("source_image", "")) for r in records
    ]
    parent_images_per_record = np.array(parent_ids)

    # Create grouped splits with no leakage across parent images.
    train_idx, val_idx, test_idx = create_grouped_splits(
        parent_images_per_record,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state_1=random_state_1,
        random_state_2=random_state_2,
    )

    # Build class distributions for summary.
    species_all = np.array([r["species_name"] for r in records])

    def _counts(indices: np.ndarray) -> Counter:
        return Counter(species_all[indices])

    class_counts = {
        "train": _counts(train_idx),
        "val": _counts(val_idx),
        "test": _counts(test_idx),
    }

    # Build record lists for each split.
    def _select(indices: np.ndarray) -> List[Dict]:
        return [records[int(i)] for i in indices]

    train_records = _select(train_idx)
    val_records = _select(val_idx)
    test_records = _select(test_idx)

    # Additional info for inspection or logging.
    info: Dict = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "parent_images_per_record": parent_images_per_record,
        "class_counts": class_counts,
    }

    return train_records, val_records, test_records, info

if __name__ == "__main__":
    # When run as a script, generate splits from the default JSON and
    # write them to CSV files in the data directory.
    train_records, val_records, test_records, info = get_data_splits()
    _print_split_summary(info)

    _write_csv_from_list_of_dicts(DEFAULT_TRAIN_CSV, train_records)
    _write_csv_from_list_of_dicts(DEFAULT_VAL_CSV, val_records)
    _write_csv_from_list_of_dicts(DEFAULT_TEST_CSV, test_records)

    print(
        "[done] wrote:\n"
        f"  {DEFAULT_TRAIN_CSV}\n"
        f"  {DEFAULT_VAL_CSV}\n"
        f"  {DEFAULT_TEST_CSV}"
    )
