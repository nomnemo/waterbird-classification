import json, csv, re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List
import numpy as np # type: ignore
from sklearn.model_selection import GroupShuffleSplit # type: ignore

"""
3. MAKE GROUPS AND SPLITS
This script reads a metadata JSON file containing bird species image records,
and creates training, validation, and test splits (80/10/10) while ensuring that
there is no leakage of images from the same source group (e.g., same parent image).
It outputs CSV files for each split.
"""

# === CONFIG ===
json_data = Path(r"data/metadata_balanced_t100.json")
out_dir = json_data.parent
train_csv = out_dir / "split_train.csv"
val_csv   = out_dir / "split_val.csv"
test_csv  = out_dir / "split_test.csv"

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

def _write_csv_from_list_of_dicts(path: Path, rows: List[dict]):
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

def create_grouped_splits(
    parent_images_per_record: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state_1: int = 42,
    random_state_2: int = 123
) -> tuple:
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
        tuple: (train_idx, val_idx, test_idx) - numpy arrays of record indices
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Create index array for all records
    idx = np.arange(len(parent_images_per_record))
    
    # SPLIT 1: train vs holdout
    # holdout_ratio = val_ratio + test_ratio
    holdout_ratio = 1.0 - train_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=random_state_1)
    train_idx, hold_idx = next(gss1.split(idx, groups=parent_images_per_record))
    
    # SPLIT 2: val vs test (from the holdout set)
    # Among holdout, test gets: test_ratio / holdout_ratio of the holdout
    val_test_split_ratio = test_ratio / holdout_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_test_split_ratio, random_state=random_state_2)
    val_rel, test_rel = next(gss2.split(hold_idx, groups=parent_images_per_record[hold_idx]))
    val_idx = hold_idx[val_rel]
    test_idx = hold_idx[test_rel]
    
    return train_idx, val_idx, test_idx

def _print_top_5_species(name: str, indices: List[int]):
    """
    Prints top5 species for a given split.
    """
    sub = record_species_list[indices]
    c = Counter(sub)
    total = len(sub)
    top = ", ".join(f"{k}:{v}" for k,v in c.most_common(5))
    print(f"[{name}] n={total} | top5: {top}")

def _get_records(indices):
    """
    Given a list of indices, return the corresponding records.
    """
    return [records[i] for i in indices]

# === MAIN ===
print(f"[info] reading {json_data}")
records = json.loads(json_data.read_text(encoding="utf-8"))

# get parent image for each img
list_of_parents: List[str] = [_get_parent_from_source(r.get("source_image", "")) for r in records]
parent_images_per_record = np.array(list_of_parents)

# Create 80/10/10 grouped splits (no leakage)
train_idx, val_idx, test_idx = create_grouped_splits(
    parent_images_per_record,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state_1=42,
    random_state_2=123
)

# ===== EXTRA INFO =====
# get species name for each img
record_species_list = np.array([r["species_name"] for r in records])
idx    = np.arange(len(records))

# derive a single proxy label per group (dominant class) 
parent_img_to_img_indices = defaultdict(list)
for idx, parent in enumerate(parent_images_per_record):
    parent_img_to_img_indices[parent].append(idx)

print(f"[info] parent-grouped sizes: \n"
      f"      train={len(train_idx)}, \n"
      f"      val={len(val_idx)}, test={len(test_idx)}")
print(f"[info] unique parents: total={len(parent_img_to_img_indices)}, "
      f"train={len(set(parent_images_per_record[train_idx]))}, "
      f"val={len(set(parent_images_per_record[val_idx]))}, "
      f"test={len(set(parent_images_per_record[test_idx]))}")
_print_top_5_species("train", train_idx)
_print_top_5_species("val  ", val_idx)
_print_top_5_species("test ", test_idx)

# ==== write CSVs ====
_write_csv_from_list_of_dicts(train_csv, _get_records(train_idx))
_write_csv_from_list_of_dicts(val_csv,   _get_records(val_idx))
_write_csv_from_list_of_dicts(test_csv,  _get_records(test_idx))
print(f"[done] wrote:\n  {train_csv}\n  {val_csv}\n  {test_csv}")