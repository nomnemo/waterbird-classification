import json, csv, re
from pathlib import Path
from collections import Counter, defaultdict
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
meta_path = Path(r"C:/Users/Audub/Classification/results/metadata_balanced_t100.json")
out_dir = meta_path.parent
train_csv = out_dir / "split_train.csv"
val_csv   = out_dir / "split_val.csv"
test_csv  = out_dir / "split_test.csv"

def parent_from_source(p: str) -> str:
    # extracts '102741' from 'images\\102741 - 00001.jpg'
    fname = Path(p).name
    m = re.match(r"(\d+)\s*-\s*\d+\.(jpg|jpeg|png)$", fname, re.IGNORECASE)
    return m.group(1) if m else Path(p).stem

def write_csv(path: Path, rows):
    # build a header that is the union of all keys (handles 'species_name_original')
    header = set()
    for r in rows:
        header.update(r.keys())
    header = list(header)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

print(f"[info] reading {meta_path}")
records = json.loads(meta_path.read_text(encoding="utf-8"))

# groups: parent prefix per record
groups = np.array([parent_from_source(r.get("source_image","")) for r in records])
labels = np.array([r["species_name"] for r in records])
idx    = np.arange(len(records))

# OPTIONAL: derive a single proxy label per group (dominant class)
# apparently – helps approximate stratification checks later
group_to_idxs = defaultdict(list)
for i,g in enumerate(groups):
    group_to_idxs[g].append(i)
group_to_major = {}
for g, idcs in group_to_idxs.items():
    maj = Counter(labels[idcs]).most_common(1)[0][0]
    group_to_major[g] = maj

# split 80/10/10 grouped (no leakage)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, hold_idx = next(gss1.split(idx, groups=groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=123)
val_rel, test_rel = next(gss2.split(hold_idx, groups=groups[hold_idx]))
val_idx  = hold_idx[val_rel]
test_idx = hold_idx[test_rel]

def summarize(name, indices):
    sub = labels[indices]
    c = Counter(sub)
    total = len(sub)
    top = ", ".join(f"{k}:{v}" for k,v in c.most_common(5))
    print(f"[{name}] n={total} | top5: {top}")

print(f"[info] parent-grouped sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
print(f"[info] unique parents: total={len(group_to_idxs)}, "
      f"train={len(set(groups[train_idx]))}, val={len(set(groups[val_idx]))}, test={len(set(groups[test_idx]))}")

summarize("train", train_idx)
summarize("val  ", val_idx)
summarize("test ", test_idx)

# write CSVs
def select(indices):
    return [records[i] for i in indices]

write_csv(train_csv, select(train_idx))
write_csv(val_csv,   select(val_idx))
write_csv(test_csv,  select(test_idx))
print(f"[done] wrote:\n  {train_csv}\n  {val_csv}\n  {test_csv}")