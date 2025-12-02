import json, csv, collections, os
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, numpy as np # type: ignore

"""
1.DATASET DISTRIBUTION ANALYSIS

This script analyzes a metadata JSON file containing bird species image records,
and generates distribution statistics such as counts per species,
coverage by minimum count thresholds, and tern species summary.
It outputs CSV files and plots to visualize the dataset distribution.
"""

# === CONFIG ===
# metadata_in = r"C:/Users/Audub/saahil_classification/data/metadata_full.json"
metadata_in = r"C:/Users/Audub/Classification/results/metadata_balanced_t100.json"
outdir = Path(".")   # current working directory

# === LOAD ===
with open(metadata_in, "r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"[info] loaded {len(meta):,} crop records from {metadata_in}")

# === COUNT PER SPECIES ===
by_species = collections.Counter()
by_species_id = collections.defaultdict(set)

for r in meta:
    sp = r.get("species_name", "UNKNOWN")
    sid = r.get("species_id", None)
    by_species[sp] += 1
    if sid is not None:
        by_species_id[sp].add(sid)

total = sum(by_species.values())
uniq = len(by_species)

print(f"[info] total crops: {total:,}")
print(f"[info] unique species: {uniq}")
print("[info] top 10:")
for sp, c in by_species.most_common(10):
    print(f"  {sp:>8} : {c}")

# === WRITE species_counts.csv ===
outdir.mkdir(parents=True, exist_ok=True)
species_csv = outdir / "species_counts.csv"
with open(species_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["species_name", "count", "percent", "species_ids_seen"])
    for sp, c in by_species.most_common():
        ids_for_sp = ";".join(map(str, sorted(by_species_id.get(sp, []))))
        pct = (c / total) * 100 if total else 0.0
        w.writerow([sp, c, f"{pct:.3f}", ids_for_sp])
print(f"[done] wrote {species_csv}")

# === COVERAGE BY THRESHOLD (helps choose 'OTH R < X') ===
thresholds = list(range(5, 55, 5)) + [75, 100, 150, 200, 300, 500]
coverage_csv = outdir / "coverage_by_min_count.csv"
with open(coverage_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["min_count", "num_species_kept", "num_images_kept", "percent_images_kept"])
    for t in thresholds:
        kept_species = [sp for sp, c in by_species.items() if c >= t]
        kept_counts = sum(by_species[sp] for sp in kept_species)
        pct = (kept_counts / total * 100) if total else 0.0
        w.writerow([t, len(kept_species), kept_counts, f"{pct:.2f}"])
print(f"[done] wrote {coverage_csv}")

# === TERN SUMMARY ===
tern_labels = {"MTRN", "ROTE", "SATE"}
if any(sp in by_species for sp in tern_labels):
    tern_csv = outdir / "tern_summary.csv"
    with open(tern_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for sp in sorted(tern_labels):
            w.writerow([sp, by_species.get(sp, 0)])
        total_terns = sum(by_species.get(sp, 0) for sp in tern_labels)
        w.writerow(["TERNS_TOTAL", total_terns])
    print(f"[done] wrote {tern_csv}")

print("[info] finished analysis.")

# === PLOTTING ===
df = pd.read_csv(str(species_csv)).sort_values("count", ascending=False)

plt.figure(figsize=(14,6))
bars = plt.bar(df["species_name"], df["count"], color="#2b83ba")
plt.xticks(rotation=60, ha="right", fontsize=9)
plt.ylabel("Number of crops", fontsize=11)
plt.title("Bird species sample distribution", fontsize=13)

# add numeric labels on top of bars
for bar, value in zip(bars, df["count"]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, height + 300,  # small offset
        f"{int(value):,}", ha="center", va="bottom", fontsize=8, rotation=90
    )

plt.tight_layout()
plt.savefig("species_bar_labeled.png", dpi=250)
plt.show()

# cumulative coverage curve
df["cum_count"] = df["count"].cumsum()
df["cum_pct"] = 100 * df["cum_count"] / df["count"].sum()
plt.figure(figsize=(8,5))
plt.plot(np.arange(len(df)), df["cum_pct"], marker="o")
plt.axhline(90, color="red", ls="--")
plt.xlabel("Top-k species (sorted)")
plt.ylabel("Cumulative % of all images")
plt.title("Cumulative coverage by species rank")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("species_cumulative.png", dpi=200)
plt.show()
