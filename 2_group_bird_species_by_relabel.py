import json, collections, csv
from pathlib import Path

"""
2.DATASET PREPARATION WITH RELABELING

This script reads a metadata JSON file containing bird species image records,
and relabels species with fewer than a specified threshold of images to 'OTHER'.
It then writes out the modified metadata and a summary CSV report.
"""

# === INPUT / OUTPUT ===
meta_in  = Path(r"C:/Users/Audub/saahil_classification/data/metadata_full.json")
result_outdir = Path(r"C:/Users/Audub/Classification/results")
# write output into the results directory (create a new file for threshold 100)
meta_out = result_outdir / "metadata_balanced_t100.json"
report_csv = meta_out.with_suffix(".summary.csv")

# combine species with fewer than this many examples into the OTHERS bucket
THRESH = 100
OTHER_LABEL = "OTHERS"

print(f"[info] reading {meta_in}")
data = json.loads(meta_in.read_text(encoding="utf-8"))

# count per species
cnt = collections.Counter(r.get("species_name", "UNKNOWN") for r in data)

# which species get merged
to_other = {sp for sp,c in cnt.items() if c < THRESH}
print(f"[info] classes < {THRESH}: {len(to_other)} → merged to {OTHER_LABEL}")
if to_other:
    print("       ", ", ".join(sorted(to_other)))

# relabel and build new counts
new_data = []
for r in data:
    sp = r.get("species_name", "UNKNOWN")
    if sp in to_other:
        r = dict(r)
        r["species_name_original"] = sp
        r["species_name"] = OTHER_LABEL
    new_data.append(r)

# write out
meta_out.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
print(f"[done] wrote {meta_out}")

# summary CSV
after = collections.Counter(r["species_name"] for r in new_data)
total = sum(after.values())
with report_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["species_name", "count", "percent"])
    for sp, c in after.most_common():
        w.writerow([sp, c, f"{(c/total*100):.3f}"])
print(f"[done] wrote {report_csv}")
