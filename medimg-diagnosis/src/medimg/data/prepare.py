"""Build a manifest (and resized image cache) from NIH ChestX-ray14.

Usage:
  python -m medimg.data.prepare --raw-dir data/raw/nih --out-dir data/processed \
      --labels 8 --patient-fraction 0.3 --resize 256
Expected raw layout: Data_Entry_2017.csv plus the extracted images (any sub-folder depth).
"""
import argparse
import csv
import json
import os
import re
from collections import Counter

from PIL import Image

from medimg.config import LABELS_8, LABELS_14
from medimg.data.splits import assign_split, keep_patient


def parse_age(raw):
    m = re.search(r"\d+", str(raw))
    if not m:
        return ""
    age = int(m.group())
    return age if 0 < age < 120 else ""  # NIH file contains a few impossible ages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--labels", type=int, choices=(8, 14), default=8)
    ap.add_argument("--patient-fraction", type=float, default=1.0)
    ap.add_argument("--resize", type=int, default=256, help="0 = keep original size")
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--test-pct", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    labels = LABELS_8 if a.labels == 8 else LABELS_14
    csv_path = a.csv or os.path.join(a.raw_dir, "Data_Entry_2017.csv")
    files = {}
    for root, _, names in os.walk(a.raw_dir):
        for n in names:
            if n.lower().endswith(".png"):
                files[n] = os.path.join(root, n)

    img_out = os.path.join(a.out_dir, "images")
    os.makedirs(img_out, exist_ok=True)
    rows, missing = [], 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            pid = r["Patient ID"]
            if not keep_patient(pid, a.patient_fraction, a.seed):
                continue
            name = r["Image Index"]
            if name not in files:
                missing += 1
                continue
            findings = set(r["Finding Labels"].split("|"))
            dst = files[name]
            if a.resize:
                dst = os.path.join(img_out, name)
                if not os.path.exists(dst):
                    Image.open(files[name]).convert("L").resize((a.resize, a.resize), Image.BILINEAR).save(dst)
            row = {
                "image_path": dst, "patient_id": pid, "age": parse_age(r["Patient Age"]),
                "sex": r["Patient Gender"], "view": r["View Position"],
                "split": assign_split(pid, a.val_pct, a.test_pct, a.seed),
            }
            row.update({l: int(l in findings) for l in labels})
            rows.append(row)

    by_patient = {}
    for r in rows:
        by_patient.setdefault(r["patient_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_patient.values()), "patient leakage across splits!"

    manifest = os.path.join(a.out_dir, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    stats = {"images": len(rows), "missing_files": missing, "labels": labels}
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        stats[split] = {
            "images": len(sub),
            "patients": len({r["patient_id"] for r in sub}),
            "prevalence": {l: round(sum(r[l] for r in sub) / max(len(sub), 1), 4) for l in labels},
        }
    stats["view_counts"] = dict(Counter(r["view"] for r in rows))
    with open(os.path.join(a.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: stats[k] for k in ("images", "missing_files")}), "->", manifest)


if __name__ == "__main__":
    main()
