"""Held-out test evaluation: per-label AUROC (+95% bootstrap CI), calibration (ECE), subgroup AUROC,
reliability diagram and Grad-CAM failure cases for the presentation.

  python -m medimg.train.evaluate --ckpt models/densenet121_nih.pt --manifest data/processed/manifest.csv
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from medimg.data.dataset import ChestXrayDataset
from medimg.data.transforms import eval_transform
from medimg.explain.gradcam import GradCAM, blend
from medimg.models.classifier import load_checkpoint


def auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def bootstrap_ci(y, p, n=200, seed=0):
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def ece(y, p, bins=10):
    edges, total, err = np.linspace(0, 1, bins + 1), len(y), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            err += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return float(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/densenet121_nih.pt")
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--out", default="reports/eval_metrics.json")
    ap.add_argument("--fail-dir", default="reports/failure_cases")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(a.ckpt, dev)
    labels = ck["labels"]
    ds = ChestXrayDataset(a.manifest, "test", labels, eval_transform(ck["img_size"]))
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in DataLoader(ds, a.batch_size, num_workers=a.workers):
            ps.append(torch.sigmoid(model(x.to(dev))).cpu())
            ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()

    per = {l: {"auroc": auc(y[:, i], p[:, i]), "ci95": bootstrap_ci(y[:, i], p[:, i]),
               "ece": ece(y[:, i], p[:, i]), "prevalence": float(y[:, i].mean())} for i, l in enumerate(labels)}
    valid = [v["auroc"] for v in per.values() if not np.isnan(v["auroc"])]

    rows = ds.rows
    sex = np.array([r["sex"] for r in rows])
    age = np.array([float(r["age"]) if r["age"] else np.nan for r in rows])
    groups = {"sex=F": sex == "F", "sex=M": sex == "M", "age<40": age < 40,
              "age40-64": (age >= 40) & (age < 65), "age>=65": age >= 65}
    sub = {}
    for g, m in groups.items():
        a_ = [auc(y[m, i], p[m, i]) for i in range(len(labels)) if m.sum() > 0]
        a_ = [v for v in a_ if not np.isnan(v)]
        sub[g] = {"n": int(m.sum()), "mean_auroc": float(np.mean(a_)) if a_ else None}

    result = {"version": ck["version"], "n_test": len(ds), "mean_auroc": float(np.mean(valid)),
              "per_label": per, "subgroups": sub}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)

    # reliability diagram (all labels pooled)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yf, pf = y.ravel(), p.ravel()
        edges = np.linspace(0, 1, 11)
        xs, acc = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pf > lo) & (pf <= hi)
            if m.any():
                xs.append(pf[m].mean()); acc.append(yf[m].mean())
        plt.figure(figsize=(4, 4)); plt.plot([0, 1], [0, 1], "--", c="grey"); plt.plot(xs, acc, "o-")
        plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency"); plt.title("Reliability (test)")
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(a.out) or ".", "calibration.png"), dpi=150)
    except ImportError:
        print("matplotlib missing: skipping calibration plot")

    # Grad-CAM on the most confident false positives (shortcut / failure-case analysis)
    os.makedirs(a.fail_dir, exist_ok=True)
    cam = GradCAM(model)
    fp = [(p[i, j], i, j) for i in range(len(ds)) for j in range(len(labels)) if y[i, j] == 0]
    for rank, (prob, i, j) in enumerate(sorted(fp, reverse=True)[:8]):
        x, _, _ = ds[i]
        c, _, _ = cam(x[None].to(dev), class_idx=j)
        gray = np.array(Image.open(rows[i]["image_path"]).convert("L").resize((ck["img_size"],) * 2))
        Image.fromarray(blend(gray, c[0].numpy())).save(
            os.path.join(a.fail_dir, f"fp{rank:02d}_{labels[j]}_p{prob:.2f}.png"))
    print(json.dumps({"mean_auroc": result["mean_auroc"], "n_test": len(ds)}))


if __name__ == "__main__":
    main()
