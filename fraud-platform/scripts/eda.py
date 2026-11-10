"""EDA: class imbalance, amount/hour/velocity behaviour of fraud. Works on the canonical schema.
   python scripts/eda.py --data synthetic     |     python scripts/eda.py --data ieee --ieee-dir DIR
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fraud.data import make_synthetic, load_ieee
from fraud.features import build_features

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="synthetic"); ap.add_argument("--ieee-dir"); ap.add_argument("--out", default="docs/results")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
raw = make_synthetic() if a.data == "synthetic" else load_ieee(f"{a.ieee_dir}/train_transaction.csv", f"{a.ieee_dir}/train_identity.csv")
df = build_features(raw)
n, f = len(df), int(df.is_fraud.sum())
print(f"rows={n:,}  fraud={f:,} ({f/n:.3%})  imbalance ~ 1:{(n-f)//max(f,1)}  accounts={df.user_id.nunique():,}")
print(df.groupby("is_fraud")[["amount", "amt_ratio", "txn_count_1h", "device_changed", "location_changed"]].mean().round(3).to_string())

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].bar(["legit", "fraud"], [n - f, f]); ax[0,0].set_yscale("log"); ax[0,0].set_title(f"Class balance ({f/n:.2%} fraud, log scale)")
for lab, c in ((0, "tab:blue"), (1, "tab:red")):
    ax[0,1].hist(np.log10(df.loc[df.is_fraud == lab, "amount"]), bins=40, alpha=.6, density=True, color=c, label=["legit", "fraud"][lab])
ax[0,1].set_title("log10(amount)"); ax[0,1].legend()
ax[0,2].hist([np.clip(df.loc[df.is_fraud == l, "amt_ratio"], 0, 10) for l in (0, 1)], bins=30, density=True, label=["legit", "fraud"]); ax[0,2].set_title("amount / account prior mean (clipped 10)"); ax[0,2].legend()
h = df.groupby("hour").is_fraud.mean(); ax[1,0].bar(h.index, h.values); ax[1,0].set_title("fraud rate by hour")
d = df.groupby(["device_changed", "location_changed"]).is_fraud.mean().unstack(); d.plot.bar(ax=ax[1,1]); ax[1,1].set_title("fraud rate by device/location change"); ax[1,1].set_xlabel("device_changed")
v = df.assign(v=df.txn_count_1h.clip(0, 6)).groupby("v").is_fraud.mean(); ax[1,2].bar(v.index, v.values); ax[1,2].set_title("fraud rate by txns in last hour")
fig.tight_layout(); fig.savefig(f"{a.out}/eda.png", dpi=120); print("saved", f"{a.out}/eda.png")
