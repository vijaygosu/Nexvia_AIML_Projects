"""Canonical transaction schema, a synthetic generator with planted fraud rings,
and an adapter for the IEEE-CIS Kaggle dataset.

Every downstream module only sees the canonical schema (CANON), so swapping the
dataset never touches feature / model / serving code.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

CANON = ["txn_id", "ts", "user_id", "amount", "device_id", "email_domain",
         "location", "product", "is_fraud"]
UNKNOWN = -1


def make_synthetic(n_users=4000, n_txn=150_000, fraud_rate=0.012, n_rings=30,
                   days=60, seed=7) -> pd.DataFrame:
    """PaySim/IEEE-like data: ~1% fraud, 80% of it from rings that share devices."""
    rng = np.random.default_rng(seed)
    horizon = days * 86400
    home_loc = rng.integers(0, 300, n_users)
    scale = rng.lognormal(3.6, 0.6, n_users)
    email = rng.integers(0, 40, n_users)          # common domains (high degree)
    own_dev = np.arange(n_users)
    shared = rng.choice(n_users, size=int(0.03 * n_users), replace=False)
    half = len(shared) // 2                       # 3% of users share a device (family PCs)
    own_dev[shared[:half]] = own_dev[shared[half:2 * half]]
    w = rng.pareto(2.0, n_users) + 1.0
    p = w / w.sum()

    n_fraud = int(n_txn * fraud_rate)
    n_legit = n_txn - n_fraud

    # ---- legitimate traffic
    u = rng.choice(n_users, n_legit, p=p)
    legit = pd.DataFrame({
        "ts": rng.integers(0, horizon, n_legit),
        "user_id": u,
        "amount": scale[u] * rng.lognormal(0, 0.5, n_legit),
        "device_id": np.where(rng.random(n_legit) < 0.04,
                              100_000 + rng.integers(0, 50_000, n_legit), own_dev[u]),
        "email_domain": email[u],
        "location": np.where(rng.random(n_legit) < 0.06,
                             rng.integers(0, 300, n_legit), home_loc[u]),
        "product": rng.integers(0, 5, n_legit),
        "is_fraud": 0,
    })

    # ---- ring fraud: compromised accounts, shared devices, 3 one-day bursts per ring
    nr = int(n_fraud * 0.8)
    ni = n_fraud - nr
    members = [rng.choice(n_users, rng.integers(5, 13), replace=False) for _ in range(n_rings)]
    starts = rng.integers(0, horizon - 86400, (n_rings, 3))
    ring_of = rng.integers(0, n_rings, nr)
    burst = rng.integers(0, 3, nr)
    fu = np.array([rng.choice(members[r]) for r in ring_of])
    ring = pd.DataFrame({
        "ts": starts[ring_of, burst] + rng.integers(0, 86400, nr),
        "user_id": fu,
        "amount": scale[fu] * rng.lognormal(1.3, 0.6, nr),
        "device_id": 10_000 + ring_of * 3 + rng.integers(0, 3, nr),
        "email_domain": 1000 + ring_of,
        "location": rng.integers(0, 300, nr),
        "product": rng.integers(0, 5, nr),
        "is_fraud": 1,
    })

    # ---- isolated fraud: one-off account takeover on a brand-new device
    fu2 = rng.choice(n_users, ni, p=p)
    iso = pd.DataFrame({
        "ts": rng.integers(0, horizon, ni),
        "user_id": fu2,
        "amount": scale[fu2] * rng.lognormal(1.2, 0.7, ni),
        "device_id": 200_000 + np.arange(ni),
        "email_domain": email[fu2],
        "location": np.where(rng.random(ni) < 0.7, rng.integers(0, 300, ni), home_loc[fu2]),
        "product": rng.integers(0, 5, ni),
        "is_fraud": 1,
    })

    df = pd.concat([legit, ring, iso], ignore_index=True)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df))
    df["ts"] = df["ts"].astype("int64")
    df["amount"] = df["amount"].round(2).clip(lower=0.5)
    return df[CANON].astype({c: "int64" for c in CANON if c != "amount"})


def load_ieee(trans_csv: str, ident_csv: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Adapter for IEEE-CIS (train_transaction.csv [+ train_identity.csv]).

    user_id = card1|card2|addr1 (common community 'uid' heuristic), device from
    DeviceInfo, email from P_emaildomain, location from addr1.
    """
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "card2",
            "addr1", "P_emaildomain", "ProductCD", "isFraud"]
    t = pd.read_csv(trans_csv, usecols=cols, nrows=nrows)
    if ident_csv:
        i = pd.read_csv(ident_csv, usecols=["TransactionID", "DeviceInfo"])
        t = t.merge(i, on="TransactionID", how="left")
    else:
        t["DeviceInfo"] = np.nan

    def codes(s):
        c, _ = pd.factorize(s)           # NaN -> -1 == UNKNOWN
        return c.astype("int64")

    uid = t["card1"].astype(str) + "|" + t["card2"].astype(str) + "|" + t["addr1"].astype(str)
    out = pd.DataFrame({
        "txn_id": t["TransactionID"].astype("int64"),
        "ts": t["TransactionDT"].astype("int64"),
        "user_id": codes(uid),
        "amount": t["TransactionAmt"].astype(float),
        "device_id": codes(t["DeviceInfo"]),
        "email_domain": codes(t["P_emaildomain"]),
        "location": codes(t["addr1"]),
        "product": codes(t["ProductCD"]),
        "is_fraud": t["isFraud"].astype("int64"),
    })
    return out.sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True)[CANON]


def split_times(df: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    """Chronological split points (never random: random splits leak future behaviour)."""
    return df["ts"].quantile(train_frac), df["ts"].quantile(train_frac + val_frac)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.csv")
    ap.add_argument("--n-txn", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    d = make_synthetic(n_txn=a.n_txn, seed=a.seed)
    d.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(d):,} rows, fraud rate {d.is_fraud.mean():.3%}")
