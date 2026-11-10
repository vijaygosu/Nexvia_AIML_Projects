"""Single source of truth for features.

* build_features(df)  - vectorised, used for TRAINING
* UserState           - incremental, used for SERVING (state lives in Feast online store)

tests/test_features.py asserts the two produce identical values on the same
stream, which is what guarantees there is no training/serving skew.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

UNKNOWN = -1
H1, H24 = 3600, 86400
FEATURES = ["amount", "amt_ratio", "txn_count_1h", "txn_count_24h", "time_since_last",
            "device_changed", "location_changed", "hour", "user_txn_count", "product"]


def _window_counts(ts: np.ndarray, user_idx: dict, window: int) -> np.ndarray:
    out = np.zeros(len(ts), dtype=np.int64)
    for idx in user_idx.values():                 # idx: positions, ascending (df is time-sorted)
        t = ts[idx]
        lo = np.searchsorted(t, t - window, side="left")
        out[idx] = np.arange(len(t)) - lo         # earlier txns of this user with t_j >= t - window
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-free batch features: every value only uses transactions strictly earlier in the stream."""
    df = df.sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True).copy()
    g = df.groupby("user_id", sort=False)
    cnt = g.cumcount()
    prior_mean = (g["amount"].cumsum() - df["amount"]) / cnt.where(cnt > 0)
    df["amt_ratio"] = (df["amount"] / prior_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    ts = df["ts"].to_numpy()
    idx = g.indices
    df["txn_count_1h"] = _window_counts(ts, idx, H1)
    df["txn_count_24h"] = _window_counts(ts, idx, H24)
    df["time_since_last"] = (df["ts"] - g["ts"].shift()).fillna(-1.0)

    for col, out in (("device_id", "device_changed"), ("location", "location_changed")):
        prev = g[col].shift()
        df[out] = ((df[col] != prev) & prev.notna() & (df[col] != UNKNOWN) & (prev != UNKNOWN)).astype(int)

    df["hour"] = (df["ts"] // 3600) % 24
    df["user_txn_count"] = cnt
    return df


@dataclass
class UserState:
    """Per-user running aggregates. This is the payload stored in Feast."""
    n: int = 0
    total: float = 0.0
    last_ts: int = -1
    last_device: int = UNKNOWN
    last_location: int = UNKNOWN
    recent_ts: list = field(default_factory=list)      # timestamps within the last 24h

    def features(self, t: dict) -> dict:
        ts, n = int(t["ts"]), self.n
        dev, loc = int(t.get("device_id", UNKNOWN)), int(t.get("location", UNKNOWN))
        ratio = t["amount"] / (self.total / n) if n > 0 and self.total > 0 else 1.0
        return {
            "amount": float(t["amount"]),
            "amt_ratio": float(ratio),
            "txn_count_1h": sum(1 for x in self.recent_ts if x >= ts - H1),
            "txn_count_24h": sum(1 for x in self.recent_ts if x >= ts - H24),
            "time_since_last": float(ts - self.last_ts) if n > 0 else -1.0,
            "device_changed": int(n > 0 and dev != UNKNOWN and self.last_device != UNKNOWN and dev != self.last_device),
            "location_changed": int(n > 0 and loc != UNKNOWN and self.last_location != UNKNOWN and loc != self.last_location),
            "hour": (ts // 3600) % 24,
            "user_txn_count": n,
            "product": int(t.get("product", 0)),
        }

    def update(self, t: dict) -> "UserState":
        ts = int(t["ts"])
        self.recent_ts = [x for x in self.recent_ts if x >= ts - H24] + [ts]
        self.n += 1
        self.total += float(t["amount"])
        self.last_ts = ts
        self.last_device = int(t.get("device_id", UNKNOWN))
        self.last_location = int(t.get("location", UNKNOWN))
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserState":
        return cls(n=int(d["n"]), total=float(d["total"]), last_ts=int(d["last_ts"]),
                   last_device=int(d["last_device"]), last_location=int(d["last_location"]),
                   recent_ts=[int(x) for x in (d.get("recent_ts") or [])])


class MemoryStateStore:
    """Dev/test backend with the same interface as the Feast-backed store."""

    def __init__(self, states: dict | None = None):
        self._s = dict(states or {})

    def get(self, user_id: int) -> UserState:
        s = self._s.get(int(user_id))
        return UserState.from_dict(s.to_dict()) if s else UserState()   # copy

    def put(self, user_id: int, state: UserState) -> None:
        self._s[int(user_id)] = state


def replay_states(df: pd.DataFrame) -> dict:
    """Fold a time-sorted history into per-user states (used to seed the online store)."""
    states: dict[int, UserState] = {}
    for r in df.sort_values(["ts", "txn_id"]).itertuples(index=False):
        t = {"ts": r.ts, "amount": r.amount, "device_id": r.device_id,
             "location": r.location, "product": r.product}
        states.setdefault(int(r.user_id), UserState()).update(t)
    return states
