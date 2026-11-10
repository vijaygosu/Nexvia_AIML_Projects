"""Online scorer: Feast/memory state -> shared feature code -> XGBoost -> blend with GNN."""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pandas as pd

from .features import FEATURES, UserState, MemoryStateStore
from .model_io import load_model

STATE_FIELDS = ["n", "total", "last_ts", "last_device", "last_location", "recent_ts"]


class FeastStateStore:
    """Reads/writes UserState through Feast (online read, push for writes)."""

    def __init__(self, repo_path: str):
        from feast import FeatureStore
        self.fs = FeatureStore(repo_path=repo_path)
        self.refs = [f"user_state:{c}" for c in STATE_FIELDS]

    def get(self, user_id: int) -> UserState:
        r = self.fs.get_online_features(features=self.refs,
                                        entity_rows=[{"user_id": int(user_id)}]).to_dict()
        if r["n"][0] is None:
            return UserState()
        return UserState.from_dict({k: r[k][0] for k in STATE_FIELDS})

    def put(self, user_id: int, state: UserState) -> None:
        from feast.data_source import PushMode
        now = pd.Timestamp.utcnow().tz_localize(None)
        row = {"user_id": int(user_id), **state.to_dict(), "event_timestamp": now, "created": now}
        self.fs.push("user_state_push", pd.DataFrame([row]), to=PushMode.ONLINE)


class Scorer:
    def __init__(self, model, meta: dict, gnn: dict, store):
        self.model, self.meta, self.gnn, self.store = model, meta, gnn, store
        self.features = meta.get("features", FEATURES)
        self.threshold = float(meta["threshold"])
        self.w = float(meta["blend_w"])
        self._lock = threading.Lock()

    @classmethod
    def from_dir(cls, model_dir="models", backend=None, feast_repo="feature_repo"):
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        model = load_model(model_dir, meta["model_file"])
        g = pd.read_csv(os.path.join(model_dir, "gnn_scores.csv"))
        gnn = dict(zip(g["user_id"].astype(int), g["gnn_score"]))
        backend = backend or os.getenv("STATE_BACKEND", "memory")
        if backend == "feast":
            store = FeastStateStore(feast_repo)
        else:
            s = pd.read_csv(os.path.join(model_dir, "user_state.csv"))
            states = {int(r.user_id): UserState.from_dict(
                {**r._asdict(), "recent_ts": json.loads(r.recent_ts)}) for r in s.itertuples(index=False)}
            store = MemoryStateStore(states)
        return cls(model, meta, gnn, store)

    def score(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        uid = int(txn["user_id"])
        with self._lock:                                  # per-user state must update in order
            state = self.store.get(uid)
            feats = state.features(txn)
            x = pd.DataFrame([[feats[c] for c in self.features]], columns=self.features)
            p_x = float(self.model.predict_proba(x)[0, 1])
            p_g = self.gnn.get(uid)
            p_g_eff = p_x if p_g is None else float(p_g)   # cold start: unseen account -> trust XGBoost
            p = self.w * p_x + (1 - self.w) * p_g_eff
            self.store.put(uid, state.update(txn))
        return {
            "txn_id": int(txn["txn_id"]), "user_id": uid, "ts": int(txn["ts"]),
            "amount": float(txn["amount"]),
            "fraud_score": p, "xgb_score": p_x,
            "gnn_score": None if p_g is None else float(p_g),
            "is_alert": bool(p >= self.threshold),
            "features": feats,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
