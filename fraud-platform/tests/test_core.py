import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from fraud.data import make_synthetic, CANON
from fraud.features import FEATURES, build_features, UserState, MemoryStateStore
from fraud.graph import build_graph, user_labels, ring_scores
from fraud.drift import psi, run as drift_run
from fraud.metrics import evaluate, best_f1_threshold
from fraud.serving import Scorer

DF = make_synthetic(n_users=600, n_txn=12_000, n_rings=8, seed=3)


class TestData(unittest.TestCase):
    def test_schema_and_imbalance(self):
        self.assertEqual(list(DF.columns), CANON)
        self.assertTrue(0.005 < DF.is_fraud.mean() < 0.03)      # heavy imbalance, ~1%
        self.assertTrue(DF.ts.is_monotonic_increasing)
        self.assertEqual(DF.txn_id.nunique(), len(DF))


class TestFeatures(unittest.TestCase):
    def test_training_serving_parity(self):
        """Vectorised (training) features == incremental (serving) features."""
        batch = build_features(DF)
        store, rows = MemoryStateStore(), []
        for r in batch.itertuples(index=False):
            t = r._asdict()
            st = store.get(t["user_id"])
            rows.append(st.features(t))
            store.put(t["user_id"], st.update(t))
        online = pd.DataFrame(rows)
        for c in FEATURES:
            np.testing.assert_allclose(online[c].to_numpy(float), batch[c].to_numpy(float),
                                       rtol=1e-9, atol=1e-9, err_msg=c)

    def test_no_future_leakage(self):
        """Mutating a later transaction must not change earlier features."""
        base = build_features(DF)
        mod = DF.copy()
        k = len(mod) - 50
        mod.loc[k:, "amount"] *= 100
        mod.loc[k:, "device_id"] = 999_999
        after = build_features(mod)
        pd.testing.assert_frame_equal(base.loc[:k - 1, FEATURES], after.loc[:k - 1, FEATURES])

    def test_first_txn_defaults(self):
        f = UserState().features({"ts": 1000, "amount": 50.0, "device_id": 1, "location": 1, "product": 0})
        self.assertEqual((f["amt_ratio"], f["time_since_last"], f["device_changed"], f["txn_count_1h"]), (1.0, -1.0, 0, 0))

    def test_velocity_and_device_change(self):
        s = UserState()
        for i in range(5):
            s.update({"ts": 1000 + i * 60, "amount": 10, "device_id": 1, "location": 1})
        f = s.features({"ts": 1400, "amount": 100, "device_id": 2, "location": 1, "product": 0})
        self.assertEqual(f["txn_count_1h"], 5)
        self.assertEqual(f["device_changed"], 1)
        self.assertEqual(f["location_changed"], 0)
        self.assertAlmostEqual(f["amt_ratio"], 10.0)


class TestGraph(unittest.TestCase):
    def test_high_degree_entities_dropped(self):
        df = pd.DataFrame({"user_id": np.arange(100), "device_id": np.arange(100),
                           "email_domain": 0, "ts": 0, "amount": 1.0, "is_fraud": 0})
        users, edges, n = build_graph(df, max_degree=50)
        self.assertEqual(n, 100 + 100)                          # the shared 'email 0' node was removed

    def test_ring_baseline_ranks_ring_members_higher(self):
        users, edges, n = build_graph(DF)
        t = DF.ts.quantile(0.6)
        y, mask = user_labels(DF, users, t)
        s = ring_scores(users, edges, n, y, mask)
        fut = DF[DF.ts >= t]
        fut_fraud_users = set(fut.loc[fut.is_fraud == 1, "user_id"])
        is_f = np.array([u in fut_fraud_users for u in users])
        self.assertGreater(s[is_f].mean(), s[~is_f].mean())


class TestDrift(unittest.TestCase):
    def test_psi(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 5000)
        self.assertLess(psi(a, rng.normal(0, 1, 5000)), 0.05)
        self.assertGreater(psi(a, rng.normal(1.5, 1, 5000)), 0.5)
        self.assertGreater(psi(np.r_[np.zeros(900), np.ones(100)], np.r_[np.zeros(500), np.ones(500)]), 0.2)

    def test_trigger(self):
        ref = build_features(DF)[FEATURES]
        same = drift_run(ref, ref.sample(1000, random_state=1, replace=True), out_dir=tempfile.mkdtemp())
        self.assertFalse(same["retrain_triggered"])
        shifted = ref.sample(1000, random_state=1, replace=True).copy()
        shifted["amount"] *= 5; shifted["amt_ratio"] *= 4; shifted["device_changed"] = 1
        shifted["location_changed"] = 1; shifted["txn_count_1h"] += 6; shifted["txn_count_24h"] += 9
        self.assertTrue(drift_run(ref, shifted, out_dir=tempfile.mkdtemp())["retrain_triggered"])


class _Dummy:
    def predict_proba(self, X):                                  # score grows with amt_ratio & device change
        p = np.clip(0.02 + 0.1 * (X["amt_ratio"] - 1).clip(lower=0) + 0.5 * X["device_changed"], 0, 1).to_numpy()
        return np.column_stack([1 - p, p])


class TestScorer(unittest.TestCase):
    def test_score_updates_state_and_alerts(self):
        meta = {"features": FEATURES, "threshold": 0.4, "blend_w": 0.5}
        sc = Scorer(_Dummy(), meta, {7: 0.9}, MemoryStateStore())
        base = {"txn_id": 1, "user_id": 7, "ts": 1000, "amount": 20.0, "device_id": 1, "location": 1, "product": 0}
        r1 = sc.score(base)
        r2 = sc.score({**base, "txn_id": 2, "ts": 1100, "device_id": 2, "amount": 200.0})
        self.assertFalse(r1["is_alert"] and r1["gnn_score"] is None)
        self.assertEqual(r2["features"]["device_changed"], 1)
        self.assertEqual(r2["features"]["user_txn_count"], 1)    # state advanced after txn 1
        self.assertTrue(r2["is_alert"])
        r3 = sc.score({**base, "txn_id": 3, "user_id": 99})      # unseen account -> cold-start fallback
        self.assertIsNone(r3["gnn_score"])
        self.assertAlmostEqual(r3["fraud_score"], r3["xgb_score"])


class TestMetrics(unittest.TestCase):
    def test_eval(self):
        y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .3, .8, .9])
        r = evaluate(y, p, best_f1_threshold(y, p))
        self.assertEqual((r["fp"], r["fn"]), (0, 0)); self.assertEqual(r["pr_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
