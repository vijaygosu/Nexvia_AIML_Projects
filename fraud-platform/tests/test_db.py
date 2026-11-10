import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fraud.features import FEATURES
from fraud.service.db import DB


def _res(i, alert, score):
    return {"txn_id": i, "user_id": i, "ts": i, "amount": 10.0, "fraud_score": score, "xgb_score": score,
            "gnn_score": None, "is_alert": alert, "latency_ms": 1.0 + i, "features": {f: 0.0 for f in FEATURES}}


class TestDB(unittest.TestCase):
    def setUp(self):
        self.db = DB(os.path.join(tempfile.mkdtemp(), "t.db"))

    def test_case_lifecycle_and_stats(self):
        self.db.log(_res(1, True, 0.9), sim_label=1, e2e_ms=5)
        self.db.log(_res(2, True, 0.8), sim_label=0, e2e_ms=7)
        self.db.log(_res(3, False, 0.1), sim_label=1, e2e_ms=6)      # missed fraud
        a = self.db.alerts("open")
        self.assertEqual([c["txn_id"] for c in a], [1, 2])             # sorted by risk
        self.assertTrue(self.db.feedback(a[0]["case_id"], "fraud", "me", "ok"))
        self.assertFalse(self.db.feedback(999, "fraud", "me", ""))
        s = self.db.stats()
        self.assertEqual((s["scored"], s["alerts"], s["open_cases"], s["closed_cases"], s["confirmed_fraud"]), (3, 2, 1, 1, 1))
        self.assertAlmostEqual(s["sim_alert_precision"], 0.5)
        self.assertAlmostEqual(s["sim_recall"], 0.5)
        self.assertEqual(len(self.db.labelled_feedback()), 1)


if __name__ == "__main__":
    unittest.main()
