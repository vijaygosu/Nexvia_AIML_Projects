# Results

> **Read this first.** The numbers below are a *smoke test* produced in a sandbox that had no XGBoost, PyTorch or
> Kafka: scikit-learn `HistGradientBoostingClassifier` stands in for XGBoost and the label-based connected-component
> "ring" baseline stands in for the GraphSAGE model, on **synthetic** data. They show the pipeline works and the
> methodology is sound; they are **not** the final XGBoost/PyG numbers and must not be presented as such.
> Re-run `make train` (and `make train-ieee`) on your machine and paste your own numbers into the table below -
> `docs/results/metrics.json`, `pr_curve.png`, `feature_importance.png` and the MLflow UI are regenerated.

## Sandbox smoke test (synthetic, 150k txns, 1.2% fraud, chronological 60/20/20 split)
Test = final 20% of time (30,000 txns, 277 fraud = 0.92%). Threshold chosen on validation.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Tabular (sklearn HGB, stand-in for XGBoost) | 0.798 | 0.996 | 0.774 | 0.643 | 0.702 |
| Graph ring baseline alone (stand-in for GNN) | 0.030 | 0.793 | 0.003 | 0.011 | 0.005 |
| **Blend (w_xgb = 0.9)** | **0.814** | 0.997 | 0.786 | 0.729 | 0.757 |

No-skill PR-AUC = fraud rate = 0.009. Blend confusion matrix on test: TP 202, FP 55, FN 75, TN 29,668.

Other verified facts (all reproducible with `make test` / the commands in the README):
* **Training/serving parity:** online `UserState` features vs vectorised training features: max abs difference 8e-15 over 6,000 replayed test transactions.
* **No leakage:** changing the last 50 transactions leaves all earlier feature rows byte-identical (unit test).
* **Scoring latency (in-process, in-memory state, sklearn model):** p50 1.7 ms, p95 3.3 ms, p99 3.9 ms. This excludes Kafka and Feast; measure the real path with `/stats` (`latency_ms` and `e2e_ms`).
* **Drift check:** an unshifted window flags no monitored feature; a window with 4x amounts / always-new devices flags `amount, amt_ratio, device_changed, location_changed` and triggers retraining.

## Your results (fill in after running on your machine)
| Run | Dataset | Imbalance | Model | PR-AUC (test) | Precision | Recall | Notes |
|---|---|---|---|---|---|---|---|
| 1 | IEEE-CIS | class_weight | XGBoost | | | | baseline |
| 2 | IEEE-CIS | smote | XGBoost | | | | ablation |
| 3 | IEEE-CIS | class_weight | XGBoost + GNN blend | | | | final |

Also report: p50/p95/p99 from `/stats`, drift demo screenshot, and what the GNN adds (ablation: XGBoost vs GNN vs blend).
