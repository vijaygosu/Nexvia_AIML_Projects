# API reference (FastAPI - interactive docs at `http://localhost:8000/docs`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + active threshold / blend weight |
| POST | `/score` | score one transaction synchronously |
| GET | `/alerts?status=open\|closed&limit=200` | analyst queue, highest risk first |
| POST | `/alerts/{case_id}/feedback` | analyst label `fraud`/`legit` (closes the case) |
| GET | `/stats` | volumes, open/closed cases, latency p50/p95/p99 (scoring + end-to-end), simulation precision/recall |
| GET | `/drift` | latest drift report (`monitoring/reports/latest.json`) |
| POST | `/admin/reload` | hot-load retrained artifacts from `MODEL_DIR` |

## `POST /score`
```json
{"txn_id": 1, "ts": 5184000, "user_id": 42, "amount": 129.99,
 "device_id": 7, "email_domain": 3, "location": 12, "product": 1}
```
Response
```json
{"txn_id": 1, "fraud_score": 0.93, "xgb_score": 0.97, "gnn_score": 0.62, "is_alert": true, "latency_ms": 2.1}
```
`gnn_score` is `null` for accounts not in the graph (cold start; the blend then uses the XGBoost score).

## `POST /alerts/{id}/feedback`
```json
{"label": "fraud", "analyst": "priya", "note": "device shared with 9 other accounts"}
```

## Kafka topics
| Topic | Producer | Value |
|---|---|---|
| `transactions` | `fraud.service.producer` | canonical transaction + `produced_at` (+ `is_fraud` used only for simulation metrics) |
| `fraud-scores` | API consumer thread | `{txn_id, user_id, fraud_score, is_alert, latency_ms}` |

## Data dictionary (canonical schema)
| Column | Meaning | IEEE-CIS source |
|---|---|---|
| txn_id, ts | id, seconds on a monotonic clock | TransactionID, TransactionDT |
| user_id | account/card identity | `card1\|card2\|addr1` |
| amount | transaction amount | TransactionAmt |
| device_id, email_domain, location, product | integer-coded categoricals, -1 = unknown | DeviceInfo, P_emaildomain, addr1, ProductCD |
| is_fraud | label | isFraud |

## Model features (`fraud.features.FEATURES`)
`amount`, `amt_ratio` (amount / account's prior mean), `txn_count_1h`, `txn_count_24h` (velocity), `time_since_last`, `device_changed`, `location_changed`, `hour`, `user_txn_count`, `product`.
