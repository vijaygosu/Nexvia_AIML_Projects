# Real-Time Financial Fraud Detection Platform

Reads live transactions from Kafka, scores each one in milliseconds with an XGBoost + graph-neural-network blend,
links suspicious accounts into fraud rings, monitors drift (Evidently) with an automatic retraining trigger,
and gives analysts a Streamlit console for alerts, case status and feedback labels.

**Stack:** Kafka · XGBoost · PyTorch Geometric · Feast · MLflow · FastAPI · Kubernetes (Minikube) · Evidently AI · Streamlit

Architecture and design decisions: [`docs/architecture.md`](docs/architecture.md) · API and data dictionary: [`docs/API.md`](docs/API.md) · Results: [`docs/RESULTS.md`](docs/RESULTS.md)

## Repository layout
```
src/fraud/data.py          canonical schema, synthetic generator (planted rings), IEEE-CIS adapter
src/fraud/features.py      build_features (training) + UserState (serving): one definition, parity-tested
src/fraud/graph.py         account-device-email graph, GraphSAGE (PyG), connected-component ring baseline
src/fraud/train.py         XGBoost + GNN + blend + threshold, MLflow tracking, writes serving artifacts
src/fraud/serving.py       online Scorer (Feast or in-memory state), cold-start fallback
src/fraud/drift.py         Evidently + PSI drift check, retrain trigger, hot reload
src/fraud/service/         FastAPI app + Kafka consumer, SQLite cases DB, Kafka producer (replay)
dashboard/app.py           Streamlit analyst console
feature_repo/              Feast definitions (entity, push source, feature view)
k8s/  Dockerfile*  docker-compose.yml  Makefile  tests/
```

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
make setup                      # pip install -r requirements.txt
make test                       # 12 unit tests (feature parity, leakage, graph, drift, scorer, DB)
python scripts/eda.py --data synthetic     # or --data ieee --ieee-dir ...

# Data: synthetic works out of the box. For IEEE-CIS download from Kaggle, then:  make train-ieee IEEE_DIR=/path/to/ieee-fraud-detection
make train                      # XGBoost + PyG GNN -> models/, data/replay.csv, docs/results/*, MLflow run
make mlflow-ui                  # http://localhost:5000  (runs, params, PR-AUC, models)
make feast                      # optional: load account state into Feast (then STATE_BACKEND=feast)

docker compose up --build -d    # Kafka + API (+ Kafka consumer) + dashboard + MLflow
make replay                     # producer streams the held-out window into Kafka
# dashboard http://localhost:8501   API docs http://localhost:8000/docs
make drift-demo                 # injects drift mid-stream -> Evidently report -> retrain -> hot reload
```
No Docker/GPU? `make train-smoke` trains with scikit-learn + the ring baseline (no xgboost/torch needed) and
`make api` serves it directly.

## Kubernetes (Minikube)
```bash
minikube start --cpus 4 --memory 6g
make train                      # models/ and data/*.csv must exist before building the image
make k8s-build                  # builds images inside Minikube's Docker
make k8s-deploy                 # Kafka, API+consumer, dashboard, MLflow, drift CronJob, shared volume
make k8s-demo                   # runs the producer Job (10k transactions)
minikube service fraud-dashboard --url     # dashboard      (API: service fraud-api, MLflow: service mlflow)
kubectl get pods,cronjob
```

## How the requirements are met
| Requirement | Where |
|---|---|
| EDA, imbalance (<1% fraud) | `scripts/eda.py` -> `docs/results/eda.png` (class balance, amount, velocity, device/location-change fraud rates) |
| Velocity, amount vs average, device/location change | `features.py` |
| Feast so training/serving share features | `feature_repo/`, `FeastStateStore`, `test_training_serving_parity` |
| XGBoost with class weights or SMOTE, PR-AUC, MLflow | `train.py` (`--imbalance class_weight\|smote`), `eval_metric=aucpr`, `mlflow.*` |
| Graph + GNN, blended with XGBoost | `graph.py`, blend weight tuned on validation |
| Kafka producer + FastAPI consumer | `service/producer.py`, `service/api.py` |
| Evidently drift + automatic retraining | `drift.py`, `k8s/50-drift-cronjob.yaml`, `/admin/reload` |
| Analyst dashboard with alerts, case status, feedback | `dashboard/app.py`, `/alerts`, `/alerts/{id}/feedback` |
| Kubernetes | `k8s/` |

## Known limitations / next steps
* Analyst feedback labels are stored (`DB.labelled_feedback`) but not yet merged into the retraining set.
* GNN scores are precomputed per account (batch refresh); streaming GNN inference and incremental graph updates are future work.
* On Kubernetes the API uses the in-memory state backend by default (Feast is demonstrated locally); a hot reload resets that in-memory state to the training snapshot.
* The retrain job in the cluster uses the XGBoost + ring baseline (the serving image does not ship torch).
