# 10-minute presentation outline
1. **Problem (1 min)** - fraud < 1% of transactions; cost of a miss vs cost of a false alert; why milliseconds matter.
2. **Design (2 min)** - the architecture diagram; the one decision to dwell on: one feature definition for training and serving (show the parity test).
3. **Data and modelling (2 min)** - imbalance, why PR-AUC not accuracy, chronological split, class weights vs SMOTE (your ablation).
4. **Graph / rings (1.5 min)** - what is linked, why hub entities are removed, what the GNN adds on top of XGBoost (your blend table).
5. **Live demo (2 min)** - producer -> alerts appearing -> analyst marks fraud -> drift demo -> retrain -> reload.
6. **Challenges (1 min)** - leakage, hub nodes, `user_txn_count` false drift, Kafka-on-k8s `KAFKA_PORT` collision (`enableServiceLinks: false`).
7. **Results and next steps (0.5 min)** - your numbers; feedback labels into retraining, streaming GNN, Feast on k8s.

# Demo script
`docker compose up -d` -> open dashboard -> `make replay` -> sort by score, open a case, mark fraud -> `make drift-demo`
-> show drift bars + "RETRAIN TRIGGERED" -> new model loaded (`/health` threshold changes) -> show MLflow run list.
