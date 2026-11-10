PY=PYTHONPATH=src python
.PHONY: setup data train train-smoke test mlflow-ui feast api producer drift drift-demo dashboard up replay k8s-build k8s-deploy k8s-demo
setup:        ; pip install -r requirements.txt
data:         ; $(PY) -m fraud.data --out data/synthetic.csv
train:        ; $(PY) -m fraud.train --data synthetic          # XGBoost + PyG GNN, logged to MLflow
train-ieee:   ; $(PY) -m fraud.train --data ieee --ieee-dir $(IEEE_DIR)
train-smote:  ; $(PY) -m fraud.train --data synthetic --imbalance smote --run-name smote
train-smoke:  ; $(PY) -m fraud.train --data synthetic --model sklearn --gnn ring   # no heavy deps
test:         ; $(PY) -m unittest discover -s tests -v
mlflow-ui:    ; mlflow ui --port 5000
feast:        ; $(PY) scripts/seed_feast.py
api:          ; STATE_BACKEND=$${STATE_BACKEND:-memory} KAFKA_BOOTSTRAP=$${KAFKA_BOOTSTRAP:-} $(PY) -m uvicorn fraud.service.api:app --port 8000
producer:     ; $(PY) -m fraud.service.producer --rate 50
drift:        ; $(PY) -m fraud.drift --retrain
drift-demo:   ; $(PY) -m fraud.service.producer --rate 200 --drift-after 3000 --limit 8000 && sleep 5 && $(PY) -m fraud.drift --retrain
dashboard:    ; streamlit run dashboard/app.py
up:           ; docker compose up --build -d
replay:       ; docker compose --profile replay up producer
k8s-build:    ; eval $$(minikube docker-env) && docker build -t fraud-api:latest . && docker build -f Dockerfile.dashboard -t fraud-dashboard:latest .
k8s-deploy:   ; kubectl apply -f k8s/
k8s-demo:     ; kubectl delete job fraud-producer --ignore-not-found && kubectl apply -f k8s/jobs/producer-job.yaml
