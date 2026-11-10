# API, Kafka producer and drift/retrain job all share this image (different commands).
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY src ./src
COPY models ./models
COPY data/replay.csv data/reference_features.csv ./data/
COPY feature_repo ./feature_repo
ENV PYTHONPATH=/app/src MODEL_DIR=/app/models DB_PATH=/app/data/cases.db REPORT_DIR=/app/monitoring/reports
RUN mkdir -p monitoring/reports
EXPOSE 8000
CMD ["uvicorn", "fraud.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
