.PHONY: test up down demo-data
test:
	pytest -q
up:
	docker compose up --build -d
down:
	docker compose down
demo-data:
	python scripts/upload_to_orthanc.py --src data/demo_images --orthanc http://localhost:8042
