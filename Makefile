# Fichier : Makefile

# Variables
ENV_NAME=venv
PYTHON=$(ENV_NAME)/bin/python
PIP=$(ENV_NAME)/bin/pip
DATA=churn-bigml-80.csv
MODEL=churn_model.joblib
IMAGE_NAME=rayen_nom_4ds10_mlops
DOCKERHUB_USER=ton_nom_dutilisateur_dockerhub

.PHONY: install format lint security prepare train evaluate test mlflow-server run-api docker-build docker-run docker-push docker-all monitoring-start monitoring-stop clean clean-mlflow

# 1. Créer et activer l'environnement virtuel, installer les dépendances
install:
	@echo "📦 Création de l'environnement virtuel et installation des dépendances..."
	python3 -m venv $(ENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# 2. Formatage automatique (black)
format: install
	@echo "🎨 Formatage du code avec Black..."
	$(PYTHON) -m black .

# 3. Vérification qualité du code (flake8 + pylint)
lint: install
	@echo "🔍 Analyse du code avec flake8 et pylint..."
	$(PYTHON) -m flake8 main.py model_pipeline.py app.py
	$(PYTHON) -m pylint main.py model_pipeline.py app.py

# 4. Sécurité (bandit)
security: install
	@echo "🔐 Vérification de la sécurité du code avec Bandit..."
	$(PYTHON) -m bandit -r .

# 5. Préparer les données
prepare: install
	@echo "🧪 Préparation des données..."
	. $(ENV_NAME)/bin/activate && python main.py --data $(DATA) --prepare

# 6. Entraîner le modèle
train: install
	@echo "🧠 Entraînement complet du modèle..."
	. $(ENV_NAME)/bin/activate && python main.py --data $(DATA) --train --evaluate --save $(MODEL)

# 7. Évaluer le modèle chargé
evaluate: install
	@echo "📈 Évaluation du modèle sauvegardé..."
	. $(ENV_NAME)/bin/activate && python main.py --data $(DATA) --load --model $(MODEL)

# 8. Tests unitaires avec pytest
test: install
	@echo "🧪 Exécution des tests unitaires..."
	. $(ENV_NAME)/bin/activate && pytest tests/

# 9. Lancer l'interface MLflow
mlflow-server: install
	@echo "📊 Lancement de l'interface MLflow sur http://localhost:5000..."
	. $(ENV_NAME)/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# 10. Lancer l'API FastAPI
run-api: install
	@echo "🚀 Lancement de l'API FastAPI sur http://127.0.0.1:8000/docs..."
	. $(ENV_NAME)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 11. Construire l'image Docker
docker-build:
	@echo "🐳 Construction de l'image Docker..."
	docker build -t $(IMAGE_NAME) .

# 12. Exécuter le conteneur Docker
docker-run:
	@echo "🚢 Exécution du conteneur Docker..."
	docker run -p 8000:8000 --name fastapi_container $(IMAGE_NAME)

# 13. Pousser l'image sur Docker Hub
docker-push:
	@echo "☁️ Push de l'image sur Docker Hub..."
	docker tag $(IMAGE_NAME) $(DOCKERHUB_USER)/$(IMAGE_NAME):latest
	docker push $(DOCKERHUB_USER)/$(IMAGE_NAME):latest

# 14. Lancer Elasticsearch et Kibana
monitoring-start:
	@echo "🔍 Lancement de la stack Elasticsearch et Kibana..."
	docker-compose up -d

# 15. Arrêter Elasticsearch et Kibana
monitoring-stop:
	@echo "🛑 Arrêt de la stack Elasticsearch et Kibana..."
	docker-compose down

# 16. Tâche complète pour Docker
docker-all: docker-build docker-run docker-push

# 17. Exécution complète (sans Docker)
all: install format lint security prepare train evaluate test mlflow-server

# 18. Nettoyage partiel (MLflow)
clean-mlflow:
	@echo "🧹 Nettoyage des fichiers MLflow..."
	rm -rf mlruns mlflow.db

# 19. Nettoyage complet
clean: clean-mlflow
	@echo "🧹 Nettoyage des fichiers générés..."
	rm -rf $(MODEL) $(ENV_NAME) __pycache__ *.pyc

# Ouvrir le front
open-frontend:
	@echo "🌐 Ouverture du frontend..."
	wslview http://127.0.0.1:8000/static/index.html
