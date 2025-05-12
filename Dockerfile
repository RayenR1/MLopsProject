# Utiliser une image de base Python 3.9 slim pour réduire la taille
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY main.py .
COPY model_pipeline.py .
COPY app.py .
COPY data/ ./data/
COPY models/ ./models/

# Installer les dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Exposer le port 8000 pour FastAPI
EXPOSE 8000

# Lancer l'application FastAPI avec uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
