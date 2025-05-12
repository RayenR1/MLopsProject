from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from model_pipeline import prepare_data, balance_data, train_model, save_model
import mlflow
from datetime import datetime

# Chemins des fichiers
MODEL_PATH = "models/churn_model.joblib"
SCALER_PATH = "models/scaler.joblib"

# Charger le modèle et le scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur de chargement du modèle ou scaler : {e}")

# Initialiser FastAPI
app = FastAPI(
    title="API Churn Prediction",
    description="API pour prédire le churn des clients télécom",
    version="1.0.0"
)

# Monter le dossier static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modèle Pydantic pour la requête de prédiction
class PredictRequest(BaseModel):
    features: list[float]
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
                ]
            }
        }

# Modèle Pydantic pour la requête de réentraînement
class RetrainRequest(BaseModel):
    data_path: str
    class Config:
        schema_extra = {
            "example": {
                "data_path": "data/churn-bigml-80.csv"
            }
        }

@app.get("/health")
def health_check():
    """Vérifier l'état de l'API"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    """Prédire le churn à partir des features fournies"""
    try:
        # Valider le nombre de features
        expected_features = model.n_features_in_
        if len(req.features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Attendu {expected_features} features, reçu {len(req.features)}"
            )
        
        # Prétraiter et prédire
        features = np.array(req.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Logger dans MLflow
        with mlflow.start_run():
            mlflow.log_param("input_features", req.features)
            mlflow.log_metric("prediction", prediction)
            mlflow.set_tag("endpoint", "predict")
        
        return {"prediction": int(prediction), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")

@app.post("/retrain")
def retrain(req: RetrainRequest):
    """Réentraîner le modèle avec un nouveau fichier de données"""
    try:
        # Vérifier l'existence du fichier
        import os
        if not os.path.exists(req.data_path):
            raise HTTPException(status_code=400, detail=f"Fichier {req.data_path} introuvable")
        
        # Réentraîner
        X_train, X_test, y_train, y_test = prepare_data(req.data_path)
        X_res, y_res = balance_data(X_train, y_train)
        new_model = train_model(X_res, y_res)
        
        # Sauvegarder le nouveau modèle et scaler
        save_model(new_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        # Mettre à jour le modèle en mémoire
        global model
        model = new_model
        
        return {
            "message": "Modèle réentraîné et sauvegardé",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de réentraînement : {str(e)}")
