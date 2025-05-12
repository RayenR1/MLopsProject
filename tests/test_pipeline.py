# tests/test_pipeline.py

import pytest
import numpy as np
from model_pipeline import prepare_data, load_model, save_model
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    return "data/churn-bigml-80.csv"

def test_prepare_data(sample_data):
    X_train, X_test, y_train, y_test = prepare_data(sample_data)
    assert X_train.shape[0] > 0, "X_train est vide"
    assert X_test.shape[0] > 0, "X_test est vide"
    assert y_train.shape[0] > 0, "y_train est vide"
    assert y_test.shape[0] > 0, "y_test est vide"
    assert X_train.shape[1] == X_test.shape[1], "Nombre de features différent entre X_train et X_test"

def test_save_load_model(tmp_path):
    # Créer un modèle factice
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model_path = str(tmp_path / "test_model.joblib")

    # Sauvegarder le modèle
    save_model(model, model_path)

    # Charger le modèle
    loaded_model = load_model(model_path)

    assert isinstance(loaded_model, RandomForestClassifier), "Le modèle chargé n'est pas un RandomForestClassifier"
    assert loaded_model.n_estimators == 10, "Les hyperparamètres ne correspondent pas"
