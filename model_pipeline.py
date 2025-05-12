import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import joblib
import mlflow
import mlflow.sklearn
import mlflow.data
import matplotlib.pyplot as plt
import seaborn as sns
import json
import psutil
import time
import logging
import traceback
from elasticsearch import Elasticsearch
from logging.handlers import QueueHandler
import queue

# Configurer le logging avec Elasticsearch
logging.basicConfig(level=logging.INFO, filename="training.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Détecter si le code s'exécute dans GitHub Actions
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

# Initialiser Elasticsearch avec des tentatives (uniquement si pas dans GitHub Actions)
es = None
max_attempts = 10
attempt = 1
if not IS_GITHUB_ACTIONS:
    while attempt <= max_attempts:
        try:
            es = Elasticsearch("http://127.0.0.1:9200")  # Utiliser 127.0.0.1 au lieu de localhost
            if es.ping():
                logger.info("Connexion à Elasticsearch réussie")
                break
            else:
                logger.warning(f"Tentative {attempt}/{max_attempts} : Elasticsearch non prêt")
        except Exception as e:
            logger.warning(f"Tentative {attempt}/{max_attempts} : Erreur de connexion - {e}")
        time.sleep(5)  # Attendre 5 secondes avant de réessayer
        attempt += 1

    if es is None or not es.ping():
        logger.error("Connexion à Elasticsearch échouée après plusieurs tentatives")
        raise ConnectionError("Impossible de se connecter à Elasticsearch")
else:
    logger.info("Exécution dans GitHub Actions : Elasticsearch désactivé")

# Configurer un handler pour envoyer les logs à Elasticsearch (uniquement si es est disponible)
log_queue = queue.Queue()
if es:
    class ElasticsearchHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            try:
                es.index(index="mlflow-metrics", body={
                    "timestamp": record.created,
                    "level": record.levelname,
                    "message": log_entry,
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                })
            except Exception as e:
                logger.error(f"Erreur lors de l'envoi des logs à Elasticsearch : {e}")

    es_handler = ElasticsearchHandler()
    es_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(es_handler)
else:
    logger.info("Elasticsearch non disponible : logs ne seront pas envoyés à Elasticsearch")

logger.info("Module model_pipeline.py chargé")

# Configurer MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Churn_Prediction_Experiment")

def plot_roc_curve(y_true, y_scores, run_id):
    logger.info(f"Plotting ROC curve for run {run_id}")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_path = f"roc_curve_{run_id}.png"
    plt.savefig(roc_path)
    plt.close()
    return roc_path, roc_auc

def plot_confusion_matrix(y_true, y_pred, run_id):
    logger.info(f"Plotting confusion matrix for run {run_id}")
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    cm_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(cm_path)
    plt.close()
    return cm_path

def plot_precision_recall_curve(y_true, y_scores, run_id):
    logger.info(f"Plotting precision-recall curve for run {run_id}")
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, color='purple', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    pr_path = f"pr_curve_{run_id}.png"
    plt.savefig(pr_path)
    plt.close()
    return pr_path

def plot_prediction_distribution(y_scores, run_id):
    logger.info(f"Plotting prediction distribution for run {run_id}")
    plt.figure()
    sns.histplot(y_scores, bins=20, kde=True)
    plt.title('Distribution des probabilités prédites')
    dist_path = f"pred_distribution_{run_id}.png"
    plt.savefig(dist_path)
    plt.close()
    return dist_path

def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    mlflow.log_metric("cpu_usage_percent", cpu_usage)
    mlflow.log_metric("memory_usage_percent", memory_usage)
    logger.info(f"System Metrics - CPU: {cpu_usage}%, Memory: {memory_usage}")

def prepare_data(filepath, test_size=0.2):
    logger.info(f"Preparing data from {filepath}")
    logger.info(f"Call stack: {''.join(traceback.format_stack()[:-1])}")
    print(f"📥 Préparation des données... (Appel avec filepath={filepath})")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File {filepath} not found")
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")

    # Convertir les colonnes entières en float64
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('float64')

    # Supprimer les colonnes inutiles
    df.drop(["State", "Area code"], axis=1, inplace=True)

    # Encodage
    le = LabelEncoder()
    df["International plan"] = le.fit_transform(df["International plan"])
    df["Voice mail plan"] = le.fit_transform(df["Voice mail plan"])
    df["Churn"] = df["Churn"].map({False: 0, True: 1})

    # Vérifier les labels
    logger.info(f"Labels uniques dans Churn: {df['Churn'].unique()}")

    # Séparation features / target
    X = df.drop(["Churn"], axis=1)
    y = df["Churn"]

    # Normalisation
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Sauvegarder le scaler
    scaler_path = "models/scaler.joblib"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    # Logger le dataset
    dataset = mlflow.data.from_pandas(
        df=df,
        name="churn_dataset",
        source=f"file://{os.path.abspath(filepath)}"
    )
    mlflow.log_input(dataset, context="training")
    mlflow.end_run()

    return X_train, X_test, y_train, y_test

def balance_data(X_train, y_train):
    logger.info("Balancing data with SMOTE and ENN")
    logger.info(f"Call stack: {''.join(traceback.format_stack()[:-1])}")
    print("⚖️ Équilibrage des données...")
    smote = SMOTE(random_state=42)
    enn = EditedNearestNeighbours()
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res, y_res = enn.fit_resample(X_res, y_res)
    logger.info(f"Labels uniques après équilibrage: {np.unique(y_res)}")
    if len(np.unique(y_res)) < 2:
        logger.warning("Attention : une seule classe présente après équilibrage !")
    return X_res, y_res

def train_model(X_train, y_train):
    logger.info("Starting model training")
    logger.info(f"Call stack: {''.join(traceback.format_stack()[:-1])}")
    print("🧠 Entraînement du modèle...")
    start_time = time.time()

    # Grille d'hyperparamètres
    param_grid = [
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": 15, "min_samples_split": 10}
    ]

    best_model = None
    best_f1 = 0.0

    # Exemple d'entrée pour la signature
    input_example = X_train[:1]

    for params in param_grid:
        mlflow.end_run()
        with mlflow.start_run(run_name=f"RF_n{params['n_estimators']}_d{params['max_depth']}") as run:
            # Logger les paramètres
            mlflow.log_params(params)
            mlflow.set_tag("environment", "development")
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("developer", "Rayen")
            mlflow.set_tag("project", "4ds-10")

            # Logger les métriques système
            log_system_metrics()

            # Entraîner le modèle
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Logger le modèle avec input_example
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="ChurnPredictionModel",
                input_example=input_example
            )

            # Simuler plusieurs passes
            for epoch in range(1, 4):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                mlflow.log_metric("train_accuracy_epoch", accuracy_score(y_train, y_pred), step=epoch)

            # Mettre à jour le meilleur modèle
            y_pred = model.predict(X_train)
            f1 = f1_score(y_train, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

            # Courbe d'apprentissage
            scoring = 'f1' if len(np.unique(y_train)) > 1 else 'accuracy'
            if scoring == 'accuracy':
                logger.warning("Seule une classe présente dans y_train, passage à scoring='accuracy'")

            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1
            )
            plt.figure()
            plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
            plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
            plt.xlabel('Training examples')
            plt.ylabel(f'{scoring.capitalize()} Score')
            plt.title('Learning Curve')
            lc_path = f"learning_curve_{run.info.run_id}.png"
            plt.savefig(lc_path)
            plt.close()
            mlflow.log_artifact(lc_path)
            os.remove(lc_path)

    training_time = time.time() - start_time
    mlflow.log_metric("training_time_seconds", training_time)
    logger.info(f"Training completed in {training_time:.2f} seconds")
    mlflow.end_run()
    return best_model

def evaluate_model(model, X_test, y_test):
    logger.info("Starting model evaluation")
    logger.info(f"Call stack: {''.join(traceback.format_stack()[:-1])}")
    print("📊 Évaluation du modèle...")
    mlflow.end_run()
    with mlflow.start_run(nested=True, run_name="Evaluation"):
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]

        # Calcul des métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }

        # Logger les métriques
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Logger les métriques système
        log_system_metrics()

        # ROC et AUC
        roc_path, roc_auc = plot_roc_curve(y_test, y_scores, mlflow.active_run().info.run_id)
        mlflow.log_artifact(roc_path)
        mlflow.log_metric("roc_auc", roc_auc)

        # Matrice de confusion
        cm_path = plot_confusion_matrix(y_test, y_pred, mlflow.active_run().info.run_id)
        mlflow.log_artifact(cm_path)

        # Courbe précision-rappel
        pr_path = plot_precision_recall_curve(y_test, y_scores, mlflow.active_run().info.run_id)
        mlflow.log_artifact(pr_path)

        # Distribution des prédictions
        dist_path = plot_prediction_distribution(y_scores, mlflow.active_run().info.run_id)
        mlflow.log_artifact(dist_path)

        # Rapport de classification
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_path = f"classification_report_{mlflow.active_run().info.run_id}.json"
        with open(report_path, "w") as f:
            json.dump(report, f)
        mlflow.log_artifact(report_path)

        # Logger les métriques par classe
        for label, metrics_dict in report.items():
            if isinstance(metrics_dict, dict):
                mlflow.log_metric(f"precision_class_{label}", metrics_dict["precision"])
                mlflow.log_metric(f"recall_class_{label}", metrics_dict["recall"])
                mlflow.log_metric(f"f1_class_{label}", metrics_dict["f1-score"])

        # Feature importance
        feature_names = [
            "Account length", "International plan", "Voice mail plan", "Number vmail messages",
            "Total day minutes", "Total day calls", "Total day charge", "Total eve minutes",
            "Total eve calls", "Total eve charge", "Total night minutes", "Total night calls",
            "Total night charge", "Total intl minutes", "Total intl calls", "Total intl charge",
            "Customer service calls"
        ]
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        feature_importance_path = f"feature_importance_{mlflow.active_run().info.run_id}.csv"
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path)

        # Logger feature importance comme métriques
        for i, feature in enumerate(feature_names):
            mlflow.log_metric(f"feature_importance_{feature.replace(' ', '_')}", model.feature_importances_[i])

        # Évaluation automatique
        time.sleep(2)  # Pause pour éviter la surcharge du serveur MLflow
        eval_data = pd.DataFrame(X_test, columns=feature_names)
        eval_data["Churn"] = y_test.values
        try:
            mlflow.evaluate(
                model=f"runs:/{mlflow.active_run().info.run_id}/model",
                data=eval_data,
                targets="Churn",
                model_type="classifier",
                evaluators=["default"]
            )
        except Exception as e:
            logger.error(f"Erreur lors de mlflow.evaluate: {e}")

        # Comparaison avec le modèle en production
        try:
            prod_model = mlflow.sklearn.load_model("models:/ChurnPredictionModel/Production")
            y_pred_prod = prod_model.predict(X_test)
            mlflow.log_metric("accuracy_production", accuracy_score(y_test, y_pred_prod))
        except Exception as e:
            logger.info(f"No production model available: {e}")

        # Graphique de comparaison des métriques
        plt.figure()
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Performance Metrics')
        plt.xticks(rotation=45)
        metrics_path = f"metrics_bar_{mlflow.active_run().info.run_id}.png"
        plt.savefig(metrics_path)
        plt.close()
        mlflow.log_artifact(metrics_path)

        # Logger le fichier de traces
        mlflow.log_artifact("training.log")

        # Nettoyer les fichiers temporaires
        for path in [roc_path, cm_path, pr_path, dist_path, report_path, feature_importance_path, metrics_path]:
            os.remove(path)

        # Affichage console
        print("📊 Rapport de classification :\n")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("🧱 Matrice de confusion :\n")
        print(confusion_matrix(y_test, y_pred))

        logger.info("Evaluation completed")
        return metrics

def save_model(model, filename="models/churn_model.joblib"):
    logger.info(f"Saving model to {filename}")
    print(f"💾 Sauvegarde du modèle dans {filename}...")
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, filename)
    mlflow.log_artifact(filename)

def load_model(filename="models/churn_model.joblib"):
    logger.info(f"Loading model from {filename}")
    print(f"📤 Chargement du modèle depuis {filename}...")
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        logger.error(f"Model file {filename} not found")
        raise FileNotFoundError(f"Le fichier modèle {filename} n'existe pas")

if __name__ == "__main__":
    # Charger les données
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")

    # Équilibrer les données
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    # Entraîner le modèle
    model = train_model(X_train_balanced, y_train_balanced)

    # Évaluer le modèle
    metrics = evaluate_model(model, X_test, y_test)

    # Sauvegarder le modèle
    save_model(model)
