import argparse
import os
import logging
import mlflow
from model_pipeline import (
    prepare_data,
    balance_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# Configurer le logging
logging.basicConfig(level=logging.INFO, filename="training.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    mlflow.end_run()  # Terminer tout run actif
    logger.info("Début de l'exécution de main()")
    print("DÉBUT EXÉCUTION MAIN")
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour la détection du churn télécom"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Chemin vers le fichier CSV (ex. churn-bigml-80.csv)"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Préparer les données uniquement"
    )
    parser.add_argument(
        "--train", action="store_true", help="Lancer l'entraînement du modèle"
    )
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument(
        "--save", type=str, help="Nom du fichier où sauvegarder le modèle"
    )
    parser.add_argument("--model", type=str, help="Nom du fichier du modèle à charger")
    parser.add_argument(
        "--load", action="store_true", help="Charger un modèle existant"
    )

    args = parser.parse_args()
    print(f"Arguments reçus : {args}")
    logger.info(f"Arguments reçus : {args}")

    # Construire le chemin complet vers le fichier de données
    data_path = os.path.join("data", args.data)
    if not os.path.exists(data_path):
        print(f"❌ Le fichier {data_path} n'existe pas")
        logger.error(f"Le fichier {data_path} n'existe pas")
        return

    try:
        if args.prepare:
            print("📥 Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(data_path)
            print("✅ Données prêtes")
            logger.info("Données prêtes")
            return

        if args.train:
            print("📥 Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(data_path)
            print("⚖️ Rééquilibrage des données...")
            X_res, y_res = balance_data(X_train, y_train)
            print("🧠 Entraînement du modèle...")
            model = train_model(X_res, y_res)

            if args.evaluate:
                print("📈 Évaluation du modèle...")
                evaluate_model(model, X_test, y_test)

            if args.save:
                save_model(model, args.save)
                print(f"💾 Modèle sauvegardé dans {args.save}")

        elif args.load and args.model:
            if not os.path.exists(args.model):
                print(f"❌ Le fichier modèle {args.model} n'existe pas")
                logger.error(f"Le fichier modèle {args.model} n'existe pas")
                return
            print(f"📦 Chargement du modèle depuis {args.model}...")
            model = load_model(args.model)
            print("📥 Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(data_path)
            print("📈 Évaluation du modèle...")
            evaluate_model(model, X_test, y_test)

        else:
            parser.print_help()
            return

    except Exception as e:
        print(f"❌ Erreur : {e}")
        logger.error(f"Erreur : {e}")
        return
    finally:
        print("FIN EXÉCUTION MAIN")
        logger.info("Fin de l'exécution de main()")

if __name__ == "__main__":
    main()
