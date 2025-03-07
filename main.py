from logging import getLogger
import mlflow
import mlflow.sklearn
import logging
from elasticsearch import Elasticsearch
from model_pipeline import (
    test_prepare_data,
    test_train_model,
    test_evaluate_model,
    test_save_model,
    test_load_model,
)

# Configuration Elasticsearch
ES_HOST = "http://localhost:9200"
INDEX_NAME = "mlflow-logs"


def setup_elasticsearch():
    """Vérifie si l'index Elasticsearch existe et le crée si nécessaire."""
    es = Elasticsearch([ES_HOST])

    # Vérifier si l'index existe
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(
            index=INDEX_NAME, ignore=400
        )  # Ignore l'erreur si l'index existe déjà
        print(f" Index '{INDEX_NAME}' créé dans Elasticsearch.")
    else:
        print(f" Index '{INDEX_NAME}' existe déjà.")

    return es


def log_to_elasticsearch(es, log_data):
    """Enregistre les logs de MLflow dans Elasticsearch."""
    try:
        es.index(index=INDEX_NAME, document=log_data)
        print(f" Log inséré dans Elasticsearch: {log_data}")
    except Exception as e:
        print(f"❌ Erreur lors de l'insertion des logs dans Elasticsearch: {e}")


def main():
    """Exécute le pipeline complet avec MLflow et enregistre les logs dans Elasticsearch."""

    # Initialisation d'Elasticsearch
    es = setup_elasticsearch()

    # Configuration de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow UI tourne sur ce port
    mlflow.set_experiment("Churn_Prediction")  # Nom de l'expérience

    # Début d'une exécution MLflow
    with mlflow.start_run():
        X_train_scaled, X_test_scaled, y_train, y_test = test_prepare_data()

        # Entraînement du modèle
        model = test_train_model(X_train_scaled, y_train)

        # Enregistrement des hyperparamètres
        params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.3}
        for param, value in params.items():
            mlflow.log_param(param, value)

        # Évaluation du modèle
        results = test_evaluate_model(model, X_test_scaled, y_test)
        if results:
            mlflow.log_metric("accuracy", results["accuracy"])

            # Enregistrement des résultats dans Elasticsearch
            log_data = {
                "experiment": "Churn_Prediction",
                "accuracy": results["accuracy"],
                "params": params,
            }
            log_to_elasticsearch(es, log_data)

        # Sauvegarde du modèle
        test_save_model(model, "xgboost_model.pkl")
        loaded_model = test_load_model("xgboost_model.pkl")
        mlflow.sklearn.log_model(model, "xgboost_model")

    print(" Enregistrement terminé avec MLflow et Elasticsearch !")


if __name__ == "__main__":
    main()
