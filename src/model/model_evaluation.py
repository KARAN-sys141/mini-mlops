import os
import logging
import json
import numpy as np
import joblib
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import load_npz

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(levelname)s - %(message)s"
)

def evaluate_model(features_dir: str, model_dir: str, metrics_dir: str):
    try:
        # Initialize Dagshub + MLflow tracking
        dagshub.init(repo_owner='KARAN-sys141', repo_name='mini-mlops', mlflow=True)
        mlflow.set_tracking_uri('https://dagshub.com/KARAN-sys141/mini-mlops.mlflow')
        mlflow.set_experiment("LogisticRegression-Experiment")

        with mlflow.start_run():
            logging.info("Loading test data...")
            X_test = load_npz(os.path.join(features_dir, "X_test.npz"))
            y_test = np.load(os.path.join(features_dir, "y_test.npy"), allow_pickle=True)

            model_path = os.path.join(model_dir, "model.pkl")
            model = joblib.load(model_path)
            logging.info(f"Model loaded from: {model_path}")

            logging.info("Making predictions and evaluating...")
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }

            # Save metrics locally
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Evaluation metrics saved to: {metrics_path}")

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Log metrics file as artifact
            mlflow.log_artifact(metrics_path, artifact_path="metrics")

            logging.info("✅ Metrics logged to MLflow")

            print("\n✅ Model Evaluation Completed!")
            print(f"📊 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"📊 Precision: {metrics['precision']:.4f}")
            print(f"📊 Recall:    {metrics['recall']:.4f}")
            print(f"📊 F1 Score:  {metrics['f1_score']:.4f}")

    except Exception as e:
        logging.exception("Error during model evaluation:")
        raise e

if __name__ == "__main__":
    FEATURES_DIR = "artifacts/features"
    MODEL_DIR = "artifacts/model"
    METRICS_DIR = "artifacts/metrics"

    evaluate_model(FEATURES_DIR, MODEL_DIR, METRICS_DIR)
