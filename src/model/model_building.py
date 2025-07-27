import os
import logging
import joblib
import numpy as np
import mlflow
import dagshub
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(levelname)s - %(message)s"
)

def train_and_save_model(features_dir: str, model_dir: str):
    try:
        # Initialize DAGsHub tracking
        dagshub.init(repo_owner='KARAN-sys141', repo_name='mini-mlops', mlflow=True)
        mlflow.set_tracking_uri('https://dagshub.com/KARAN-sys141/mini-mlops.mlflow')
        mlflow.set_experiment("LogisticRegression-Experiment")

        logging.info("Loading training data...")
        X_train = load_npz(os.path.join(features_dir, "X_train.npz"))
        y_train = np.load(os.path.join(features_dir, "y_train.npy"), allow_pickle=True)

        clf_params = {
            "C": 1,
            "penalty": "l2",
            "solver": "liblinear",
            "random_state": 42
        }

        with mlflow.start_run():
            logging.info(f"Training LogisticRegression with params: {clf_params}")
            model = LogisticRegression(**clf_params)
            model.fit(X_train, y_train)

            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Model saved to: {model_path}")

            # Log params
            mlflow.log_params(clf_params)

            # Log model artifact manually (safe for Dagshub)
            mlflow.log_artifact(model_path, artifact_path="model")

            logging.info("✅ Parameters and model artifact logged to MLflow.")

    except Exception as e:
        logging.exception("Error during model training:")
        raise e

if __name__ == "__main__":
    FEATURES_DIR = "artifacts/features"
    MODEL_DIR = "artifacts/model"

    train_and_save_model(FEATURES_DIR, MODEL_DIR)
