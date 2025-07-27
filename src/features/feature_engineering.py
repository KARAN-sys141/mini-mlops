import os
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow
import dagshub
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(levelname)s - %(message)s"
)

def vectorize_and_save(processed_dir: str, features_dir: str, max_features: int = None):
    try:
        # Initialize DAGsHub + MLflow tracking
        dagshub.init(repo_owner='KARAN-sys141', repo_name='mini-mlops', mlflow=True)
        mlflow.set_experiment("Feature-Engineering")

        with mlflow.start_run():
            logging.info(f"Logging parameters: max_features={max_features}")
            mlflow.log_param("max_features", max_features)

            train_path = os.path.join(processed_dir, "train.csv")
            test_path = os.path.join(processed_dir, "test.csv")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Preprocessed data loaded.")

            # Drop rows with missing content
            train_df = train_df.dropna(subset=["content"])
            test_df = test_df.dropna(subset=["content"])

            vectorizer_params = {}
            if max_features:
                vectorizer_params["max_features"] = max_features

            vectorizer = CountVectorizer(**vectorizer_params)
            logging.info("Fitting CountVectorizer on training data...")
            X_train = vectorizer.fit_transform(train_df["content"])
            X_test = vectorizer.transform(test_df["content"])

            y_train = train_df["sentiment"].values
            y_test = test_df["sentiment"].values

            os.makedirs(features_dir, exist_ok=True)

            save_npz(os.path.join(features_dir, "X_train.npz"), X_train)
            save_npz(os.path.join(features_dir, "X_test.npz"), X_test)
            np.save(os.path.join(features_dir, "y_train.npy"), y_train)
            np.save(os.path.join(features_dir, "y_test.npy"), y_test)

            vectorizer_path = os.path.join(features_dir, "vectorizer.pkl")
            joblib.dump(vectorizer, vectorizer_path)

            # Log artifacts manually to MLflow (safe for Dagshub)
            mlflow.log_artifact(vectorizer_path, artifact_path="features")

            logging.info("Feature vectors and labels saved successfully and logged to MLflow.")

    except Exception as e:
        logging.exception("Error during feature engineering:")
        raise e

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    FEATURES_DIR = "artifacts/features"
    MAX_FEATURES = 10000  # Example param, adjust or make dynamic as needed

    vectorize_and_save(PROCESSED_DIR, FEATURES_DIR, max_features=MAX_FEATURES)
