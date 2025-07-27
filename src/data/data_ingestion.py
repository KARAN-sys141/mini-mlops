import os
import pandas as pd
import logging
import mlflow
import dagshub
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(levelname)s - %(message)s"
)

def fetch_and_split_data(url: str, raw_dir: str, test_size: float = 0.2, random_state: int = 42):
    try:
        # Initialize DagsHub + MLflow tracking
        dagshub.init(repo_owner='KARAN-sys141', repo_name='mini-mlops', mlflow=True)
        mlflow.set_experiment("Data-Ingestion")

        with mlflow.start_run():
            logging.info(f"Logging parameters: test_size={test_size}, random_state={random_state}")
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("dataset_url", url)

            logging.info(f"Fetching dataset from: {url}")
            df = pd.read_csv(url)

            if 'tweet_id' in df.columns:
                df.drop(columns=['tweet_id'], inplace=True)
                logging.info("Dropped 'tweet_id' column")

            logging.info("Splitting data into train and test sets...")
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

            os.makedirs(raw_dir, exist_ok=True)
            train_path = os.path.join(raw_dir, "train.csv")
            test_path = os.path.join(raw_dir, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info(f"Train data saved at: {train_path}")
            logging.info(f"Test data saved at: {test_path}")

    except Exception as e:
        logging.exception("Error during data ingestion:")
        raise e

if __name__ == "__main__":
    DATA_URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
    RAW_DIR = "data/raw"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    fetch_and_split_data(DATA_URL, RAW_DIR, test_size=TEST_SIZE, random_state=RANDOM_STATE)
