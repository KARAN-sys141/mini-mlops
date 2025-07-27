import os
import pandas as pd
import re
import string
import logging
import mlflow
import dagshub
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(levelname)s - %(message)s"
)

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def remove_stop_words(text, stop_words):
    return " ".join([word for word in text.split() if word not in stop_words])

def remove_numbers(text):
    return "".join([char for char in text if not char.isdigit()])

def remove_punctuation(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", "", text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def lemmatize_text(text, lemmatizer):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def clean_text_column(df, stop_words, lemmatizer):
    logging.info("Starting text normalization...")
    df['content'] = df['content'].astype(str)
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(lambda x: remove_stop_words(x, stop_words))
    df['content'] = df['content'].apply(remove_numbers)
    df['content'] = df['content'].apply(remove_punctuation)
    df['content'] = df['content'].apply(remove_urls)
    df['content'] = df['content'].apply(lambda x: lemmatize_text(x, lemmatizer))
    logging.info("Text normalization complete.")
    return df

def preprocess_and_save(raw_dir: str, processed_dir: str, stopword_lang: str = "english"):
    try:
        # Initialize DagsHub + MLflow tracking
        dagshub.init(repo_owner='KARAN-sys141', repo_name='mini-mlops', mlflow=True)
        mlflow.set_experiment("Data-Preprocessing")

        with mlflow.start_run():
            logging.info(f"Logging parameters: stopword_lang={stopword_lang}")
            mlflow.log_param("stopword_lang", stopword_lang)

            stop_words = set(stopwords.words(stopword_lang))
            lemmatizer = WordNetLemmatizer()

            os.makedirs(processed_dir, exist_ok=True)

            train_path = os.path.join(raw_dir, "train.csv")
            test_path = os.path.join(raw_dir, "test.csv")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded raw train and test data.")

            train_df = clean_text_column(train_df, stop_words, lemmatizer)
            test_df = clean_text_column(test_df, stop_words, lemmatizer)

            train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

            logging.info("Preprocessed data saved successfully.")

    except Exception as e:
        logging.exception("Error during data preprocessing:")
        raise e

if __name__ == "__main__":
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    STOPWORD_LANG = "english"

    preprocess_and_save(RAW_DIR, PROCESSED_DIR, stopword_lang=STOPWORD_LANG)
