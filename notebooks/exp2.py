import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os

import dagshub
dagshub.init(repo_owner='KARAN-sys141',repo_name='mini_mlops',mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/KARAN-sys141/mini-mlops.mlflow')

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])
df.head()

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = "".join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise


df = normalize_text(df)

X = df['sentiment'].isin(['happiness','sadness'])
df = df[X]

df['sentiment'] = df['sentiment'].replace({'sadness':0,'happiness':1})

mlflow.set_experiment('Bow vs Tfidf')

vectorizers = {
    'BoW':CountVectorizer(),
    'TF-IDF':TfidfVectorizer()
}


algorithms = {
    'LogisticRegression':LogisticRegression(),
    'MultinomialNB':MultinomialNB(),
    'XGboost':XGBClassifier(),
    'RandomForest':RandomForestClassifier(),
    'GradientBoosting':GradientBoostingClassifier()
}


with mlflow.start_run(run_name='All Experimants') as parent_run:
    for algo_name,algorithm in algorithms.items():
        for vec_name,vectorizer in vectorizers.items():
            with mlflow.start_run(run_name=f'{algo_name} with {vec_name}',nested=True) as child_run:
                X = vectorizer.fit_transform(df['content'])
                y = df['sentiment']
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                
                mlflow.log_param('vectorizer',vec_name)
                mlflow.log_param('algorithm',algo_name)
                mlflow.log_param('test_size',0.2)
                
                model = algorithm
                model.fit(X_train,y_train)
                
                if algo_name == 'LogisticRegression':
                    mlflow.log_param('C',model.C)
                elif algo_name == 'MultinomialNB':
                    mlflow.log_param('alpha',model.alpha)
                elif algo_name == 'XGboost':
                    mlflow.log_param('n_estimators',model.n_estimators)
                    mlflow.log_param('learning_rate',model.learning_rate)
                elif algo_name == 'RandomForest':
                    mlflow.log_param('n_estimators',model.n_estimators)
                    mlflow.log_param('max_depth',model.max_depth)
                elif algo_name == 'GradientBoosting':
                    mlflow.log_param('n_estimators',model.n_estimators)
                    mlflow.log_param('learning_rate',model.learning_rate)
                    mlflow.log_param('max_depth',model.max_depth)
                    
                    
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test,y_pred)
                precision = precision_score(y_test,y_pred)
                recall = recall_score(y_test,y_pred)
                f1 = f1_score(y_test,y_pred)


                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_metric('precision', precision)
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('f1', f1)     
                
                from mlflow.sklearn import save_model
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = f"model_{algo_name}_{vec_name}_{timestamp}"

                save_model(model, model_dir)

                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, model_dir)
                        mlflow.log_artifact(file_path, artifact_path=os.path.join("model", os.path.dirname(relative_path)))

                
                mlflow.log_artifact(__file__)  
                
                print(f'Algorithm: {algo_name}, Feature Engineering: {vec_name}')
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")             


                    
                