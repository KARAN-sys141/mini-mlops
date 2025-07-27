import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import dagshub

dagshub.init(repo_owner='KARAN-sys141', repo_name='mini_mlops', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/KARAN-sys141/mini-mlops.mlflow')

df = pd.read_csv(
    'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
).drop(columns=['tweet_id'])

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
    return "".join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    return re.sub('[%s]' % re.escape(string.punctuation), '', text)

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

df = normalize_text(df)

df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('LoR HT')

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

with mlflow.start_run():
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    params_list = grid_search.cv_results_['params']
    mean_scores = grid_search.cv_results_['mean_test_score']
    std_scores = grid_search.cv_results_['std_test_score']

    for params, mean_score, std_score in zip(params_list, mean_scores, std_scores):
        with mlflow.start_run(run_name=f'LR with params: {params}', nested=True):
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mlflow.log_params(params)
            mlflow.log_metric('mean_cv_score', mean_score)
            mlflow.log_metric('std_cv_score', std_score)
            mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
            mlflow.log_metric('precision', precision_score(y_test, y_pred))
            mlflow.log_metric('recall', recall_score(y_test, y_pred))
            mlflow.log_metric('f1_score', f1_score(y_test, y_pred))

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric('best_f1_score', best_score)
    
    algo_name = "logistic_regression"
    vec_name = "count_vectorizer"


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

    try:
        mlflow.log_artifact(__file__)
    except:
        pass  

    print(f"\n Best Params: {best_params}")
    print(f" Best F1 Score: {best_score:.4f}")
