import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import string
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import dagshub
import mlflow


dagshub.init(repo_owner='rajeshxdatascience', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/rajeshxdatascience/mlops-mini-project.mlflow")

df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv").drop(columns=['tweet_id'])

# Transform the data

import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")


def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text: str) -> str:
    text = " ".join(
        [word for word in text.split() if not word.isdigit()]
    )
    return text


def lower_case(text: str) -> str:
    return text.lower()


def removing_punctuation(text: str) -> str:
    text = re.sub(
        '[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def removing_urls(text: str) -> str:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df["content"] = df["content"].astype(str)

    df["content"] = df["content"].apply(lower_case)
    df["content"] = df["content"].apply(removing_urls)
    df["content"] = df["content"].apply(removing_numbers)
    df["content"] = df["content"].apply(removing_punctuation)
    df["content"] = df["content"].apply(remove_stop_words)
    df["content"] = df["content"].apply(lemmatization)

    return df

df = normalize_text(df)

x = df['sentiment'].isin(['happiness','sadness'])
df = df[x]

df['sentiment'] = df['sentiment'].replace({'sadness':0, 'happiness':1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Set the experiment name
mlflow.set_experiment("LoR Hyperparameter Tuning")

param_grid = {
    'C':[0.1, 1, 10],
    'penalty':['l1', 'l2'],
    'solver': ['liblinear']
}

with mlflow.start_run():

    # Perform grid search
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=1)
    grid_search.fit(X_train, y_train)

    for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)


            # Model evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test, y_pred,zero_division=0)
            recall = recall_score(y_test,y_pred,zero_division=0)
            f1 = f1_score(y_test,y_pred,zero_division=0)

            # Log parameters

            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1",f1)

            print(f"Mean CV Score: {mean_score}, Std CV Score: {std_score}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")

    # Log the best run 
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric('best_f1_score', best_score)

    print(best_score)
    print(best_params)