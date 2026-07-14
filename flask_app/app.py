from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import os
import pickle
import pandas as pd

# Get DagsHub token from environment variable
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# DagsHub credentials for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = "rajeshxdatascience"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# MLflow tracking URI
repo_owner = "rajeshxdatascience"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)

app = Flask(__name__)

# load model from model registry
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)

    # show
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")