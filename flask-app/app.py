from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle

app = Flask(__name__)

dagshub.init(repo_owner='rajeshxdatascience', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/rajeshxdatascience/mlops-mini-project.mlflow")

# load model from registry
model_name = 'my_model'
model_version = 6

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

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

    # prediction
    result = model.predict(features)

    # show
    return str(result[0])

app.run(debug=True)