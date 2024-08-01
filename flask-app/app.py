from flask import Flask, render_template, request
import mlflow
from mlflow.tracking import MlflowClient
from preprocessing_utility import normalize_text
import dagshub
import pickle
import os

mlflow.set_tracking_uri('https://dagshub.com/aniketnandanwar09/mlops-mini-project.mlflow')
dagshub.init(repo_owner='aniketnandanwar09', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)

# Load model from model registry
def get_latest_model_version(model_name):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
    return latest_versions[0].version if latest_versions else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

if model_version is None:
    raise ValueError("No model version found for model: {}".format(model_name))

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer_path = os.path.join('models', 'vectorizer.pkl')
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer not found at path: {vectorizer_path}")

vectorizer = pickle.load(open(vectorizer_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Clean
    text = normalize_text(text)

    # BOW
    features = vectorizer.transform([text])

    # Prediction
    result = model.predict(features)

    # Show
    return render_template('index.html', result=result[0])

if __name__ == '__main__':
    app.run(debug=True)
