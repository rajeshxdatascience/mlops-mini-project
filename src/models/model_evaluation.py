import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import dagshub
import mlflow

dagshub.init(repo_owner='rajeshxdatascience', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/rajeshxdatascience/mlops-mini-project.mlflow")

# logging configure

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_data(test_path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        test_data = pd.read_csv(test_path)

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.info("Test data loaded successfully.")

        return X_test, y_test

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def load_model(model_path: str):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info("Model loaded successfully.")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def evaluate_model(model,X_test: np.ndarray,y_test: np.ndarray) -> dict:

    try:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_prob),
        }

        logger.info("Model evaluated successfully.")

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def save_metrics(metrics: dict, output_path: str) -> None:
    try:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Metrics saved successfully.")

    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise
    
def main() -> None:
    mlflow.set_experiment("dvc-pipeline")
    mlflow.start_run()
    try:
        X_test, y_test = load_data("./data/features/test_bow.csv")

        model = load_model("models/model.pkl")

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        save_metrics(metrics, "metrics.json")

        logger.info("Model evaluation pipeline completed successfully.")

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model parameters to mlflow
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param_name , param_value in params.items():
                mlflow.log_param(param_name, param_value)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()