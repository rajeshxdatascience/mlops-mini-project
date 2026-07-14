import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import dagshub
import mlflow
import os


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

def save_model_info(
    run_id: str,
    model_uri: str,
    file_path: str
) -> None:
    """Save the MLflow run ID and actual model URI."""

    try:
        model_info = {
            "run_id": run_id,
            "model_uri": model_uri
        }

        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)

        logger.info(
            "Model info saved successfully to %s",
            file_path
        )

    except Exception as e:
        logger.error(
            "Error while saving model info: %s",
            e
        )
        raise
def main() -> None:

    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:

        try:
            # Load test data
            X_test, y_test = load_data(
                "./data/features/test_bow.csv"
            )

            # Load trained model
            model = load_model(
                "models/model.pkl"
            )

            # Evaluate model
            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            # Save metrics locally
            save_metrics(
                metrics,
                "metrics.json"
            )

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(
                    metric_name,
                    metric_value
                )

            # Log model parameters
            if hasattr(model, "get_params"):
                params = model.get_params()

                for param_name, param_value in params.items():
                    mlflow.log_param(
                        param_name,
                        param_value
                    )

            # Log model and capture actual model URI
            logged_model = mlflow.sklearn.log_model(sk_model=model,artifact_path="model")

            logger.info(
                "Model logged to MLflow with URI: %s",
                logged_model.model_uri
            )

            # Save model information
            save_model_info(
                run_id=run.info.run_id,
                model_uri=logged_model.model_uri,
                file_path="reports/experiment_info.json"
            )

            # Log metrics artifact
            mlflow.log_artifact(
                "metrics.json"
            )

            logger.info(
                "Model evaluation pipeline completed successfully."
            )

        except Exception as e:
            logger.error(
                "Pipeline failed: %s",
                e
            )
            raise


if __name__ == "__main__":
    main()