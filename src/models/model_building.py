import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from sklearn.linear_model import LogisticRegression

# logging configure

logger = logging.getLogger("model_building")
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


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        logger.info("Parameters loaded successfully.")

        return params["model_building"]

    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_path}")
        raise

    except KeyError:
        logger.error("'model_building' section not found in params.yaml")
        raise


def load_data(train_path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        train_data = pd.read_csv(train_path)

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        logger.info("Training data loaded successfully.")

        return X_train, y_train

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def train_model(X_train: np.ndarray,y_train: np.ndarray,params: dict) -> LogisticRegression:

    try:
        lr = LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver=params['solver']
        )

        lr.fit(X_train, y_train)

        logger.info("Model trained successfully.")

        return lr
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

# save the model

def save_model(model: LogisticRegression, model_path: str) -> None:
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info("Model saved successfully.")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")

        X_train, y_train = load_data("./data/features/train_bow.csv")

        model = train_model(
            X_train,
            y_train,
            params
        )

        save_model(model, "models/model.pkl")

        logger.info("Model building pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()