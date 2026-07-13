import numpy as np
import pandas as pd
import yaml
import logging
import os 
import pickle

from sklearn.feature_extraction.text import CountVectorizer

# logging configure

logger = logging.getLogger("feature_engineering")
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


def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        logger.info("Parameters loaded successfully.")

        return params["feature_engineering"]["max_features"]

    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_path}")
        raise

    except KeyError:
        logger.error("'feature_engineering.max_features' not found in params.yaml")
        raise


def load_data(train_path: str,test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_data.fillna("", inplace=True)
        test_data.fillna("", inplace=True)

        logger.info("Processed data loaded successfully.")

        return train_data, test_data

    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

# apply BoW 

def apply_bow(train_data: pd.DataFrame,test_data: pd.DataFrame,max_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:

    try:
        X_train = train_data["content"].values
        y_train = train_data["sentiment"].values

        X_test = test_data["content"].values
        y_test = test_data["sentiment"].values

        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test

        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

        logger.info("Bag of Words feature engineering completed successfully.")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise


def save_data(
    data_path: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    try:
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(
            os.path.join(data_path, "train_bow.csv"),
            index=False
        )

        test_df.to_csv(
            os.path.join(data_path, "test_bow.csv"),
            index=False
        )

        logger.info("Feature data saved successfully.")

    except Exception as e:
        logger.error(f"Error saving feature data: {e}")
        raise

def main() -> None:

    try:
        max_features = load_params("params.yaml")

        train_data, test_data = load_data(
            "./data/processed/train_processed_data.csv",
            "./data/processed/test_processed_data.csv"
        )

        train_df, test_df = apply_bow(
            train_data,
            test_data,
            max_features
        )

        save_data(
            os.path.join("data", "features"),
            train_df,
            test_df
        )

        logger.info("Feature engineering pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()