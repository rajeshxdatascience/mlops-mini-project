import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
import logging

# logging configure

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
            logger.debug("test size retrieved")
        return params["data_ingestion"]["test_size"]
    
    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_path}")
        raise
    
    except KeyError:
        logger.error("'data_ingestion.test_size' not found in params.yaml")
        raise

def read_data(url: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(url)
        return df
    
    except Exception as e:
        logger.error(f"Error reading data from {url}: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:

    try:
        df.drop(columns=['tweet_id'],inplace=True)

        final_df = df[df['sentiment'].isin(['happiness','sadness'])].copy()

        final_df["sentiment"] = final_df["sentiment"].replace({
        "happiness": 1,
        "sadness": 0
        })

        return final_df
    
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:

    try:
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)

    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():

    try:
        test_size = load_params('params.yaml')
        df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")

        final_df = process_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        data_path = os.path.join("data","raw")

        save_data(data_path, train_data, test_data)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":

    main()