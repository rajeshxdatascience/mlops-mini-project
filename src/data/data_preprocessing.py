import numpy as np
import pandas as pd
import os 
import logging

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Logging Configure

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
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


# fetch the data from data/raw

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logger.info("Train and test data loaded successfully.")

        return train_data, test_data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# transform the data

def download_nltk_resources() -> None:
    try:
        nltk.download("wordnet")
        nltk.download("stopwords")

        logger.info("NLTK resources downloaded successfully.")

    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
  lemmatization = WordNetLemmatizer()

  text = text.split()

  text = [lemmatization.lemmatize(y) for y in text]

  return " ".join(text)

def remove_stop_words(text: str) -> str:
  stop_words = set(stopwords.words("english"))
  Text = [i for i in str(text).split() if i not in stop_words]

  return " ".join(Text)

def removing_numbers(text: str) -> str:
    text = " ".join([word for word in text.split() if not word.isdigit()])
    return text

def lower_case(text: str) -> str:

  text = text.split()

  text = [y.lower() for y in text]

  return " ".join(text)

def removing_punctuation(text: str) -> str:

  text = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
  text = text.replace(':', "", )

  text = re.sub(r'\s+',' ', text)
  text = " ".join(text.split())
  return text.strip()

def removing_urls(text: str) -> str:
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  return url_pattern.sub(r'', text)

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(removing_urls)
        df["content"] = df["content"].apply(removing_numbers)
        df["content"] = df["content"].apply(removing_punctuation)
        df["content"] = df["content"].apply(remove_stop_words)
        df["content"] = df["content"].apply(lemmatization)

        logger.info("Text normalization completed successfully.")

        return df

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise
    
def save_data(data_path: str,train_data: pd.DataFrame,test_data: pd.DataFrame,) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(data_path, "train_processed_data.csv"),
            index=False,
        )

        test_data.to_csv(
            os.path.join(data_path, "test_processed_data.csv"),
            index=False,
        )

        logger.info("Processed data saved successfully.")

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise
    
def main() -> None:
    try:
        train_data, test_data = load_data(
            "./data/raw/train.csv",
            "./data/raw/test.csv",
        )

        download_nltk_resources()

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        save_data(
            os.path.join("data", "processed"),
            train_processed_data,
            test_processed_data,
        )

        logger.info("Data preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()