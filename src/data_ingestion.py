import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        try:
            self.config = config["data_ingestion"]
            self.bucket_name = self.config["bucket_name"]
            self.file_name = self.config["bucket_file_name"]
            self.train_test_ratio = self.config.get(
                "train_ratio", 0.8
            )  # Default to 0.8 if key is missing

            # Ensure the directory exists
            os.makedirs(RAW_DIR, exist_ok=True)

            logger.info(
                f"Data Ingestion started with bucket: {self.bucket_name}, file: {self.file_name}"
            )

        except KeyError as e:
            logger.error(f"Missing key in configuration: {str(e)}")
            raise CustomException("Invalid config file", e)

    def download_csv_from_gcp(self):
        """
        Downloads the CSV file from Google Cloud Storage.
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error while downloading the CSV file: {str(e)}")
            raise CustomException("Failed to download CSV file", e)

    def split_data(self):
        """
        Splits the downloaded CSV data into training and testing sets.
        """
        try:
            logger.info("Starting the data splitting process")

            if not os.path.exists(RAW_FILE_PATH):
                raise FileNotFoundError(f"Raw data file not found: {RAW_FILE_PATH}")

            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                data, test_size=1 - self.train_test_ratio, random_state=32
            )

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error while splitting data: {str(e)}")
            raise CustomException("Failed to split data into train and test sets", e)

    def run(self):
        """
        Runs the full data ingestion pipeline.
        """
        try:
            logger.info("Starting data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")

        finally:
            logger.info("Data ingestion process finished")


if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_PATH)  # Load config from YAML
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
