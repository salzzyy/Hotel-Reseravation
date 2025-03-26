import os
import yaml
import pandas as pd  # Keep only one import
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def read_yaml(file_path):
    """
    Reads a YAML file and returns its content as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Parsed YAML data as a dictionary.
    :raises CustomException: If the file is not found or there's an issue reading it.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"Successfully read YAML file: {file_path}")
            return config
        
    except Exception as e:
        logger.error(f"Error reading YAML file: {file_path} - {str(e)}")
        raise CustomException("Failed to read YAML file", e)


def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error in loading the data: {e}")  # Fixed error formatting
        raise CustomException("Failed to load data", e)
