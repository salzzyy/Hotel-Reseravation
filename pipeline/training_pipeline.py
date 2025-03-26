from src.data_ingestion import DataIngestion
from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
import logging  # Ensure logger is properly set up

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        config = read_yaml(CONFIG_PATH)

        ## DATA INGESTION
        data_ingestion = DataIngestion(config)
        data_ingestion.run()

        ## DATA PROCESSING
        processor = DataProcessor(
            TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH
        )
        processor.process()

        ## MODEL TRAINING
        trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
        trainer.run()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
