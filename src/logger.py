import logging
import os
from datetime import datetime

# Create a directory for logs
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure the directory exists

# Define log file name with today's date
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Set log level to INFO
)

def get_logger(name):
    """
    Create and return a logger with the given name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set logging level

    return logger  # Return the configured logger