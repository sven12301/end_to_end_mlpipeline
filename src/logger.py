import logging
import os
from datetime import datetime

def setup_log_directory() -> str:
    """
    Creates a directory for logs if it doesn't exist and returns the path.
    
    Returns:
    - str: The path to the log directory.
    """
    log_directory = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_directory, exist_ok=True)
    return log_directory

def get_log_file_path() -> str:
    """
    Generates a log file path with a timestamp to avoid overwriting logs.
    
    Returns:
    - str: The full path for the log file.
    """
    log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    return os.path.join(setup_log_directory(), log_file)

# Define the log file path
LOG_FILE_PATH = get_log_file_path()

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)
