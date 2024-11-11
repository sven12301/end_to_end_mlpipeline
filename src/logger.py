import logging
import os
from datetime import datetime

def setup_log_directory() -> str:
    """
    Creates a new directory for logs based on the current timestamp and returns the path.
    
    Returns:
    - str: The path to the timestamped log directory.
    """
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_directory = os.path.join(os.getcwd(), "logs", timestamp)
    os.makedirs(log_directory, exist_ok=True)
    return log_directory

def get_log_file_path() -> str:
    """
    Generates a log file path inside the timestamped log directory.
    
    Returns:
    - str: The full path for the log file inside the timestamped directory.
    """
    log_directory = setup_log_directory()
    log_file = "app.log"  # Consistent log file name within each timestamped directory
    return os.path.join(log_directory, log_file)

# Define the log file path
LOG_FILE_PATH = get_log_file_path()

# Configure logging to write to the file and console
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Console handler for real-time logging output
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)
