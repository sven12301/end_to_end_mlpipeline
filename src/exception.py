import sys
from src.logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generates a detailed error message, including the file name and line number
    where the error occurred, as well as the error message.
    
    Parameters:
    - error (Exception): The exception that was raised.
    - error_detail (sys): System module to extract traceback information.
    
    Returns:
    - str: A formatted error message with detailed information.
    """
    _, _, traceback_info = error_detail.exc_info()  # returns (type, value, traceback)
    file_name = traceback_info.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in Python script: '{file_name}', "
        f"line: {traceback_info.tb_lineno}, "
        f"message: {error}"
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception class to provide a detailed error message, including the
    file name, line number, and the specific error that occurred.
    """
    def __init__(self, error_message: str, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error message directly
    
    def __str__(self) -> str:
        return self.error_message
