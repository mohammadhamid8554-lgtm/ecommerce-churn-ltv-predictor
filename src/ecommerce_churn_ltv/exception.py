import sys
from ecommerce_churn_ltv.logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Extracts detailed information from an exception including file name,

    line number, and the error message.
    """
    # exc_tb (traceback) contains the line number and file path where the error occurred
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the exception was raised
    assert exc_tb is not None
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format the error message details cleanly
    error_message = (
        f"Error occurred in Python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """Custom exception class that inherits from Python's base Exception class.

    Formats and logs detailed error traces automatically.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        # Pass the string representation of error to the superclass initializer
        super().__init__(str(error_message))

        # Capture the formatted detailed error message
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        # Returns the detailed error string when str(e) or raise is called
        return self.error_message

if __name__ == "__main__":
    try:
        # Intentional divide-by-zero error to test the exception handler
        a = 1 / 0
    except Exception as e:
        logging.info("Testing CustomException handling...")
        raise CustomException(e, sys)