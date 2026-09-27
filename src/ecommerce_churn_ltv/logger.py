import logging 
import os
from datetime import datetime

# Create a unique timestamped log file name

LOG_FILE = f"{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log"

# define the path for the log directory in the workspace

logs_path = os.path.join(os.getcwd(), "logs")

# Create the "logs" folder if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Construct the full absolute file path for the log file

LOG_FILE_PATH =  os.path.join(logs_path, LOG_FILE)

# Configure the logging module settings

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',level=logging.INFO
)

# Test the file

if __name__ == "__main__":
    logging.info("Logging module setup complete and working successfully!!")
