import logging
import os
import datetime
from datetime import datetime

LOG_FILE= f"{datetime.now().strftime('%d_%M_%Y_%H_%M_%S')}.log"
file_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(file_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(file_path,LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

if __name__ == '__main__':
    logging.info("Logging has started")