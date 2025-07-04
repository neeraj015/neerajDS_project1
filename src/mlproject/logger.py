'''import logging
import os
from datetime import datetime

# Get the directory where logger.py resides (i.e., src/mlproject)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create logs folder inside mlproject
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create a log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

print(f"[LOGGER] Logging initialized at {LOG_FILE_PATH}")
'''










import logging, os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
print("Creating logs dir at:", LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)
print("Log file path:", LOG_FILE_PATH)

logging.basicConfig(filename=LOG_FILE_PATH, format="%(message)s", level=logging.INFO)
logging.info("Hello world!")













''' import logging
import os
from datetime import datetime

# Create log directory only
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Construct the full log file path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Setup logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)  '''














'''import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
'''