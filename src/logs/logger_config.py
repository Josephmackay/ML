import logging
import os

# Create log folder if not exists
LOG_DIR = os.path.join(os.path.dirname(__file__))
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# Function to get logger for each module
def get_logger(name):
    return logging.getLogger(name)
