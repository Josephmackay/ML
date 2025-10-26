import pandas as pd
import os
from src.logs.logger_config import get_logger
logger = get_logger(__name__)

def load_data(file_path=None):
    # Default path if none is provided
    if file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up to ML/
        file_path = os.path.join(base_dir, "data", "Loan_default.csv")
    
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    return df
