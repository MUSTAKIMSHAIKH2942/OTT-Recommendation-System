import pandas as pd
from src.config import DATA_PATH

def load_data(file_name):
    """Load data from the specified file."""
    return pd.read_csv(DATA_PATH + file_name)
