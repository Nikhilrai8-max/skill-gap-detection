import pandas as pd
from .config import DATA_PATH

def load_skill_data(path: str = None):
    p = path if path is not None else DATA_PATH
    df = pd.read_csv(p)
    return df
