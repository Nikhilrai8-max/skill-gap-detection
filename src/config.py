import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "skill_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
PCA_MODEL_PATH = os.path.join(MODEL_DIR, "pca_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
