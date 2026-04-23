import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_bankroll():
    return 1000.0

def update_bankroll(amount, reason=None):
    pass

def compute_accuracy(asset=None):
    return {"accuracy": 0.0, "total": 0}

def load_predictions(asset, limit=50):
    return []

def mark_resolved(pred_id, actual):
    pass

def save_prediction(asset, pred):
    pass
