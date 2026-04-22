import os
from pathlib import Path

# Provide mock values
ASSETS = os.getenv("ASSETS", "BTC,ETH").split(",")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL_SECONDS", 300))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN", "")
OHLCV_BARS = int(os.getenv("OHLCV_BARS", 1000))
OHLCV_INTERVAL = os.getenv("OHLCV_INTERVAL", "5m")
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", 0.25))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", 0.05))
STARTING_BANKROLL = float(os.getenv("STARTING_BANKROLL", 1000.0))
env = os.getenv
