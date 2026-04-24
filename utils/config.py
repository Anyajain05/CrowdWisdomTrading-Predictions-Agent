import os
from pathlib import Path

from utils.runtime import ensure_user_site

ensure_user_site()

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(path: str | None = None) -> None:
        env_path = Path(path or ".env")
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env = os.getenv


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


ASSETS = _env_list("ASSETS", "BTC,ETH")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL_SECONDS", 300))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN", "")
OHLCV_BARS = int(os.getenv("OHLCV_BARS", 1000))
OHLCV_INTERVAL = os.getenv("OHLCV_INTERVAL", "5m")

KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", 0.25))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", 0.05))
STARTING_BANKROLL = float(os.getenv("STARTING_BANKROLL", 1000.0))

KRONOS_REPO_PATH = os.getenv(
    "KRONOS_REPO_PATH",
    str(PROJECT_ROOT / "third_party" / "Kronos-master"),
)
KRONOS_MODEL_NAME = os.getenv("KRONOS_MODEL_NAME", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER_NAME = os.getenv("KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_DEVICE = os.getenv("KRONOS_DEVICE", "cpu")
KRONOS_MAX_CONTEXT = int(os.getenv("KRONOS_MAX_CONTEXT", 512))
KRONOS_LOOKBACK = int(os.getenv("KRONOS_LOOKBACK", 400))
KRONOS_PRED_LEN = int(os.getenv("KRONOS_PRED_LEN", 1))
KRONOS_SAMPLE_COUNT = int(os.getenv("KRONOS_SAMPLE_COUNT", 5))

HERMES_ENABLED = os.getenv("HERMES_ENABLED", "true").lower() == "true"
HERMES_MODEL = os.getenv("HERMES_MODEL", OPENROUTER_MODEL)
