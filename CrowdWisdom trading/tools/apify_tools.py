"""
tools/apify_tools.py
────────────────────────────────────────────────────────────────────
Fetches OHLCV data from Binance via Apify Actor.
Actor: dtrungtin/binance-ohlcv-scraper (free tier).
Fallback: direct Binance public REST API (no key needed).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from utils.config import APIFY_TOKEN, OHLCV_BARS, OHLCV_INTERVAL, env
from utils.logger import get_logger

logger = get_logger("apify")

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = "dtrungtin/binance-ohlcv-scraper"

SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "MATIC": "MATICUSDT",
}

COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
           "close_time", "quote_volume", "trades", "taker_buy_base",
           "taker_buy_quote", "ignore"]


# ── Primary: Apify Actor ──────────────────────────────────────────

def fetch_via_apify(symbol: str, interval: str = "5m", limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Run Apify Actor to fetch OHLCV. Returns DataFrame or None on failure.
    """
    if not APIFY_TOKEN:
        logger.warning("APIFY_API_TOKEN not set, skipping Apify fetch")
        return None

    binance_symbol = SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}USDT")

    run_input = {
        "symbol": binance_symbol,
        "interval": interval,
        "limit": limit,
    }

    logger.info(f"Apify: starting actor for {binance_symbol} {interval} x{limit}")
    try:
        import apify_client
        client = apify_client.ApifyClient(APIFY_TOKEN)
        run = client.actor(ACTOR_ID).call(run_input=run_input, timeout_secs=120)

        if run["status"] != "SUCCEEDED":
            logger.warning(f"Apify actor status: {run['status']}")
            return None

        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        if not items:
            logger.warning("Apify actor returned empty dataset")
            return None

        df = pd.DataFrame(items)
        df = _normalize_apify_df(df)
        logger.info(f"Apify: fetched {len(df)} bars for {binance_symbol}")
        return df

    except ImportError:
        logger.warning("apify-client not installed, falling back to Binance direct API")
        return None
    except Exception as e:
        logger.warning(f"Apify actor error: {e}")
        return None


# ── Fallback: Binance Public REST API (no key needed) ─────────────

def fetch_via_binance(symbol: str, interval: str = "5m", limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV directly from Binance public API.
    No API key required. Rate limit: 1200 requests/min.
    """
    binance_symbol = SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}USDT")
    url = "https://api.binance.com/api/v3/klines"

    # Binance max per request = 1000; paginate if needed
    all_klines = []
    end_time = None

    batches_needed = (limit + 999) // 1000
    for batch in range(batches_needed):
        batch_limit = min(1000, limit - len(all_klines))
        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "limit": batch_limit,
        }
        if end_time:
            params["endTime"] = end_time

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            klines = resp.json()
            if not klines:
                break
            all_klines = klines + all_klines
            end_time = klines[0][0] - 1  # Paginate backwards
            if len(klines) < batch_limit:
                break
            time.sleep(0.1)  # Be polite to Binance
        except Exception as e:
            logger.warning(f"Binance API error (batch {batch}): {e}")
            break

    if not all_klines:
        logger.error(f"Binance: no data returned for {binance_symbol}")
        return None

    df = pd.DataFrame(all_klines, columns=COLUMNS)
    df = _normalize_binance_df(df)
    logger.info(f"Binance fallback: fetched {len(df)} bars for {binance_symbol}")
    return df


def _normalize_binance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Binance klines DataFrame."""
    df = df.copy()
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def _normalize_apify_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Apify actor output DataFrame."""
    # Apify actor may return different column names
    rename_map = {
        "openTime": "open_time",
        "closeTime": "close_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open_time" in df.columns:
        if df["open_time"].dtype == "int64":
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        else:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    return df


# ── Public Interface ──────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str = None, limit: int = None) -> pd.DataFrame:
    """
    Fetch OHLCV data: tries Apify first, falls back to Binance direct.
    Always returns a DataFrame (may raise on complete failure).
    """
    interval = interval or OHLCV_INTERVAL
    limit = limit or OHLCV_BARS

    df = fetch_via_apify(symbol, interval, limit)
    if df is not None and len(df) > 10:
        return df

    logger.info(f"Falling back to Binance direct API for {symbol}")
    df = fetch_via_binance(symbol, interval, limit)
    if df is not None and len(df) > 10:
        return df

    raise RuntimeError(f"Failed to fetch OHLCV data for {symbol} from all sources")
