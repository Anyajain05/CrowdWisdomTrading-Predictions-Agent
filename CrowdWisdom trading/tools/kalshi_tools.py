"""
tools/kalshi_tools.py
────────────────────────────────────────────────────────────────────
Kalshi REST API wrappers.
Supports both demo and live environments.
Auth: RSA-PSS signed headers (or demo mode without keys).
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional

import httpx

from utils.config import env
from utils.logger import get_logger

logger = get_logger("kalshi")

USE_DEMO = env("KALSHI_USE_DEMO", "true").lower() == "true"
BASE_URL = (
    "https://demo-api.kalshi.co"
    if USE_DEMO
    else "https://trading-api.kalshi.com"
)
API_KEY_ID = env("KALSHI_API_KEY_ID", "")
PRIVATE_KEY_PATH = env("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")


def _load_private_key():
    """Load RSA private key from PEM file, if present."""
    path = Path(PRIVATE_KEY_PATH)
    if not path.exists():
        return None
    try:
        from cryptography.hazmat.primitives import serialization
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    except Exception as e:
        logger.warning(f"Could not load Kalshi private key: {e}")
        return None


_PRIVATE_KEY = _load_private_key()


def _auth_headers(method: str, path: str) -> dict:
    """Build RSA-PSS signed headers for Kalshi API."""
    if not _PRIVATE_KEY or not API_KEY_ID:
        return {}
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}".encode()
        signature = _PRIVATE_KEY.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }
    except Exception as e:
        logger.warning(f"Failed to sign Kalshi request: {e}")
        return {}


def _get(path: str, params: dict = None) -> Optional[dict]:
    """Make authenticated GET request to Kalshi API."""
    headers = _auth_headers("GET", path)
    url = f"{BASE_URL}{path}"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code == 401 and not API_KEY_ID:
                logger.debug(f"Kalshi: unauthenticated request to {path} (read-only public data)")
                # Try public endpoint without auth
                resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"Kalshi HTTP {e.response.status_code} on {path}: {e.response.text[:200]}")
    except Exception as e:
        logger.warning(f"Kalshi request error on {path}: {e}")
    return None


def get_markets(series_ticker: str = None, limit: int = 20) -> list[dict]:
    """
    List active Kalshi markets, optionally filtered by series ticker.
    E.g. series_ticker='KXBTC' returns all BTC markets.
    """
    params = {"limit": limit, "status": "open"}
    if series_ticker:
        params["series_ticker"] = series_ticker

    data = _get("/trade-api/v2/markets", params=params)
    if not data:
        return []
    markets = data.get("markets", [])
    logger.info(f"Kalshi markets ({series_ticker or 'all'}): {len(markets)} found")
    return markets


def get_crypto_5min_markets(asset: str) -> list[dict]:
    """
    Fetch Kalshi 5-minute price bracket markets for BTC or ETH.
    Returns list of enriched dicts with implied_up_prob.
    """
    prefixes = {
        "BTC": "KXBTC",
        "ETH": "KXETH",
        "SOL": "KXSOL",
        "DOGE": "KXDOGE",
    }
    series = prefixes.get(asset.upper(), f"KX{asset.upper()}")
    markets = get_markets(series_ticker=series, limit=30)

    results = []
    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        yes_bid = m.get("yes_bid", 0) / 100  # Kalshi prices are in cents
        yes_ask = m.get("yes_ask", 0) / 100
        yes_price = (yes_bid + yes_ask) / 2 if (yes_bid + yes_ask) > 0 else 0.5
        no_price = 1.0 - yes_price

        # Only include markets that expire soon (5-15 min horizon)
        # Filter by title keyword "5" or "15 min"
        if not any(kw in title.lower() for kw in ["5", "15", "minute", "min"]):
            continue

        results.append({
            "source": "kalshi",
            "asset": asset.upper(),
            "ticker": ticker,
            "title": title,
            "yes_price": round(yes_price, 4),
            "no_price": round(no_price, 4),
            "implied_up_prob": round(yes_price, 4),
            "volume": m.get("volume", 0),
            "open_interest": m.get("open_interest", 0),
            "close_time": m.get("close_time", ""),
        })

    # If no markets found (e.g., off-hours), return a neutral placeholder
    if not results:
        logger.info(f"Kalshi: no active 5-min markets for {asset}, using neutral placeholder")
        results.append({
            "source": "kalshi",
            "asset": asset.upper(),
            "ticker": f"{series}-PLACEHOLDER",
            "title": f"{asset} price (no live market)",
            "yes_price": 0.5,
            "no_price": 0.5,
            "implied_up_prob": 0.5,
            "volume": 0,
            "open_interest": 0,
            "close_time": "",
        })

    logger.info(f"Kalshi 5-min markets for {asset}: {len(results)} found")
    return results


def get_market_by_ticker(ticker: str) -> Optional[dict]:
    """Fetch a single Kalshi market by ticker."""
    return _get(f"/trade-api/v2/markets/{ticker}")


def get_15min_markets(asset: str) -> list[dict]:
    """
    Fetch 15-minute Kalshi markets for arbitrage comparison.
    """
    prefixes = {"BTC": "KXBTC", "ETH": "KXETH"}
    series = prefixes.get(asset.upper(), f"KX{asset.upper()}")
    markets = get_markets(series_ticker=series, limit=30)

    results = []
    for m in markets:
        title = m.get("title", "")
        if "15" not in title.lower():
            continue
        yes_bid = m.get("yes_bid", 0) / 100
        yes_ask = m.get("yes_ask", 0) / 100
        yes_price = (yes_bid + yes_ask) / 2 if (yes_bid + yes_ask) > 0 else 0.5
        results.append({
            "source": "kalshi_15min",
            "asset": asset.upper(),
            "ticker": m.get("ticker", ""),
            "title": title,
            "yes_price": round(yes_price, 4),
            "implied_up_prob": round(yes_price, 4),
            "close_time": m.get("close_time", ""),
        })
    return results
