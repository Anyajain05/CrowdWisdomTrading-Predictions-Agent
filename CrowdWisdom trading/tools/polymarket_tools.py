"""
tools/polymarket_tools.py
────────────────────────────────────────────────────────────────────
Polymarket Gamma + CLOB API read-only wrappers.
No auth required for market data.
"""
from __future__ import annotations

import re
import time
from typing import Optional

import requests

from utils.logger import get_logger

logger = get_logger("polymarket")

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# ── Known token IDs for BTC/ETH 5-min markets on Polymarket ──────
# These are illustrative; real IDs change as markets roll over.
# Agent 1 will search dynamically via the Gamma API keyword search.

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "CrowdWisdomTrading/1.0"})


def _get(url: str, params: dict = None, retries: int = 1) -> Optional[dict | list]:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=3)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} fetching {url}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"Request error ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def search_crypto_markets(keyword: str, limit: int = 20) -> list[dict]:
    """
    Search Gamma API for active markets matching a keyword.
    Returns list of market dicts with question, prices, volume.
    """
    data = _get(
        f"{GAMMA_API}/markets",
        params={
            "q": keyword,
            "active": "true",
            "closed": "false",
            "limit": limit,
        },
    )
    if not data:
        return []
    markets = data if isinstance(data, list) else data.get("markets", [])
    logger.info(f"Polymarket search '{keyword}': found {len(markets)} markets")
    return markets


def get_crypto_5min_markets(asset: str) -> list[dict]:
    """
    Find 5-minute up/down prediction markets for a given crypto asset.
    Returns enriched dicts with: question, yes_price, no_price, implied_up_prob.
    """
    keywords = {
        "BTC": ["bitcoin 5-minute", "btc 5min", "bitcoin price"],
        "ETH": ["ethereum 5-minute", "eth 5min", "ethereum price"],
        "SOL": ["solana 5-minute", "sol price"],
        "DOGE": ["dogecoin 5-minute", "doge price"],
    }
    search_terms = keywords.get(asset.upper(), [f"{asset.lower()} 5-minute"])
    
    results = []
    seen_questions = set()
    
    for term in search_terms:
        markets = search_crypto_markets(term, limit=10)
        for m in markets:
            q = m.get("question", "")
            if q in seen_questions:
                continue
            seen_questions.add(q)
            
            # Parse outcome prices — Polymarket stores them as strings
            outcome_prices = m.get("outcomePrices", [])
            outcomes = m.get("outcomes", [])
            
            yes_price = 0.5
            no_price = 0.5
            
            if outcome_prices and len(outcome_prices) >= 2:
                try:
                    # outcomes[0] is usually YES, outcomes[1] is NO
                    p0 = float(outcome_prices[0])
                    p1 = float(outcome_prices[1])
                    if "yes" in str(outcomes[0]).lower() or "up" in str(outcomes[0]).lower():
                        yes_price = p0
                        no_price = p1
                    else:
                        yes_price = p1
                        no_price = p0
                except (ValueError, IndexError):
                    pass
            
            results.append({
                "source": "polymarket",
                "asset": asset.upper(),
                "question": q,
                "yes_price": round(yes_price, 4),
                "no_price": round(no_price, 4),
                "implied_up_prob": round(yes_price, 4),
                "volume": m.get("volume", 0),
                "liquidity": m.get("liquidity", 0),
                "market_id": m.get("id", ""),
                "token_id": (m.get("clobTokenIds") or [""])[0],
                "end_date": m.get("endDate", ""),
            })
    
    logger.info(f"Polymarket 5-min markets for {asset}: {len(results)} found")
    return results


def get_clob_price(token_id: str) -> Optional[dict]:
    """
    Get live bid/ask for a token from the CLOB API.
    Returns: {mid, buy, sell, spread}
    """
    if not token_id:
        return None
    mid_data = _get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
    if mid_data and "mid" in mid_data:
        mid = float(mid_data["mid"])
        return {
            "mid": mid,
            "implied_prob": round(mid, 4),
        }
    return None


def get_market_orderbook(token_id: str) -> Optional[dict]:
    """Fetch order book for a given token_id."""
    return _get(f"{CLOB_API}/book", params={"token_id": token_id})


def get_recent_trades(token_id: str, limit: int = 20) -> list[dict]:
    """Fetch recent trades for market-as-price-signal."""
    data = _get(
        f"{CLOB_API}/trades",
        params={"token_id": token_id, "limit": limit},
    )
    if isinstance(data, list):
        return data
    return []
