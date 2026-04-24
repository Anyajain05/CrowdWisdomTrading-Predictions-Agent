"""
Agent 2: fetches OHLCV data from Apify with Binance as a free fallback.

The output is normalized to a consistent schema so downstream agents can rely
on the same columns regardless of data source.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cw_utils.runtime import ensure_user_site

ensure_user_site()

import pandas as pd


@dataclass
class ApifyUsageTracker:
    total_calls: int = 0
    successful_calls: int = 0
    fallback_calls: int = 0
    total_rows_fetched: int = 0
    estimated_cu: float = 0.0
    estimated_cost_usd: float = 0.0
    log_path: Path = field(default_factory=lambda: Path("./data/apify_usage.jsonl"))

    CU_PER_RUN: float = 0.1
    COST_PER_RUN_USD: float = 0.015

    def record_apify_call(self, asset: str, rows: int, success: bool, duration_s: float = 0.0):
        self.total_calls += 1
        if success:
            self.successful_calls += 1
            self.total_rows_fetched += rows
            self.estimated_cu += self.CU_PER_RUN
            self.estimated_cost_usd += self.COST_PER_RUN_USD
        self._persist(asset, rows, "apify", success, duration_s)

    def record_fallback_call(self, asset: str, rows: int, duration_s: float = 0.0):
        self.fallback_calls += 1
        self.total_rows_fetched += rows
        self._persist(asset, rows, "binance_fallback", True, duration_s)

    def _persist(self, asset, rows, source, success, duration_s):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "source": source,
            "rows": rows,
            "success": success,
            "duration_s": round(duration_s, 2),
            "cumulative_cu": round(self.estimated_cu, 3),
            "cumulative_cost_usd": round(self.estimated_cost_usd, 4),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def summary(self) -> dict:
        return {
            "total_apify_calls": self.total_calls,
            "successful_apify_calls": self.successful_calls,
            "binance_fallback_calls": self.fallback_calls,
            "total_rows_fetched": self.total_rows_fetched,
            "estimated_cu": round(self.estimated_cu, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }

    def print_summary(self):
        summary = self.summary()
        print("\n" + "-" * 50)
        print("  Apify Usage Summary")
        print("-" * 50)
        print(f"  Apify actor calls   : {summary['total_apify_calls']} ({summary['successful_apify_calls']} ok)")
        print(f"  Binance fallback    : {summary['binance_fallback_calls']} calls")
        print(f"  Total rows fetched  : {summary['total_rows_fetched']:,}")
        print(f"  Est. compute units  : ~{summary['estimated_cu']} CU")
        print(f"  Est. cost (Apify)   : ~${summary['estimated_cost_usd']:.4f}")
        print("-" * 50 + "\n")


usage_tracker = ApifyUsageTracker()


def _pick_column(frame: pd.DataFrame, names: list[str], default=None):
    lower_map = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lower_map:
            return frame[lower_map[name.lower()]]
    return default


def _normalize_ohlcv(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "amount"])

    df = raw_df.copy()

    open_time = _pick_column(df, ["open_time", "timestamp", "timestamps", "time", "date"])
    if open_time is None:
        raise ValueError(f"Could not identify timestamp column from Apify response: {list(df.columns)}")

    normalized = pd.DataFrame()
    normalized["open_time"] = pd.to_datetime(open_time, utc=True, errors="coerce")
    normalized["open"] = pd.to_numeric(_pick_column(df, ["open", "o"]), errors="coerce")
    normalized["high"] = pd.to_numeric(_pick_column(df, ["high", "h"]), errors="coerce")
    normalized["low"] = pd.to_numeric(_pick_column(df, ["low", "l"]), errors="coerce")
    normalized["close"] = pd.to_numeric(_pick_column(df, ["close", "c"]), errors="coerce")
    volume = _pick_column(df, ["volume", "vol", "base_volume", "baseassetvolume"], 0.0)
    normalized["volume"] = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    amount = _pick_column(df, ["amount", "quote_volume", "quoteassetvolume", "turnover"])
    if amount is None:
        normalized["amount"] = normalized["close"] * normalized["volume"]
    else:
        normalized["amount"] = pd.to_numeric(amount, errors="coerce").fillna(normalized["close"] * normalized["volume"])

    normalized = normalized.dropna(subset=["open_time", "open", "high", "low", "close"])
    normalized = normalized.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return normalized


def fetch_ohlcv_apify(asset: str, bars: int = 1000, interval: str = "5m"):
    from apify_client import ApifyClient
    from cw_utils.config import APIFY_TOKEN

    t0 = time.time()
    client = ApifyClient(APIFY_TOKEN)
    run_input = {
        "symbol": f"{asset}USDT",
        "interval": interval,
        "limit": bars,
    }
    run = client.actor("jan.sirucek/binance-scraper").call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    if not items:
        raise ValueError("Apify returned empty dataset")

    normalized = _normalize_ohlcv(pd.DataFrame(items))
    usage_tracker.record_apify_call(
        asset=asset,
        rows=len(normalized),
        success=not normalized.empty,
        duration_s=time.time() - t0,
    )
    if normalized.empty:
        raise ValueError("Apify response could not be normalized into OHLCV rows")
    return normalized


def fetch_ohlcv_binance(asset: str, bars: int = 1000, interval: str = "5m"):
    import requests

    t0 = time.time()
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": f"{asset}USDT", "interval": interval, "limit": bars}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    raw = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    normalized = pd.DataFrame(
        {
            "open_time": df["open_time"],
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "amount": df["quote_volume"].fillna(df["close"] * df["volume"]),
        }
    ).dropna(subset=["open", "high", "low", "close"])

    usage_tracker.record_fallback_call(
        asset=asset,
        rows=len(normalized),
        duration_s=time.time() - t0,
    )
    return normalized.reset_index(drop=True)


def fetch_all_assets(assets: list[str] = None, save: bool = True):
    from cw_utils.config import ASSETS, DATA_DIR, OHLCV_BARS, OHLCV_INTERVAL

    assets = assets or ASSETS
    result = {}

    for asset in assets:
        print(f"Fetching OHLCV for {asset}...")
        try:
            df = fetch_ohlcv_apify(asset, bars=OHLCV_BARS, interval=OHLCV_INTERVAL)
        except Exception as apify_error:
            print(f"  Apify attempt failed for {asset}: {apify_error}. Falling back to Binance...")
            try:
                df = fetch_ohlcv_binance(asset, bars=OHLCV_BARS, interval=OHLCV_INTERVAL)
            except Exception as fallback_error:
                print(f"  Failed to fetch {asset}: {fallback_error}")
                result[asset] = pd.DataFrame()
                continue

        result[asset] = df
        if save and not df.empty:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_DIR / f"{asset}_ohlcv.csv", index=False)

    usage_tracker.print_summary()
    return result


def load_ohlcv(asset: str):
    from cw_utils.config import DATA_DIR

    path = DATA_DIR / f"{asset}_ohlcv.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    return df


def get_latest_price(asset: str) -> float:
    df = load_ohlcv(asset)
    if not df.empty and "close" in df.columns:
        return float(df["close"].iloc[-1])
    return 0.0
