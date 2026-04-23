"""
agent2_data_fetcher.py  — APIFY USAGE TRACKING ADDITIONS
────────────────────────────────────────────────────────────────────
Drop-in additions for your existing agent2_data_fetcher.py.

1. Add these imports at the top of agent2_data_fetcher.py
2. Replace your existing fetch_ohlcv_apify() function body with the one below
3. The usage_tracker is a module-level singleton — import it anywhere for stats
"""

# ── ADD THESE IMPORTS to the top of agent2_data_fetcher.py ───────
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# ── ADD THIS CLASS before your fetch functions ────────────────────

@dataclass
class ApifyUsageTracker:
    """
    Tracks Apify actor calls and estimates compute unit / credit usage.
    Persists a lightweight log to data/apify_usage.jsonl for the dashboard.
    """
    total_calls: int = 0
    successful_calls: int = 0
    fallback_calls: int = 0        # times Binance fallback was used
    total_rows_fetched: int = 0
    estimated_cu: float = 0.0      # compute units (~0.1 CU per run)
    estimated_cost_usd: float = 0.0  # ~$0.015 per actor run on free tier
    log_path: Path = field(default_factory=lambda: Path("./data/apify_usage.jsonl"))

    CU_PER_RUN: float = 0.1          # empirical: short actor, 1000 rows
    COST_PER_RUN_USD: float = 0.015  # apify free tier rate

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
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # never crash the pipeline for logging

    def get_live_usage(self) -> dict:
        """
        Fetch real-time usage from Apify API.
        Returns a dict with plan info + monthly usage, or None on error.
        """
        try:
            from apify_client import ApifyClient
            from utils.config import APIFY_TOKEN
            token = APIFY_TOKEN
            if not token:
                return {}
            client = ApifyClient(token)
            user = client.user().get()
            usage = user.get("usage", {})
            plan  = user.get("plan",  {})
            return {
                "username":              user.get("username"),
                "plan_id":               plan.get("id", "free"),
                "monthly_limit_usd":     plan.get("monthlyUsageCreditsUsd", 5.0),
                "monthly_usage_usd":     usage.get("monthlyUsageUsd",      0.0),
                "monthly_usage_credits": usage.get("monthlyUsageCredits",  0.0),
                "compute_units_used":    usage.get("monthlyComputeUnits",  0.0),
            }
        except Exception as e:
            return {"error": str(e)}

    def summary(self) -> dict:
        return {
            "total_apify_calls":    self.total_calls,
            "successful_apify_calls": self.successful_calls,
            "binance_fallback_calls": self.fallback_calls,
            "total_rows_fetched":   self.total_rows_fetched,
            "estimated_cu":         round(self.estimated_cu, 3),
            "estimated_cost_usd":   round(self.estimated_cost_usd, 4),
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "─" * 50)
        print("  🧾  Apify Usage Summary")
        print("─" * 50)
        print(f"  Apify actor calls   : {s['total_apify_calls']}  (✅ {s['successful_apify_calls']} ok)")
        print(f"  Binance fallback    : {s['binance_fallback_calls']} calls (free)")
        print(f"  Total rows fetched  : {s['total_rows_fetched']:,}")
        print(f"  Est. compute units  : ~{s['estimated_cu']} CU")
        print(f"  Est. cost (Apify)   : ~${s['estimated_cost_usd']:.4f}")
        live = self.get_live_usage()
        if live and "monthly_usage_usd" in live:
            print(f"  Live monthly spend  : ${live['monthly_usage_usd']:.4f} / ${live['monthly_limit_usd']:.2f}")
        print("─" * 50 + "\n")


# Module-level singleton — import from other modules like:
#   from agents.agent2_data_fetcher import usage_tracker
usage_tracker = ApifyUsageTracker()


# ── UPDATED fetch_ohlcv_apify() — wrap your existing function ────
# Replace the body of your current fetch_ohlcv_apify(asset, bars, interval)
# with this pattern (keep your existing Apify actor call logic):

def fetch_ohlcv_apify(asset: str, bars: int = 1000, interval: str = "5m"):
    """Fetch OHLCV via Apify with usage tracking + Binance fallback."""
    import pandas as pd
    from apify_client import ApifyClient
    from utils.config import APIFY_TOKEN
    token = APIFY_TOKEN
    t0 = time.time()

    # ── Primary: Apify actor ──────────────────────────────────────
    try:
        client = ApifyClient(token)
        run_input = {
            "symbol": f"{asset}USDT",
            "interval": interval,
            "limit": bars,
        }
        run = client.actor("jan.sirucek/binance-scraper").call(run_input=run_input)
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

        if not items:
            raise ValueError("Apify returned empty dataset")

        df = pd.DataFrame(items)
        # … your existing column mapping / cleaning …

        usage_tracker.record_apify_call(
            asset=asset,
            rows=len(df),
            success=True,
            duration_s=time.time() - t0,
        )
        return df

    except Exception as apify_err:
        print(f"  ⚠️ Apify attempt failed: {apify_err}. Falling back to Binance...")
        # ── Fallback: Binance public REST API (free, no key) ─────
        import requests
        t1 = time.time()
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": f"{asset}USDT", "interval": interval, "limit": bars}
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            cols = ["open_time","open","high","low","close","volume",
                    "close_time","quote_vol","trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(raw, columns=cols)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

            usage_tracker.record_fallback_call(
                asset=asset,
                rows=len(df),
                duration_s=time.time() - t1,
            )
            return df
        except Exception as fallback_err:
            usage_tracker.record_apify_call(asset=asset, rows=0, success=False,
                                            duration_s=time.time() - t0)
            raise RuntimeError(
                f"Both Apify and Binance failed for {asset}. "
                f"Apify: {apify_err}. Binance: {fallback_err}"
            )


# ── ADD THIS CALL at the end of fetch_all_assets() ───────────────
# After your loop completes, call:
#   usage_tracker.print_summary()
# This will print a clean summary after every pipeline run.

def fetch_all_assets(assets: list[str] = None, save: bool = True):
    from utils.config import ASSETS, DATA_DIR, OHLCV_BARS, OHLCV_INTERVAL, APIFY_TOKEN
    import pandas as pd
    
    assets = assets or ASSETS
    result = {}
    
    for asset in assets:
        print(f"Fetching OHLCV for {asset}...")
        try:
            df = fetch_ohlcv_apify(asset, bars=OHLCV_BARS, interval=OHLCV_INTERVAL)
            result[asset] = df
            if save and not df.empty:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(DATA_DIR / f"{asset}_ohlcv.csv", index=False)
        except Exception as e:
            print(f"  ❌ Failed to fetch {asset}: {e}")
            result[asset] = pd.DataFrame()
            
    usage_tracker.print_summary()
    return result

def load_ohlcv(asset: str):
    from utils.config import DATA_DIR
    import pandas as pd
    path = DATA_DIR / f"{asset}_ohlcv.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def get_latest_price(asset: str) -> float:
    df = load_ohlcv(asset)
    if not df.empty and "close" in df.columns:
        return float(df["close"].iloc[-1])
    return 0.0
