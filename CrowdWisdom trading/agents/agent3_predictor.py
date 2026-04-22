"""
agents/agent3_predictor.py
────────────────────────────────────────────────────────────────────
Agent 3: Predicts next 5-min UP/DOWN move using Kronos-inspired model.
Trains on OHLCV history, returns probability + confidence per asset.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from agents.agent2_data_fetcher_apify_tracking import fetch_all_assets, load_ohlcv
from tools.kronos_tools import KronosPredictor
from utils.config import ASSETS
from utils.logger import get_agent_logger
from utils.state_store import save_prediction

logger = get_agent_logger("agent3_predictor")

# Per-asset predictor instances (trained once, reused across loops)
_predictors: dict[str, KronosPredictor] = {}


def get_predictor(asset: str) -> KronosPredictor:
    """Get or create a KronosPredictor for an asset."""
    if asset not in _predictors:
        _predictors[asset] = KronosPredictor()
    return _predictors[asset]


def predict_asset(
    asset: str,
    df: Optional[pd.DataFrame] = None,
    retrain: bool = False,
) -> dict:
    """
    Run Kronos prediction for a single asset.

    Args:
        asset:   e.g. "BTC"
        df:      OHLCV DataFrame; loads from disk if None
        retrain: force model retraining

    Returns:
        {
          "asset": "BTC",
          "direction": "UP",
          "up_prob": 0.63,
          "down_prob": 0.37,
          "confidence": 0.26,
          "method": "ensemble_ml+rules",
          "signals": {...},
          "train_info": {...},
          "prediction_id": "uuid",
          "timestamp": "...",
        }
    """
    if df is None:
        df = load_ohlcv(asset)
        if df is None or df.empty:
            logger.info(f"No cached data for {asset}, fetching...")
            fetched = fetch_all_assets(assets=[asset])
            df = fetched.get(asset)

    if df is None or df.empty:
        logger.error(f"Cannot predict {asset}: no OHLCV data available")
        return _neutral_prediction(asset, "no_data")

    if len(df) < 50:
        logger.warning(f"{asset}: only {len(df)} bars — too few for reliable prediction")
        return _neutral_prediction(asset, "insufficient_data")

    predictor = get_predictor(asset)

    # Train / retrain if needed
    train_info = {}
    if not predictor.trained or retrain:
        logger.info(f"Training Kronos model for {asset} on {len(df)} bars...")
        train_info = predictor.train(df)

    # Predict
    pred = predictor.predict(df)
    pred["asset"] = asset
    pred["train_info"] = train_info
    pred["prediction_id"] = str(uuid.uuid4())[:8]
    pred["timestamp"] = datetime.now(timezone.utc).isoformat()
    pred["current_price"] = float(df["close"].iloc[-1])

    logger.info(
        f"  🔮 {asset}: {pred['direction']} "
        f"(up_prob={pred['up_prob']:.3f}, conf={pred['confidence']:.3f}, "
        f"method={pred['method']})"
    )
    return pred


def predict_all_assets(
    assets: list[str] = None,
    ohlcv_data: dict[str, pd.DataFrame] = None,
    retrain: bool = False,
) -> dict:
    """
    Run predictions for all assets.

    Returns:
        {
          "BTC": { ...prediction... },
          "ETH": { ...prediction... },
          "predicted_at": "...",
        }
    """
    assets = assets or ASSETS
    results = {"predicted_at": datetime.now(timezone.utc).isoformat()}

    for asset in assets:
        df = (ohlcv_data or {}).get(asset)
        pred = predict_asset(asset, df=df, retrain=retrain)
        results[asset] = pred

        # Persist to state store for feedback loop
        save_prediction(asset, {
            "id": pred["prediction_id"],
            "asset": asset,
            "direction": pred["direction"],
            "up_prob": pred["up_prob"],
            "confidence": pred["confidence"],
            "current_price": pred.get("current_price"),
            "method": pred["method"],
        })

    return results


def _neutral_prediction(asset: str, reason: str) -> dict:
    return {
        "asset": asset,
        "direction": "UP",
        "up_prob": 0.5,
        "down_prob": 0.5,
        "confidence": 0.0,
        "method": reason,
        "signals": {},
        "train_info": {},
        "prediction_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_price": None,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  🔮 AGENT 3 — KRONOS PREDICTOR")
    print("=" * 60)
    results = predict_all_assets(retrain=True)
    for asset in ASSETS:
        if asset in results:
            p = results[asset]
            print(f"\n  {asset}:")
            print(f"    Direction  : {p['direction']}")
            print(f"    UP Prob    : {p['up_prob']:.4f}")
            print(f"    Confidence : {p['confidence']:.4f}")
            print(f"    Method     : {p['method']}")
            print(f"    Signals    : {json.dumps(p.get('signals', {}), indent=6)}")
