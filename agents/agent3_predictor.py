"""
agents/agent3_predictor.py

Agent 3 predicts the next 5-minute UP/DOWN move.
It prefers the real Kronos foundation model and falls back to a local
directional model only when Kronos is not available on the machine.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from cw_utils.runtime import ensure_user_site

ensure_user_site()

import pandas as pd

from agents.agent2_data_fetcher_apify_tracking import fetch_all_assets, load_ohlcv
from cw_tools.kronos_tools import KronosPredictor
from cw_utils.config import ASSETS
from cw_utils.logger import get_agent_logger
from cw_utils.state_store import save_prediction

logger = get_agent_logger("agent3_predictor")

_predictors: dict[str, KronosPredictor] = {}


def get_predictor(asset: str) -> KronosPredictor:
    if asset not in _predictors:
        _predictors[asset] = KronosPredictor()
    return _predictors[asset]


def predict_asset(
    asset: str,
    df: Optional[pd.DataFrame] = None,
    retrain: bool = False,
) -> dict:
    """
    Run next-bar prediction for a single asset.
    """
    if df is None:
        df = load_ohlcv(asset)
        if df is None or df.empty:
            logger.info("No cached data for %s, fetching...", asset)
            fetched = fetch_all_assets(assets=[asset])
            df = fetched.get(asset)

    if df is None or df.empty:
        logger.error("Cannot predict %s: no OHLCV data available", asset)
        return _neutral_prediction(asset, "no_data")

    if len(df) < 50:
        logger.warning("%s: only %s bars, too few for reliable prediction", asset, len(df))
        return _neutral_prediction(asset, "insufficient_data")

    predictor = get_predictor(asset)
    train_info = {}

    if not predictor.trained or retrain:
        logger.info("Preparing predictor backend for %s on %s bars...", asset, len(df))
        train_info = predictor.train(df)

    pred = predictor.predict(df)
    pred["asset"] = asset
    pred["train_info"] = train_info
    pred["backend_status"] = {
        "name": predictor.backend_status.name,
        "ready": predictor.backend_status.ready,
        "detail": predictor.backend_status.detail,
    }
    pred["prediction_id"] = str(uuid.uuid4())[:8]
    pred["timestamp"] = datetime.now(timezone.utc).isoformat()
    pred["current_price"] = float(df["close"].iloc[-1])

    logger.info(
        "  %s: %s (up_prob=%.3f, conf=%.3f, method=%s, backend=%s)",
        asset,
        pred["direction"],
        pred["up_prob"],
        pred["confidence"],
        pred["method"],
        pred.get("backend", "?"),
    )
    return pred


def predict_all_assets(
    assets: list[str] = None,
    ohlcv_data: dict[str, pd.DataFrame] = None,
    retrain: bool = False,
) -> dict:
    assets = assets or ASSETS
    results = {"predicted_at": datetime.now(timezone.utc).isoformat()}

    for asset in assets:
        df = (ohlcv_data or {}).get(asset)
        pred = predict_asset(asset, df=df, retrain=retrain)
        results[asset] = pred

        save_prediction(
            asset,
            {
                "id": pred["prediction_id"],
                "asset": asset,
                "direction": pred["direction"],
                "up_prob": pred["up_prob"],
                "confidence": pred["confidence"],
                "current_price": pred.get("current_price"),
                "method": pred["method"],
                "backend": pred.get("backend"),
                "forecast_close": pred.get("forecast_close"),
                "forecast_delta_pct": pred.get("forecast_delta_pct"),
            },
        )

    return results


def _neutral_prediction(asset: str, reason: str) -> dict:
    return {
        "asset": asset,
        "direction": "UP",
        "up_prob": 0.5,
        "down_prob": 0.5,
        "confidence": 0.0,
        "method": reason,
        "backend": "none",
        "signals": {},
        "train_info": {},
        "backend_status": {"name": "none", "ready": False, "detail": reason},
        "prediction_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_price": None,
        "forecast_close": None,
        "forecast_delta_pct": None,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  AGENT 3 - KRONOS PREDICTOR")
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
            print(f"    Backend    : {p.get('backend')}")
            print(f"    Signals    : {json.dumps(p.get('signals', {}), indent=6)}")
