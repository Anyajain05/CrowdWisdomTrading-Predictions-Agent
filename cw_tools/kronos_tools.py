"""
tools/kronos_tools.py

Real Kronos integration for next-bar crypto forecasting, with a safe fallback
backend when the Kronos repo or model weights are not available locally.
"""
from __future__ import annotations

import importlib
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from cw_utils.runtime import ensure_user_site

ensure_user_site()

import numpy as np
import pandas as pd

from cw_utils.config import (
    KRONOS_DEVICE,
    KRONOS_LOOKBACK,
    KRONOS_MAX_CONTEXT,
    KRONOS_MODEL_NAME,
    KRONOS_PRED_LEN,
    KRONOS_REPO_PATH,
    KRONOS_SAMPLE_COUNT,
    KRONOS_TOKENIZER_NAME,
)
from cw_utils.logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger("kronos")


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(prices: pd.Series, period: int = 20, std_mult: float = 2.0):
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    pct_b = (prices - lower) / (upper - lower + 1e-10)
    return upper, ma, lower, pct_b


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_cp = (df["high"] - df["close"].shift(1)).abs()
    low_cp = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    for lag in [1, 2, 3, 5, 10, 15, 30]:
        df[f"ret_{lag}"] = close.pct_change(lag)

    df["rsi_14"] = compute_rsi(close, 14)
    df["rsi_7"] = compute_rsi(close, 7)
    df["rsi_14_norm"] = (df["rsi_14"] - 50) / 50
    df["rsi_7_norm"] = (df["rsi_7"] - 50) / 50

    macd, signal, hist = compute_macd(close)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    df["macd_hist_norm"] = hist / (close + 1e-10)

    _, bb_mid, _, pct_b = compute_bollinger(close)
    df["bb_pct_b"] = pct_b
    df["price_vs_bb_mid"] = (close - bb_mid) / (bb_mid + 1e-10)

    df["atr"] = compute_atr(df)
    df["atr_pct"] = df["atr"] / (close + 1e-10)

    df["vol_ma_20"] = volume.rolling(20).mean()
    df["vol_ratio"] = volume / (df["vol_ma_20"] + 1e-10)

    df["mom_5"] = close / close.shift(5) - 1
    df["mom_10"] = close / close.shift(10) - 1

    df["body"] = (df["close"] - df["open"]) / (df["open"] + 1e-10)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["open"] + 1e-10)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["open"] + 1e-10)

    df["target"] = (close.shift(-1) > close).astype(int)
    return df


FEATURE_COLS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    "ret_30",
    "rsi_14_norm",
    "rsi_7_norm",
    "macd_hist_norm",
    "bb_pct_b",
    "price_vs_bb_mid",
    "atr_pct",
    "vol_ratio",
    "mom_5",
    "mom_10",
    "body",
    "upper_wick",
    "lower_wick",
]


@dataclass
class BackendStatus:
    name: str
    ready: bool
    detail: str


class FallbackDirectionalPredictor:
    """
    Lightweight fallback when Kronos foundation weights are unavailable.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=500, C=0.5, class_weight="balanced", random_state=42
        )
        self.trained = False

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        feat_df = build_features(df).dropna(subset=FEATURE_COLS + ["target"]).iloc[:-1]
        if len(feat_df) < 50:
            return {"status": "insufficient_data", "rows": len(feat_df)}

        X = self.scaler.fit_transform(feat_df[FEATURE_COLS].values)
        y = feat_df["target"].values
        self.model.fit(X, y)
        self.trained = True
        return {
            "status": "trained",
            "rows": len(feat_df),
            "train_accuracy": round(float(self.model.score(X, y)), 4),
        }

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        feat_df = build_features(df).dropna(subset=FEATURE_COLS)
        if feat_df.empty:
            return self._momentum_fallback(df)

        last_row = feat_df.iloc[-1]
        rule_pack = self._rule_based_signals(last_row)
        ml_up_prob = None

        if self.trained:
            X = self.scaler.transform(feat_df[FEATURE_COLS].iloc[-1:].values)
            ml_up_prob = float(self.model.predict_proba(X)[0][1])

        rule_up_prob = rule_pack["rule_up_prob"]
        if ml_up_prob is None:
            ensemble = rule_up_prob
            method = "fallback_rules_only"
        else:
            ensemble = 0.6 * ml_up_prob + 0.4 * rule_up_prob
            method = "fallback_ml_plus_rules"

        direction = "UP" if ensemble >= 0.5 else "DOWN"
        confidence = abs(ensemble - 0.5) * 2
        return {
            "direction": direction,
            "up_prob": round(float(ensemble), 4),
            "down_prob": round(float(1 - ensemble), 4),
            "confidence": round(float(confidence), 4),
            "method": method,
            "backend": "fallback",
            "signals": rule_pack["signals"],
            "forecast_close": None,
            "forecast_delta_pct": None,
        }

    def _rule_based_signals(self, row: pd.Series) -> dict[str, Any]:
        signals: dict[str, str] = {}
        votes: list[float] = []

        rsi = row.get("rsi_14", 50)
        if pd.notna(rsi):
            if rsi < 30:
                signals["rsi"] = "OVERSOLD_BUY"
                votes.append(0.7)
            elif rsi > 70:
                signals["rsi"] = "OVERBOUGHT_SELL"
                votes.append(0.3)
            else:
                signals["rsi"] = "NEUTRAL"
                votes.append(0.5)

        hist = row.get("macd_hist", 0)
        if pd.notna(hist):
            signals["macd"] = "BULLISH" if hist > 0 else "BEARISH"
            votes.append(0.65 if hist > 0 else 0.35)

        pct_b = row.get("bb_pct_b", 0.5)
        if pd.notna(pct_b):
            if pct_b < 0.1:
                signals["bb"] = "NEAR_LOWER_BAND_BUY"
                votes.append(0.65)
            elif pct_b > 0.9:
                signals["bb"] = "NEAR_UPPER_BAND_SELL"
                votes.append(0.35)
            else:
                signals["bb"] = "NEUTRAL"
                votes.append(0.5)

        mom = row.get("mom_5", 0)
        if pd.notna(mom):
            signals["momentum"] = "POSITIVE" if mom > 0 else "NEGATIVE"
            votes.append(0.55 if mom > 0 else 0.45)

        vol_ratio = row.get("vol_ratio", 1.0)
        if pd.notna(vol_ratio) and vol_ratio > 1.5:
            body = row.get("body", 0)
            signals["volume"] = "HIGH_VOLUME_UP" if body > 0 else "HIGH_VOLUME_DOWN"
            votes.append(0.65 if body > 0 else 0.35)

        return {
            "rule_up_prob": float(np.mean(votes)) if votes else 0.5,
            "signals": signals,
        }

    def _momentum_fallback(self, df: pd.DataFrame) -> dict[str, Any]:
        if len(df) < 2:
            up_prob = 0.5
        else:
            last_return = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
            up_prob = 0.55 if last_return > 0 else 0.45

        return {
            "direction": "UP" if up_prob >= 0.5 else "DOWN",
            "up_prob": round(float(up_prob), 4),
            "down_prob": round(float(1 - up_prob), 4),
            "confidence": 0.1,
            "method": "fallback_momentum",
            "backend": "fallback",
            "signals": {},
            "forecast_close": None,
            "forecast_delta_pct": None,
        }


class KronosPredictor:
    """
    Uses the real Kronos forecasting stack when available.

    Expected setup:
    - clone https://github.com/shiyu-coder/Kronos
    - set KRONOS_REPO_PATH to that clone
    - install torch/transformers/huggingface-hub/safetensors
    """

    def __init__(self):
        self.trained = False
        self._fallback = FallbackDirectionalPredictor()
        self._kronos_predictor = None
        self._backend_status = self._load_kronos_backend()

    @property
    def backend_status(self) -> BackendStatus:
        return self._backend_status

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        if self._kronos_predictor is not None:
            self.trained = True
            return {
                "status": "pretrained_model_loaded",
                "backend": "kronos_foundation",
                "rows": int(len(df)),
                "lookback_used": int(min(len(df), KRONOS_LOOKBACK, KRONOS_MAX_CONTEXT)),
            }

        info = self._fallback.train(df)
        self.trained = self._fallback.trained
        info["backend"] = "fallback"
        info["detail"] = self._backend_status.detail
        return info

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        if self._kronos_predictor is not None:
            try:
                return self._predict_with_kronos(df)
            except Exception as exc:
                logger.warning("Kronos backend failed at inference, falling back: %s", exc)

        fallback_result = self._fallback.predict(df)
        fallback_result["backend_detail"] = self._backend_status.detail
        return fallback_result

    def _load_kronos_backend(self) -> BackendStatus:
        try:
            if KRONOS_REPO_PATH:
                repo = Path(KRONOS_REPO_PATH).expanduser().resolve()
                if repo.exists():
                    repo_str = str(repo)
                    if repo_str not in sys.path:
                        sys.path.insert(0, repo_str)

            model_module = importlib.import_module("model")
            kronos_cls = getattr(model_module, "Kronos")
            tokenizer_cls = getattr(model_module, "KronosTokenizer")
            predictor_cls = getattr(model_module, "KronosPredictor")

            tokenizer = tokenizer_cls.from_pretrained(KRONOS_TOKENIZER_NAME)
            model = kronos_cls.from_pretrained(KRONOS_MODEL_NAME)
            self._kronos_predictor = predictor_cls(
                model,
                tokenizer,
                device=KRONOS_DEVICE,
                max_context=KRONOS_MAX_CONTEXT,
            )
            logger.info("Loaded real Kronos backend: %s / %s", KRONOS_MODEL_NAME, KRONOS_TOKENIZER_NAME)
            return BackendStatus(
                name="kronos_foundation",
                ready=True,
                detail=f"Loaded {KRONOS_MODEL_NAME} with tokenizer {KRONOS_TOKENIZER_NAME}",
            )
        except Exception as exc:
            detail = (
                "Real Kronos unavailable. Set KRONOS_REPO_PATH to a local clone of "
                "https://github.com/shiyu-coder/Kronos and install its dependencies. "
                f"Load error: {exc}"
            )
            logger.warning(detail)
            return BackendStatus(name="fallback", ready=False, detail=detail)

    def _predict_with_kronos(self, df: pd.DataFrame) -> dict[str, Any]:
        prepared = self._prepare_kronos_frame(df)
        lookback_df = prepared.tail(min(len(prepared), KRONOS_LOOKBACK, KRONOS_MAX_CONTEXT)).copy()
        x_timestamp = lookback_df["timestamp"]
        y_timestamp = self._future_timestamps(x_timestamp.iloc[-1], KRONOS_PRED_LEN)

        forecast_df = self._kronos_predictor.predict(
            df=lookback_df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=KRONOS_PRED_LEN,
            T=1.0,
            top_p=0.9,
            sample_count=KRONOS_SAMPLE_COUNT,
        )
        if forecast_df is None or len(forecast_df) == 0:
            raise RuntimeError("Kronos returned an empty forecast")

        forecast_close = float(forecast_df["close"].iloc[0])
        last_close = float(lookback_df["close"].iloc[-1])
        delta_pct = (forecast_close / last_close) - 1

        volatility = float(lookback_df["close"].pct_change().tail(48).std() or 0.0)
        scale = max(volatility * 4, 1e-4)
        up_prob = 1.0 / (1.0 + math.exp(-(delta_pct / scale)))
        direction = "UP" if forecast_close >= last_close else "DOWN"
        confidence = min(abs(up_prob - 0.5) * 2, 0.999)

        return {
            "direction": direction,
            "up_prob": round(float(up_prob), 4),
            "down_prob": round(float(1 - up_prob), 4),
            "confidence": round(float(confidence), 4),
            "method": "kronos_forecast_close",
            "backend": "kronos_foundation",
            "signals": {
                "lookback_bars": int(len(lookback_df)),
                "pred_len": int(KRONOS_PRED_LEN),
                "device": KRONOS_DEVICE,
            },
            "forecast_close": round(forecast_close, 6),
            "forecast_delta_pct": round(float(delta_pct), 6),
        }

    def _prepare_kronos_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [str(col).lower() for col in frame.columns]

        if "timestamp" not in frame.columns:
            if "open_time" in frame.columns:
                frame["timestamp"] = pd.to_datetime(frame["open_time"], utc=True)
            elif "timestamps" in frame.columns:
                frame["timestamp"] = pd.to_datetime(frame["timestamps"], utc=True)
            else:
                freq = self._infer_frequency(frame)
                start = pd.Timestamp.utcnow().floor(freq) - (len(frame) - 1) * pd.to_timedelta(freq)
                frame["timestamp"] = pd.date_range(start=start, periods=len(frame), freq=freq, tz="UTC")
        else:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

        for col in ["open", "high", "low", "close"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

        if "amount" not in frame.columns:
            frame["amount"] = frame["close"] * frame["volume"]
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(frame["close"] * frame["volume"])

        frame = frame.dropna(subset=["open", "high", "low", "close", "timestamp"]).sort_values("timestamp")
        return frame.reset_index(drop=True)

    def _future_timestamps(self, last_ts: pd.Timestamp, pred_len: int) -> pd.Series:
        freq = pd.to_timedelta(self._infer_freq_from_last_timestamp(last_ts))
        timestamps = [last_ts + freq * step for step in range(1, pred_len + 1)]
        return pd.Series(pd.to_datetime(timestamps, utc=True))

    def _infer_frequency(self, frame: pd.DataFrame) -> str:
        if "timestamp" in frame.columns and len(frame) > 2:
            ordered = pd.to_datetime(frame["timestamp"], utc=True).sort_values()
            diffs = ordered.diff().dropna()
            if not diffs.empty:
                return self._timedelta_to_freq(diffs.mode().iloc[0])
        return "5min"

    def _infer_freq_from_last_timestamp(self, _last_ts: pd.Timestamp) -> str:
        return "5min"

    def _timedelta_to_freq(self, value: pd.Timedelta) -> str:
        minutes = max(int(value / timedelta(minutes=1)), 1)
        return f"{minutes}min"
