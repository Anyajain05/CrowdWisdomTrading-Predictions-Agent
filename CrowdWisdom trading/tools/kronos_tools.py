"""
tools/kronos_tools.py
────────────────────────────────────────────────────────────────────
Kronos-inspired multi-horizon crypto price direction forecast.

Inspired by: https://github.com/shiyu-coder/Kronos
Key ideas:
  - Multi-scale temporal feature extraction
  - Return-based features at multiple lookbacks (1, 3, 5, 15, 30 bars)
  - Technical indicators: RSI, MACD, Bollinger Bands, ATR
  - Volatility-adjusted directional prediction
  - Ensemble of logistic regression + rule-based signals
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from utils.logger import get_logger

logger = get_logger("kronos")


# ── Feature Engineering ────────────────────────────────────────────

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
    """
    Build Kronos-inspired feature matrix from OHLCV DataFrame.
    Returns df with feature columns added.
    """
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    # ── Multi-scale returns ────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 15, 30]:
        df[f"ret_{lag}"] = close.pct_change(lag)

    # ── RSI ────────────────────────────────────────────────────────
    df["rsi_14"] = compute_rsi(close, 14)
    df["rsi_7"] = compute_rsi(close, 7)
    # Normalize RSI to [-1, 1]: RSI 50 → 0
    df["rsi_14_norm"] = (df["rsi_14"] - 50) / 50
    df["rsi_7_norm"] = (df["rsi_7"] - 50) / 50

    # ── MACD ───────────────────────────────────────────────────────
    macd, signal, hist = compute_macd(close)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    df["macd_hist_norm"] = hist / (close + 1e-10)

    # ── Bollinger Bands ─────────────────────────────────────────────
    _, bb_mid, _, pct_b = compute_bollinger(close)
    df["bb_pct_b"] = pct_b  # 0=at lower band, 1=at upper band
    df["price_vs_bb_mid"] = (close - bb_mid) / (bb_mid + 1e-10)

    # ── ATR (volatility) ──────────────────────────────────────────
    df["atr"] = compute_atr(df)
    df["atr_pct"] = df["atr"] / (close + 1e-10)

    # ── Volume features ───────────────────────────────────────────
    df["vol_ma_20"] = volume.rolling(20).mean()
    df["vol_ratio"] = volume / (df["vol_ma_20"] + 1e-10)

    # ── Price momentum ────────────────────────────────────────────
    df["mom_5"] = close / close.shift(5) - 1
    df["mom_10"] = close / close.shift(10) - 1

    # ── Candle body & wick features ───────────────────────────────
    df["body"] = (df["close"] - df["open"]) / (df["open"] + 1e-10)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["open"] + 1e-10)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["open"] + 1e-10)

    # ── Target: next-bar direction (1=up, 0=down) ─────────────────
    df["target"] = (close.shift(-1) > close).astype(int)

    return df


FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_15", "ret_30",
    "rsi_14_norm", "rsi_7_norm",
    "macd_hist_norm",
    "bb_pct_b", "price_vs_bb_mid",
    "atr_pct",
    "vol_ratio",
    "mom_5", "mom_10",
    "body", "upper_wick", "lower_wick",
]


# ── Kronos-Style Predictor ────────────────────────────────────────

class KronosPredictor:
    """
    Lightweight Kronos-inspired directional predictor.
    Uses sklearn LogisticRegression trained on the feature set above.
    Augmented with rule-based signal ensemble.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=500, C=0.5, class_weight="balanced", random_state=42
        )
        self.trained = False
        self.feature_importances: dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> dict:
        """Train on historical OHLCV data."""
        feat_df = build_features(df)
        feat_df = feat_df.dropna(subset=FEATURE_COLS + ["target"])
        # Use all but last row (target is unknown for last)
        feat_df = feat_df.iloc[:-1]

        if len(feat_df) < 50:
            logger.warning(f"Insufficient data for training: {len(feat_df)} rows")
            return {"status": "insufficient_data", "rows": len(feat_df)}

        X = feat_df[FEATURE_COLS].values
        y = feat_df["target"].values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True

        # Compute importances from coefficients
        for feat, coef in zip(FEATURE_COLS, self.model.coef_[0]):
            self.feature_importances[feat] = float(coef)

        train_accuracy = self.model.score(X_scaled, y)
        logger.info(f"Kronos model trained: {len(feat_df)} rows, train_acc={train_accuracy:.3f}")
        return {
            "status": "trained",
            "rows": len(feat_df),
            "train_accuracy": round(train_accuracy, 4),
        }

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict direction for the next bar.
        Returns: {direction, up_prob, confidence, signals, method}
        """
        feat_df = build_features(df)
        feat_df = feat_df.dropna(subset=FEATURE_COLS)

        if feat_df.empty:
            return self._fallback_predict(df)

        last_row = feat_df[FEATURE_COLS].iloc[-1:].values

        # Rule-based signals (always available)
        rule_signals = self._rule_based_signals(feat_df.iloc[-1])

        if self.trained:
            X_scaled = self.scaler.transform(last_row)
            ml_up_prob = float(self.model.predict_proba(X_scaled)[0][1])
        else:
            ml_up_prob = None

        # Ensemble: average ML prob with rule-based prob
        rule_prob = rule_signals["rule_up_prob"]
        if ml_up_prob is not None:
            ensemble_prob = 0.6 * ml_up_prob + 0.4 * rule_prob
            method = "ensemble_ml+rules"
        else:
            ensemble_prob = rule_prob
            method = "rules_only"

        direction = "UP" if ensemble_prob >= 0.5 else "DOWN"
        confidence = abs(ensemble_prob - 0.5) * 2  # 0=random, 1=certain

        return {
            "direction": direction,
            "up_prob": round(ensemble_prob, 4),
            "down_prob": round(1 - ensemble_prob, 4),
            "confidence": round(confidence, 4),
            "method": method,
            "ml_up_prob": round(ml_up_prob, 4) if ml_up_prob is not None else None,
            "rule_up_prob": round(rule_prob, 4),
            "signals": rule_signals["signals"],
        }

    def _rule_based_signals(self, row: pd.Series) -> dict:
        """Fast rule-based directional signals."""
        signals = {}
        votes = []

        # RSI signal
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

        # MACD histogram signal
        hist = row.get("macd_hist", 0)
        if pd.notna(hist):
            signals["macd"] = "BULLISH" if hist > 0 else "BEARISH"
            votes.append(0.65 if hist > 0 else 0.35)

        # Bollinger Band signal
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

        # Short-term momentum
        mom = row.get("mom_5", 0)
        if pd.notna(mom):
            signals["momentum"] = "POSITIVE" if mom > 0 else "NEGATIVE"
            votes.append(0.55 if mom > 0 else 0.45)

        # Volume signal
        vol_ratio = row.get("vol_ratio", 1)
        if pd.notna(vol_ratio) and vol_ratio > 1.5:
            body = row.get("body", 0)
            signals["volume"] = "HIGH_VOLUME_" + ("UP" if body > 0 else "DOWN")
            votes.append(0.65 if body > 0 else 0.35)

        rule_up_prob = float(np.mean(votes)) if votes else 0.5
        return {"rule_up_prob": rule_up_prob, "signals": signals}

    def _fallback_predict(self, df: pd.DataFrame) -> dict:
        """Last-resort: simple momentum fallback."""
        if len(df) < 2:
            return {"direction": "UP", "up_prob": 0.5, "down_prob": 0.5,
                    "confidence": 0.0, "method": "random", "signals": {}}
        last_return = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
        up_prob = 0.55 if last_return > 0 else 0.45
        return {
            "direction": "UP" if up_prob > 0.5 else "DOWN",
            "up_prob": round(up_prob, 4),
            "down_prob": round(1 - up_prob, 4),
            "confidence": 0.1,
            "method": "momentum_fallback",
            "signals": {"last_return": last_return},
        }
