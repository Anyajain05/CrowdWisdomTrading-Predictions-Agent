"""
agents/agent5_feedback_loop.py
────────────────────────────────────────────────────────────────────
Agent 5: Hermes feedback loop — scores past predictions, detects
15-min vs 3×5-min arbitrage, updates bankroll P&L simulation.

Arbitrage logic:
  - Kalshi 15-min market has an implied P(up) for next 15 min
  - Polymarket has 3 consecutive 5-min markets
  - P_implied_15min ≈ P_5min_1 × P_5min_2 × P_5min_3 (independent, correlated)
  - If |kalshi_15min_prob - poly_chain_prob| > threshold → arbitrage signal
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from agents.agent2_data_fetcher_apify_tracking import get_latest_price, load_ohlcv
from tools.kalshi_tools import get_15min_markets
from tools.polymarket_tools import get_crypto_5min_markets
from utils.config import ASSETS, DATA_DIR
from utils.logger import get_agent_logger
from utils.state_store import (
    compute_accuracy,
    get_bankroll,
    load_predictions,
    mark_resolved,
    save_prediction,
    update_bankroll,
)

logger = get_agent_logger("agent5_feedback")

ARB_THRESHOLD = 0.08   # 8% gap triggers arbitrage alert


# ── Prediction Scoring ────────────────────────────────────────────

def resolve_past_predictions(assets: list[str] = None) -> dict:
    """
    For each unresolved prediction, check if we now know the outcome.
    Uses current price vs price at prediction time to determine direction.

    Returns: {resolved_count, correct_count, accuracy}
    """
    assets = assets or ASSETS
    resolved_count = 0
    correct_count = 0

    for asset in assets:
        current_price = get_latest_price(asset)
        if current_price is None:
            df = load_ohlcv(asset)
            if df is not None and not df.empty:
                current_price = float(df["close"].iloc[-1])
        if current_price is None:
            logger.warning(f"  Cannot resolve {asset}: no current price")
            continue

        preds = load_predictions(asset=asset, limit=50)
        unresolved = [p for p in preds if not p.get("resolved") and p.get("current_price")]

        for pred in unresolved:
            old_price = pred.get("current_price")
            if old_price is None:
                continue
            # Actual direction
            actual = "UP" if current_price > old_price else "DOWN"
            mark_resolved(pred["id"], actual)

            was_correct = pred["direction"] == actual
            resolved_count += 1
            if was_correct:
                correct_count += 1

            # Simulate P&L
            bet_usd = pred.get("bet_usd", 0)
            if bet_usd and bet_usd > 0:
                market_prob = pred.get("market_prob", 0.5)
                payout_odds = (1.0 / market_prob - 1.0) if market_prob > 0.01 else 1.0
                pnl = bet_usd * payout_odds if was_correct else -bet_usd
                update_bankroll(
                    pnl,
                    reason=f"{asset} {pred['direction']} {'WIN' if was_correct else 'LOSS'}"
                )

            logger.info(
                f"  ✔ Resolved {asset} {pred['direction']} → actual={actual} "
                f"{'✅' if was_correct else '❌'} "
                f"(pred_price={old_price:.2f}, now={current_price:.2f})"
            )

    accuracy = correct_count / resolved_count if resolved_count else None
    result = {
        "resolved": resolved_count,
        "correct": correct_count,
        "accuracy": round(accuracy, 4) if accuracy else None,
    }
    logger.info(f"  Resolved {resolved_count} predictions, accuracy={accuracy}")
    return result


# ── Arbitrage Detection ───────────────────────────────────────────

def detect_arbitrage(assets: list[str] = None) -> list[dict]:
    """
    Detect arbitrage between:
      - Kalshi 15-min implied P(up)
      - 3× chained Polymarket 5-min implied P(up)

    Chain logic: if 5-min P(up) events are i.i.d. with probability p,
    then 15-min P(at least 2 of 3 up) ≈ 3p²(1-p) + p³
    But we use a simpler heuristic: poly_chain_prob = poly_5min_prob (avg)
    and compare directly with kalshi 15-min.

    Returns: list of arbitrage opportunities
    """
    assets = assets or ASSETS
    opportunities = []

    for asset in assets:
        try:
            # Kalshi 15-min probability
            kalshi_15min = get_15min_markets(asset)
            if not kalshi_15min:
                logger.debug(f"  No Kalshi 15-min markets for {asset}")
                kalshi_prob = None
            else:
                kalshi_prob = sum(m["implied_up_prob"] for m in kalshi_15min) / len(kalshi_15min)

            # Polymarket 5-min probability (proxy for next 15 min)
            poly_5min = get_crypto_5min_markets(asset)
            if not poly_5min:
                poly_prob = None
            else:
                poly_prob = sum(m["implied_up_prob"] for m in poly_5min) / len(poly_5min)

            if kalshi_prob is None or poly_prob is None:
                logger.debug(f"  Missing data for {asset} arbitrage check")
                continue

            # 3-period chain probability (simplified):
            # P(majority up in 3 periods) = 3p²(1-p) + p³
            p = poly_prob
            poly_chain_15min = 3 * p**2 * (1 - p) + p**3

            gap = kalshi_prob - poly_chain_15min
            abs_gap = abs(gap)

            logger.info(
                f"  {asset} arbitrage check: "
                f"kalshi_15min={kalshi_prob:.3f}, "
                f"poly_chain_15min={poly_chain_15min:.3f}, "
                f"gap={gap:+.3f}"
            )

            if abs_gap >= ARB_THRESHOLD:
                direction = "LONG_KALSHI" if gap > 0 else "LONG_POLYMARKET"
                opp = {
                    "asset": asset,
                    "type": "15min_vs_5min_chain",
                    "kalshi_15min_prob": round(kalshi_prob, 4),
                    "poly_5min_prob": round(poly_prob, 4),
                    "poly_chain_15min": round(poly_chain_15min, 4),
                    "gap": round(gap, 4),
                    "abs_gap": round(abs_gap, 4),
                    "direction": direction,
                    "description": (
                        f"Kalshi 15-min prob ({kalshi_prob:.3f}) "
                        f"{'>' if gap>0 else '<'} "
                        f"Poly 3×5-min chain ({poly_chain_15min:.3f}) "
                        f"by {abs_gap:.3f} → {direction}"
                    ),
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                }
                opportunities.append(opp)
                logger.info(f"  ⚡ ARBITRAGE: {opp['description']}")

        except Exception as e:
            logger.warning(f"  Arbitrage check failed for {asset}: {e}")

    return opportunities


# ── Rolling Accuracy Stats ────────────────────────────────────────

def compute_rolling_stats() -> dict:
    """Compute rolling accuracy + P&L across all assets."""
    stats = {
        "ALL": compute_accuracy(),
        "bankroll": round(get_bankroll(), 2),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    for asset in ASSETS:
        stats[asset] = compute_accuracy(asset)
    return stats


# ── Full Feedback Run ─────────────────────────────────────────────

def run_feedback_loop(assets: list[str] = None) -> dict:
    """
    Run the full feedback loop:
    1. Resolve past predictions
    2. Compute accuracy stats
    3. Detect arbitrage
    4. Save results

    Returns a summary dict.
    """
    assets = assets or ASSETS
    logger.info("Starting feedback loop...")

    # 1. Resolve
    resolution = resolve_past_predictions(assets)

    # 2. Stats
    stats = compute_rolling_stats()

    # 3. Arbitrage
    arb_ops = detect_arbitrage(assets)

    # 4. Save arbitrage to disk
    if arb_ops:
        arb_path = DATA_DIR / "arbitrage_opportunities.jsonl"
        with open(arb_path, "a") as f:
            for op in arb_ops:
                f.write(json.dumps(op) + "\n")
        logger.info(f"  Saved {len(arb_ops)} arbitrage opportunities to {arb_path}")

    result = {
        "resolution": resolution,
        "accuracy_stats": stats,
        "arbitrage_opportunities": arb_ops,
        "loop_ran_at": datetime.now(timezone.utc).isoformat(),
    }

    _print_feedback_summary(result)
    return result


def _print_feedback_summary(result: dict) -> None:
    stats = result.get("accuracy_stats", {})
    resolution = result.get("resolution", {})
    arb = result.get("arbitrage_opportunities", [])

    lines = ["", "─" * 60, "  📈 FEEDBACK LOOP SUMMARY", "─" * 60]
    lines.append(f"  Resolved predictions : {resolution.get('resolved', 0)}")
    lines.append(f"  Correct this run     : {resolution.get('correct', 0)}")

    all_stats = stats.get("ALL", {})
    if all_stats.get("total", 0) >= 5:
        lines.append(f"  Rolling accuracy     : {all_stats['accuracy']:.1%} ({all_stats['total']} resolved)")
    else:
        lines.append(f"  Rolling accuracy     : N/A (need ≥5 resolved)")

    lines.append(f"  Bankroll             : ${stats.get('bankroll', 0):.2f}")

    for asset in ASSETS:
        a = stats.get(asset, {})
        if a.get("total", 0) > 0:
            lines.append(f"  {asset} accuracy       : {a['accuracy']:.1%} ({a['total']} resolved)")

    if arb:
        lines.append(f"\n  ⚡ ARBITRAGE OPPORTUNITIES: {len(arb)}")
        for op in arb:
            lines.append(f"    • {op['description']}")
    else:
        lines.append("\n  No arbitrage opportunities detected.")

    lines.append("─" * 60)
    print("\n".join(lines))


if __name__ == "__main__":
    print("=" * 60)
    print("  📈 AGENT 5 — FEEDBACK LOOP")
    print("=" * 60)
    run_feedback_loop()
