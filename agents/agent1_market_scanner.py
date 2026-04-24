"""
agents/agent1_market_scanner.py
────────────────────────────────────────────────────────────────────
Agent 1: Scans Polymarket and Kalshi for crypto 5-min predictions.

Outputs aggregated implied probabilities for BTC/ETH next-5-min
UP/DOWN moves from prediction markets.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from cw_tools.kalshi_tools import get_crypto_5min_markets as kalshi_5min
from cw_tools.polymarket_tools import get_crypto_5min_markets as poly_5min
from cw_utils.config import ASSETS
from cw_utils.logger import get_agent_logger

logger = get_agent_logger("agent1_scanner")


def scan_prediction_markets(assets: list[str] = None) -> dict:
    """
    Scan both Polymarket and Kalshi for 5-min crypto prediction markets.
    
    Returns:
        {
          "BTC": {
            "polymarket": [...markets...],
            "kalshi": [...markets...],
            "consensus_up_prob": 0.58,
            "market_signal": "UP",
            "confidence": 0.16,
            "data_sources": 2,
          },
          "ETH": { ... },
          "scan_time": "2026-04-22T10:00:00Z"
        }
    """
    assets = assets or ASSETS
    result = {"scan_time": datetime.now(timezone.utc).isoformat()}

    for asset in assets:
        logger.info(f"Scanning prediction markets for {asset}...")
        asset_data = {}

        # ── Polymarket ─────────────────────────────────────────────
        try:
            poly_markets = poly_5min(asset)
            asset_data["polymarket"] = poly_markets
            poly_probs = [m["implied_up_prob"] for m in poly_markets if m.get("liquidity", 0) >= 0]
            asset_data["poly_avg_prob"] = round(sum(poly_probs) / len(poly_probs), 4) if poly_probs else 0.5
            logger.info(f"  Polymarket {asset}: {len(poly_markets)} markets, avg_up={asset_data['poly_avg_prob']:.3f}")
        except Exception as e:
            logger.warning(f"  Polymarket scan failed for {asset}: {e}")
            asset_data["polymarket"] = []
            asset_data["poly_avg_prob"] = 0.5

        # ── Kalshi ─────────────────────────────────────────────────
        try:
            kalshi_markets = kalshi_5min(asset)
            asset_data["kalshi"] = kalshi_markets
            kalshi_probs = [m["implied_up_prob"] for m in kalshi_markets]
            asset_data["kalshi_avg_prob"] = round(sum(kalshi_probs) / len(kalshi_probs), 4) if kalshi_probs else 0.5
            logger.info(f"  Kalshi {asset}: {len(kalshi_markets)} markets, avg_up={asset_data['kalshi_avg_prob']:.3f}")
        except Exception as e:
            logger.warning(f"  Kalshi scan failed for {asset}: {e}")
            asset_data["kalshi"] = []
            asset_data["kalshi_avg_prob"] = 0.5

        # ── Consensus probability ──────────────────────────────────
        probs = [asset_data["poly_avg_prob"], asset_data["kalshi_avg_prob"]]
        active_probs = [p for p in probs if p != 0.5]

        if active_probs:
            # Weight: Polymarket has higher liquidity generally
            poly_w = 0.6 if asset_data["polymarket"] else 0.0
            kalshi_w = 0.4 if asset_data["kalshi"] else 0.0
            total_w = poly_w + kalshi_w
            if total_w > 0:
                consensus = (
                    poly_w * asset_data["poly_avg_prob"] +
                    kalshi_w * asset_data["kalshi_avg_prob"]
                ) / total_w
            else:
                consensus = 0.5
        else:
            consensus = 0.5

        consensus = round(consensus, 4)
        confidence = round(abs(consensus - 0.5) * 2, 4)
        signal = "UP" if consensus > 0.5 else "DOWN"

        asset_data.update({
            "consensus_up_prob": consensus,
            "market_signal": signal,
            "confidence": confidence,
            "data_sources": int(bool(asset_data["polymarket"])) + int(bool(asset_data["kalshi"])),
        })

        result[asset] = asset_data
        logger.info(
            f"  ✅ {asset} consensus: {signal} (prob={consensus:.3f}, conf={confidence:.3f})"
        )

    return result


def format_scan_summary(scan_result: dict) -> str:
    """Pretty print the scan result."""
    lines = ["", "═" * 60, "  📊 PREDICTION MARKET SCAN RESULTS", "═" * 60]
    for asset in ASSETS:
        if asset not in scan_result:
            continue
        d = scan_result[asset]
        signal = d.get("market_signal", "?")
        prob = d.get("consensus_up_prob", 0.5)
        conf = d.get("confidence", 0)
        sources = d.get("data_sources", 0)
        arrow = "⬆️ " if signal == "UP" else "⬇️ "
        lines.append(
            f"  {arrow}{asset:6s} │ Signal: {signal:4s} │ "
            f"P(up)={prob:.3f} │ Conf={conf:.3f} │ Sources: {sources}"
        )
    lines.append(f"  ⏰ Scanned at: {scan_result.get('scan_time', 'N/A')}")
    lines.append("═" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    result = scan_prediction_markets()
    print(format_scan_summary(result))
    print("\nRaw JSON (first asset):")
    for asset in ASSETS:
        if asset in result:
            d = dict(result[asset])
            d.pop("polymarket", None)
            d.pop("kalshi", None)
            print(f"\n{asset}:", json.dumps(d, indent=2))
            break
