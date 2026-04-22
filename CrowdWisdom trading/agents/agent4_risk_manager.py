"""
agents/agent4_risk_manager.py
────────────────────────────────────────────────────────────────────
Agent 4: Kelly Criterion position sizing + risk management.

Full Kelly:   f* = (p*b - q) / b
              where p = win probability, q = 1-p, b = net odds (payout - 1)

Fractional Kelly (safer):  f_adj = KELLY_FRACTION * f*

References:
  https://mintlify.wiki/joicodev/polymarket-bot/risk/kelly-criterion
  https://managebankroll.com/blog/polymarket-kelly-criterion-position-sizing
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from utils.config import KELLY_FRACTION, MAX_BET_PCT, STARTING_BANKROLL
from utils.logger import get_agent_logger
from utils.state_store import get_bankroll, update_bankroll

logger = get_agent_logger("agent4_risk")


def kelly_fraction(
    win_prob: float,
    payout_odds: float,
    kelly_multiplier: float = None,
) -> dict:
    """
    Compute Kelly Criterion fraction.

    Args:
        win_prob:       Probability of winning (0 < p < 1)
        payout_odds:    Net payout per unit bet (e.g., 1.0 = even money)
                        On prediction markets: (1/price - 1)
        kelly_multiplier: Override config KELLY_FRACTION (default 0.25)

    Returns:
        {
          "full_kelly": 0.12,
          "fractional_kelly": 0.03,
          "edge": 0.05,
          "recommended": 0.03,    # as fraction of bankroll
          "rationale": "...",
        }
    """
    km = kelly_multiplier if kelly_multiplier is not None else KELLY_FRACTION

    p = min(max(win_prob, 0.001), 0.999)
    q = 1 - p
    b = max(payout_odds, 0.01)

    # Core Kelly formula
    full_kelly = (p * b - q) / b

    # Edge = expected value per unit
    edge = p * b - q

    if full_kelly <= 0 or edge <= 0:
        return {
            "full_kelly": 0.0,
            "fractional_kelly": 0.0,
            "edge": round(edge, 4),
            "recommended": 0.0,
            "rationale": f"No edge (edge={edge:.4f}, full_kelly={full_kelly:.4f}) — skip bet",
        }

    fractional = full_kelly * km
    # Hard cap at MAX_BET_PCT of bankroll
    recommended = min(fractional, MAX_BET_PCT)

    rationale = (
        f"Edge={edge:.4f}, Full Kelly={full_kelly:.4f}, "
        f"Fractional ({km}x)={fractional:.4f}, "
        f"Capped={recommended:.4f} (max {MAX_BET_PCT:.0%})"
    )

    return {
        "full_kelly": round(full_kelly, 4),
        "fractional_kelly": round(fractional, 4),
        "edge": round(edge, 4),
        "recommended": round(recommended, 4),
        "rationale": rationale,
    }


def size_position(
    asset: str,
    direction: str,
    up_prob: float,
    market_yes_price: float,
    bankroll: Optional[float] = None,
    min_edge: float = 0.03,
) -> dict:
    """
    Full position sizing recommendation for one asset/direction.

    Args:
        asset:            e.g. "BTC"
        direction:        "UP" or "DOWN"
        up_prob:          Model's predicted probability of UP move
        market_yes_price: Current YES price on prediction market (0-1)
        bankroll:         Current bankroll (USD); loaded from store if None
        min_edge:         Minimum required edge before betting

    Returns:
        {
          "asset": "BTC",
          "direction": "UP",
          "bet_on": "YES",
          "model_prob": 0.63,
          "market_prob": 0.55,
          "edge": 0.08,
          "kelly": {...},
          "bankroll": 1000.0,
          "bet_usd": 12.50,
          "bet_pct": 0.0125,
          "action": "BET" | "SKIP",
          "reason": "...",
        }
    """
    if bankroll is None:
        bankroll = get_bankroll()

    # Decide which side to bet
    if direction == "UP":
        bet_on = "YES"
        model_prob = up_prob
        market_prob = market_yes_price
    else:
        bet_on = "NO"
        model_prob = 1 - up_prob
        market_prob = 1 - market_yes_price

    # Payout odds: if you pay market_prob, you win (1 - market_prob) per unit
    if market_prob <= 0.01 or market_prob >= 0.99:
        return _skip_result(asset, direction, bankroll, "Market price extreme (near 0 or 1)")

    payout_odds = (1.0 / market_prob) - 1.0

    # Kelly sizing
    kelly = kelly_fraction(model_prob, payout_odds)
    edge = kelly["edge"]

    # Edge gate
    if edge < min_edge:
        return _skip_result(
            asset, direction, bankroll,
            f"Edge too small: {edge:.4f} < min {min_edge:.4f}"
        )

    if kelly["recommended"] <= 0:
        return _skip_result(asset, direction, bankroll, kelly["rationale"])

    bet_usd = round(bankroll * kelly["recommended"], 2)
    bet_usd = max(bet_usd, 0.0)

    result = {
        "asset": asset,
        "direction": direction,
        "bet_on": bet_on,
        "model_prob": round(model_prob, 4),
        "market_prob": round(market_prob, 4),
        "edge": round(edge, 4),
        "kelly": kelly,
        "bankroll": round(bankroll, 2),
        "bet_usd": bet_usd,
        "bet_pct": round(kelly["recommended"], 4),
        "action": "BET",
        "reason": kelly["rationale"],
        "sized_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"  💰 {asset} {direction}: "
        f"edge={edge:.4f}, kelly={kelly['recommended']:.4f}, "
        f"bet=${bet_usd:.2f} ({kelly['recommended']:.2%} bankroll)"
    )
    return result


def size_all_positions(
    predictions: dict,
    market_scan: dict,
    bankroll: Optional[float] = None,
) -> dict:
    """
    Size positions for all assets using prediction + market data.

    Args:
        predictions:  Output of agent3 predict_all_assets()
        market_scan:  Output of agent1 scan_prediction_markets()
        bankroll:     USD bankroll (loads from store if None)

    Returns:
        {
          "BTC": { ...position sizing... },
          "ETH": { ...position sizing... },
          "total_risk_usd": 25.0,
          "total_risk_pct": 0.025,
          "sized_at": "...",
        }
    """
    if bankroll is None:
        bankroll = get_bankroll()

    results = {
        "sized_at": datetime.now(timezone.utc).isoformat(),
        "bankroll": round(bankroll, 2),
    }
    total_risk = 0.0

    for asset, pred in predictions.items():
        if asset in ("predicted_at",):
            continue
        if not isinstance(pred, dict):
            continue

        up_prob = pred.get("up_prob", 0.5)
        direction = pred.get("direction", "UP")
        confidence = pred.get("confidence", 0)

        # Get market consensus price
        scan = market_scan.get(asset, {})
        market_yes_price = scan.get("consensus_up_prob", 0.5)

        # Skip low-confidence predictions
        if confidence < 0.05:
            results[asset] = _skip_result(
                asset, direction, bankroll,
                f"Confidence too low: {confidence:.4f}"
            )
            continue

        sizing = size_position(
            asset=asset,
            direction=direction,
            up_prob=up_prob,
            market_yes_price=market_yes_price,
            bankroll=bankroll,
        )
        results[asset] = sizing

        if sizing["action"] == "BET":
            total_risk += sizing["bet_usd"]

    results["total_risk_usd"] = round(total_risk, 2)
    results["total_risk_pct"] = round(total_risk / bankroll, 4) if bankroll > 0 else 0
    return results


def _skip_result(asset: str, direction: str, bankroll: float, reason: str) -> dict:
    logger.info(f"  ⏭️  {asset} {direction}: SKIP — {reason}")
    return {
        "asset": asset,
        "direction": direction,
        "action": "SKIP",
        "reason": reason,
        "bet_usd": 0.0,
        "bet_pct": 0.0,
        "edge": 0.0,
        "bankroll": round(bankroll, 2),
        "sized_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  💰 AGENT 4 — KELLY RISK MANAGER")
    print("=" * 60)

    # Demo: test Kelly formula
    test_cases = [
        {"label": "Strong UP edge",   "win_prob": 0.65, "odds": 0.82},
        {"label": "Slight UP edge",   "win_prob": 0.55, "odds": 0.82},
        {"label": "No edge (fair)",   "win_prob": 0.55, "odds": 0.818},
        {"label": "Wrong direction",  "win_prob": 0.40, "odds": 1.5},
    ]

    for tc in test_cases:
        k = kelly_fraction(tc["win_prob"], tc["odds"])
        print(f"\n  {tc['label']}:")
        print(f"    Win prob     : {tc['win_prob']:.2f}")
        print(f"    Net odds     : {tc['odds']:.3f}")
        print(f"    Edge         : {k['edge']:.4f}")
        print(f"    Full Kelly   : {k['full_kelly']:.4f}")
        print(f"    Frac Kelly   : {k['fractional_kelly']:.4f}")
        print(f"    Recommended  : {k['recommended']:.4f}  ({k['recommended']:.2%} of bankroll)")

    # Demo position sizing
    print("\n\n  Position Sizing Demo (bankroll=$1000):")
    pos = size_position("BTC", "UP", up_prob=0.64, market_yes_price=0.55, bankroll=1000.0)
    for k, v in pos.items():
        if k != "kelly":
            print(f"    {k:20s}: {v}")
