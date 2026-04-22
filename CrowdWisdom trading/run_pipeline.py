"""
run_pipeline.py
────────────────────────────────────────────────────────────────────
CrowdWisdomTrading — Main Pipeline Orchestrator

Uses Hermes-style AIAgent loop pattern (OpenRouter + free model).
Coordinates all 5 agents in sequence, with optional continuous loop.

Usage:
  python run_pipeline.py                  # single run, BTC + ETH
  python run_pipeline.py --loop           # every 5 min
  python run_pipeline.py --asset BTC      # single asset
  python run_pipeline.py --assets BTC,ETH,SOL   # multiple assets
  python run_pipeline.py --retrain        # force model retrain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from agents.agent1_market_scanner import format_scan_summary, scan_prediction_markets
from agents.agent2_data_fetcher_apify_tracking import fetch_all_assets
from agents.agent3_predictor import predict_all_assets
from agents.agent4_risk_manager import size_all_positions
from agents.agent5_feedback_loop import run_feedback_loop
from utils.config import ASSETS, DATA_DIR, LOOP_INTERVAL, OPENROUTER_API_KEY, OPENROUTER_MODEL
from utils.logger import get_logger
from utils.state_store import get_bankroll

logger = get_logger("pipeline")


# ── Hermes-style LLM Synthesis ────────────────────────────────────

def synthesize_with_llm(pipeline_result: dict) -> str:
    """
    Send pipeline results to OpenRouter (Hermes-style) for a natural language
    synthesis and trading commentary.
    Falls back gracefully if API key is missing.
    """
    if not OPENROUTER_API_KEY:
        logger.debug("No OPENROUTER_API_KEY — skipping LLM synthesis")
        return _format_simple_summary(pipeline_result)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Build a compact summary for the prompt
        assets_data = {}
        for asset in ASSETS:
            pred = pipeline_result.get("predictions", {}).get(asset, {})
            sizing = pipeline_result.get("positions", {}).get(asset, {})
            market = pipeline_result.get("market_scan", {}).get(asset, {})
            arb_ops = [
                op for op in pipeline_result.get("feedback", {}).get("arbitrage_opportunities", [])
                if op.get("asset") == asset
            ]
            assets_data[asset] = {
                "model_direction": pred.get("direction"),
                "model_up_prob": pred.get("up_prob"),
                "model_confidence": pred.get("confidence"),
                "market_consensus_prob": market.get("consensus_up_prob"),
                "kelly_action": sizing.get("action"),
                "bet_usd": sizing.get("bet_usd"),
                "edge": sizing.get("edge"),
                "arbitrage": arb_ops[0].get("description") if arb_ops else None,
            }

        stats = pipeline_result.get("feedback", {}).get("accuracy_stats", {})
        bankroll = stats.get("bankroll", get_bankroll())

        prompt = f"""You are a crypto prediction market analyst. Summarize this pipeline result in 3-4 concise sentences.
Focus on: key signals, recommended actions, and any arbitrage. Be specific with numbers.

Pipeline Data:
{json.dumps(assets_data, indent=2)}

Rolling Accuracy: {stats.get('ALL', {}).get('accuracy', 'N/A')}
Bankroll: ${bankroll:.2f}

Provide: a brief market commentary, key trading signals, and one risk note."""

        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        synthesis = response.choices[0].message.content.strip()
        logger.info("LLM synthesis complete")
        return synthesis

    except Exception as e:
        logger.warning(f"LLM synthesis failed: {e}")
        return _format_simple_summary(pipeline_result)


def _format_simple_summary(pipeline_result: dict) -> str:
    """Plain-text summary without LLM."""
    lines = ["📊 PIPELINE SUMMARY"]
    for asset in ASSETS:
        pred = pipeline_result.get("predictions", {}).get(asset, {})
        sizing = pipeline_result.get("positions", {}).get(asset, {})
        direction = pred.get("direction", "?")
        up_prob = pred.get("up_prob", 0.5)
        action = sizing.get("action", "?")
        bet = sizing.get("bet_usd", 0)
        lines.append(
            f"  {asset}: {direction} (P={up_prob:.3f}) → {action}"
            + (f" ${bet:.2f}" if action == "BET" else "")
        )
    return "\n".join(lines)


# ── Single Pipeline Run ───────────────────────────────────────────

def run_once(assets: list[str] = None, retrain: bool = False) -> dict:
    """Execute the full 5-agent pipeline once."""
    assets = assets or ASSETS
    run_start = datetime.now(timezone.utc)

    print("\n" + "═" * 65)
    print(f"  🚀 CrowdWisdomTrading Pipeline  —  {run_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Assets: {', '.join(assets)}")
    print("═" * 65)

    result = {"run_time": run_start.isoformat(), "assets": assets}

    # ── Agent 1: Market Scanner ────────────────────────────────────
    print("\n[1/5] 🔍 Scanning prediction markets (Polymarket + Kalshi)...")
    try:
        market_scan = scan_prediction_markets(assets)
        result["market_scan"] = market_scan
        print(format_scan_summary(market_scan))
    except Exception as e:
        logger.error(f"Agent 1 failed: {e}")
        traceback.print_exc()
        result["market_scan"] = {}

    # ── Agent 2: Data Fetcher ──────────────────────────────────────
    print("\n[2/5] 📦 Fetching OHLCV data (Apify / Binance fallback)...")
    try:
        ohlcv_data = fetch_all_assets(assets=assets, save=True)
        result["ohlcv_bars"] = {a: len(df) for a, df in ohlcv_data.items()}
        for asset, df in ohlcv_data.items():
            if not df.empty:
                print(f"  ✅ {asset}: {len(df)} bars, latest close = {df['close'].iloc[-1]:.4f}")
    except Exception as e:
        logger.error(f"Agent 2 failed: {e}")
        traceback.print_exc()
        ohlcv_data = {}
        result["ohlcv_bars"] = {}

    # ── Agent 3: Predictor ─────────────────────────────────────────
    print("\n[3/5] 🔮 Running Kronos prediction model...")
    try:
        predictions = predict_all_assets(assets=assets, ohlcv_data=ohlcv_data, retrain=retrain)
        result["predictions"] = predictions
        for asset in assets:
            p = predictions.get(asset, {})
            print(
                f"  {asset}: {p.get('direction','?')} "
                f"(P(up)={p.get('up_prob',0.5):.3f}, "
                f"conf={p.get('confidence',0):.3f}, "
                f"method={p.get('method','?')})"
            )
    except Exception as e:
        logger.error(f"Agent 3 failed: {e}")
        traceback.print_exc()
        predictions = {}
        result["predictions"] = {}

    # ── Agent 4: Risk Manager ──────────────────────────────────────
    print("\n[4/5] 💰 Kelly Criterion position sizing...")
    try:
        positions = size_all_positions(
            predictions=predictions,
            market_scan=result["market_scan"],
        )
        result["positions"] = positions
        bankroll = positions.get("bankroll", get_bankroll())
        print(f"  Bankroll: ${bankroll:.2f}")
        for asset in assets:
            pos = positions.get(asset, {})
            action = pos.get("action", "?")
            if action == "BET":
                print(
                    f"  ✅ {asset}: BET ${pos.get('bet_usd',0):.2f} "
                    f"({pos.get('bet_pct',0):.2%} bankroll) "
                    f"on {pos.get('bet_on','?')} — edge={pos.get('edge',0):.4f}"
                )
            else:
                print(f"  ⏭️  {asset}: SKIP — {pos.get('reason','')}")
        print(
            f"  Total risk: ${positions.get('total_risk_usd',0):.2f} "
            f"({positions.get('total_risk_pct',0):.2%} bankroll)"
        )
    except Exception as e:
        logger.error(f"Agent 4 failed: {e}")
        traceback.print_exc()
        result["positions"] = {}

    # ── Agent 5: Feedback Loop ─────────────────────────────────────
    print("\n[5/5] 📈 Running feedback loop (scoring + arbitrage)...")
    try:
        feedback = run_feedback_loop(assets=assets)
        result["feedback"] = feedback
        arb_count = len(feedback.get("arbitrage_opportunities", []))
        if arb_count:
            print(f"  ⚡ {arb_count} arbitrage opportunity(ies) detected!")
    except Exception as e:
        logger.error(f"Agent 5 failed: {e}")
        traceback.print_exc()
        result["feedback"] = {}

    # ── LLM Synthesis ─────────────────────────────────────────────
    print("\n[LLM] 🤖 Synthesizing with AI commentary...")
    synthesis = synthesize_with_llm(result)
    result["synthesis"] = synthesis
    print(f"\n  {synthesis}\n")

    # ── Save full result ───────────────────────────────────────────
    run_end = datetime.now(timezone.utc)
    result["duration_seconds"] = round((run_end - run_start).total_seconds(), 1)
    _save_run_result(result)

    print("═" * 65)
    print(f"  ⏱  Pipeline completed in {result['duration_seconds']}s")
    print("═" * 65 + "\n")

    return result


def _save_run_result(result: dict) -> None:
    """Append pipeline run to JSONL log."""
    log_path = DATA_DIR / "pipeline_runs.jsonl"
    # Trim heavy fields before saving
    slim = {
        k: v for k, v in result.items()
        if k not in ("market_scan",)
    }
    # Remove raw polymarket/kalshi lists from predictions
    if "predictions" in slim:
        slim["predictions"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("signals",)}
            for k, v in slim["predictions"].items()
            if isinstance(v, dict)
        }
    with open(log_path, "a") as f:
        f.write(json.dumps(slim, default=str) + "\n")
    logger.debug(f"Saved run to {log_path}")


# ── Continuous Loop ───────────────────────────────────────────────

def run_loop(assets: list[str] = None, interval: int = None, retrain_every: int = 12) -> None:
    """
    Run the pipeline on a schedule.
    retrain_every: retrain model every N iterations
    """
    interval = interval or LOOP_INTERVAL
    assets = assets or ASSETS
    iteration = 0

    logger.info(f"Starting loop: assets={assets}, interval={interval}s")
    print(f"\n🔁 Loop mode: running every {interval}s. Ctrl+C to stop.\n")

    while True:
        iteration += 1
        retrain = (iteration % retrain_every == 1)  # Retrain on first and every N runs

        try:
            run_once(assets=assets, retrain=retrain)
        except KeyboardInterrupt:
            print("\n\n  ⚠️  Loop stopped by user.")
            break
        except Exception as e:
            logger.error(f"Pipeline run {iteration} failed: {e}")
            traceback.print_exc()

        print(f"\n  ⏳ Next run in {interval}s... (Ctrl+C to stop)\n")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  ⚠️  Loop stopped by user.")
            break


# ── CLI ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CrowdWisdomTrading Crypto Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                   # single run, default assets
  python run_pipeline.py --loop            # continuous every 5 min
  python run_pipeline.py --asset BTC       # BTC only
  python run_pipeline.py --assets BTC,ETH,SOL
  python run_pipeline.py --retrain         # force model retrain
  python run_pipeline.py --loop --interval 60  # loop every 60s
        """,
    )
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--asset", type=str, help="Single asset (e.g. BTC)")
    parser.add_argument("--assets", type=str, help="Comma-separated assets (e.g. BTC,ETH,SOL)")
    parser.add_argument("--retrain", action="store_true", help="Force Kronos model retrain")
    parser.add_argument("--interval", type=int, help="Loop interval in seconds (default: 300)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine asset list
    if args.asset:
        assets = [args.asset.upper()]
    elif args.assets:
        assets = [a.strip().upper() for a in args.assets.split(",")]
    else:
        assets = ASSETS

    if args.loop:
        run_loop(assets=assets, interval=args.interval, retrain_every=12)
    else:
        run_once(assets=assets, retrain=args.retrain)
