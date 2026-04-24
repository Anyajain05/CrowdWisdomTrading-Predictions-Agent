# CrowdWisdomTrading Predictions Agent

Python backend for the CrowdWisdomTrading internship assignment.

## Assignment coverage

This project now includes:

- Agent 1: search/scanner for BTC and ETH next-5-minute markets on Polymarket and Kalshi
- Agent 2: Apify-based OHLCV fetcher for the latest 1000 bars, with Binance fallback
- Agent 3: next-bar UP/DOWN prediction using the local Kronos repo from `third_party/Kronos-master`
- Agent 4: Kelly Criterion risk sizing
- Agent 5: feedback loop with persisted prediction history, bankroll updates, rolling accuracy, and 15-minute vs 3x 5-minute arbitrage checks
- User visibility via Streamlit dashboard
- Hermes Agent framework integration for orchestration/synthesis

## Real framework usage

This repo does not just say "Hermes-style" anymore. It includes a real Hermes Agent integration through [tools/hermes_tools.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/tools/hermes_tools.py), which imports `AIAgent` from the official Hermes Python library path documented by Nous Research.

Official Hermes Python library docs:
- [Using Hermes as a Python Library](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/guides/python-library.md)
- [Hermes Agent repo](https://github.com/nousresearch/hermes-agent)

Important note:
- Hermes Agent officially supports Linux, macOS, WSL2, and Termux
- Native Windows is not officially supported by Hermes, so on Windows the recommended runtime is WSL2

## Kronos integration

The Kronos predictor uses your local extracted archive at:

- [third_party/Kronos-master](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/third_party/Kronos-master)

Agent 3 loads:

- `KronosTokenizer`
- `Kronos`
- `KronosPredictor`

from the local Kronos repo and converts the next forecasted close into:

- `direction`
- `up_prob`
- `confidence`
- `forecast_close`
- `forecast_delta_pct`

## Project layout

- [run_pipeline.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/run_pipeline.py): main orchestrator
- [agents/agent1_market_scanner.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/agents/agent1_market_scanner.py): Polymarket + Kalshi market search
- [agents/agent2_data_fetcher_apify_tracking.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/agents/agent2_data_fetcher_apify_tracking.py): Apify/Binance OHLCV fetcher
- [agents/agent3_predictor.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/agents/agent3_predictor.py): Kronos prediction agent
- [agents/agent4_risk_manager.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/agents/agent4_risk_manager.py): Kelly sizing
- [agents/agent5_feedback_loop.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/agents/agent5_feedback_loop.py): feedback loop and arbitrage
- [tools/hermes_tools.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/tools/hermes_tools.py): Hermes Agent wrapper
- [tools/kronos_tools.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/tools/kronos_tools.py): Kronos backend integration
- [utils/state_store.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/utils/state_store.py): persistent bankroll and prediction history
- [dashboard.py](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/dashboard.py): Streamlit visibility layer

## Setup

### 1. Install project dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure env

Use [.env.example](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/.env.example) as the template.

Required:

- `OPENROUTER_API_KEY`
- `APIFY_API_TOKEN`

Recommended:

- `KRONOS_REPO_PATH=./third_party/Kronos-master`
- `HERMES_ENABLED=true`

### 3. Run

```bash
python run_pipeline.py
python run_pipeline.py --loop
streamlit run dashboard.py
```

## Notes for submission

- Do not commit `.env`; it is ignored by [.gitignore](C:/Users/itzan/Documents/Codex/2026-04-24-files-mentioned-by-the-user-crowdwisdom/.gitignore)
- Include your Apify token usage evidence from `data/apify_usage.jsonl`
- Record a short demo video showing the pipeline run and dashboard
- If running Hermes on Windows, show WSL2 or note that Hermes is wired in code and should be run under its supported environment

## Sources

- [Hermes Agent repo](https://github.com/nousresearch/hermes-agent)
- [Hermes Python library guide](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/guides/python-library.md)
- [Kronos repo](https://github.com/shiyu-coder/Kronos)
- [Apify Python client docs](https://docs.apify.com/api/client/python/docs/overview)
