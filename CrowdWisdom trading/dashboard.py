"""
dashboard.py
────────────────────────────────────────────────────────────────────
CrowdWisdomTrading — Streamlit Live Dashboard

Run: streamlit run dashboard.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.config import ASSETS, DATA_DIR
from utils.state_store import compute_accuracy, get_bankroll, load_predictions

st.set_page_config(
    page_title="CrowdWisdomTrading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .big-metric { font-size: 2rem; font-weight: bold; }
  .up   { color: #00c851; }
  .down { color: #ff4444; }
  .neutral { color: #aaaaaa; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("⚙️ CrowdWisdomTrading")
st.sidebar.caption("Crypto Prediction Pipeline Dashboard")

auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    import time
    st.sidebar.info("Auto-refreshing every 30 seconds")

selected_assets = st.sidebar.multiselect(
    "Assets",
    options=["BTC", "ETH", "SOL", "DOGE"],
    default=ASSETS,
)

if st.sidebar.button("▶️ Run Pipeline Now"):
    with st.spinner("Running pipeline..."):
        try:
            from run_pipeline import run_once
            result = run_once(assets=selected_assets)
            st.sidebar.success("Pipeline complete!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Pipeline error: {e}")


# ── Title ─────────────────────────────────────────────────────────
st.title("📈 CrowdWisdomTrading — Prediction Dashboard")
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Bankroll + Accuracy Row ────────────────────────────────────────
st.subheader("💰 Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)

bankroll = get_bankroll()
all_acc = compute_accuracy()

with col1:
    st.metric("Bankroll", f"${bankroll:.2f}")

with col2:
    total = all_acc.get("total", 0)
    acc = all_acc.get("accuracy", 0)
    st.metric("Overall Accuracy", f"{acc:.1%}" if total >= 5 else "N/A", f"{total} resolved")

with col3:
    btc_acc = compute_accuracy("BTC")
    st.metric("BTC Accuracy", f"{btc_acc.get('accuracy', 0):.1%}" if btc_acc.get("total", 0) >= 3 else "N/A")

with col4:
    eth_acc = compute_accuracy("ETH")
    st.metric("ETH Accuracy", f"{eth_acc.get('accuracy', 0):.1%}" if eth_acc.get("total", 0) >= 3 else "N/A")


# ── Apify Usage Section ────────────────────────────────────────────
st.subheader("🧾 Apify Usage & Cost Monitoring")

def get_apify_usage():
    """Fetch live Apify usage stats via API."""
    try:
        from apify_client import ApifyClient
        token = os.getenv("APIFY_API_TOKEN")
        if not token or token == "apify_api_your-token-here":
            return None, "No valid APIFY_API_TOKEN found in environment."
        client = ApifyClient(token)
        user = client.user().get()
        usage = user.get("usage", {})
        plan = user.get("plan", {})
        return {
            "monthly_usage_usd": usage.get("monthlyUsageUsd", 0),
            "monthly_usage_credits": usage.get("monthlyUsageCredits", 0),
            "compute_units_used": usage.get("monthlyComputeUnits", 0),
            "plan_id": plan.get("id", "free"),
            "monthly_limit_usd": plan.get("monthlyUsageCreditsUsd", 5.0),
            "username": user.get("username", "unknown"),
        }, None
    except Exception as e:
        return None, str(e)


# Load run count from pipeline log for call estimation
runs_path = DATA_DIR / "pipeline_runs.jsonl"
apify_call_count = 0
apify_fallback_count = 0

if runs_path.exists():
    with open(runs_path) as f:
        all_run_lines = f.readlines()
    apify_call_count = len(all_run_lines)

# Try live usage from Apify API
apify_usage, apify_err = get_apify_usage()

col_a1, col_a2, col_a3, col_a4 = st.columns(4)

with col_a1:
    if apify_usage:
        st.metric(
            "Apify Credits Used",
            f"${apify_usage['monthly_usage_usd']:.4f}",
            f"of ${apify_usage['monthly_limit_usd']:.2f} limit"
        )
    else:
        est_credits = apify_call_count * 0.015  # ~$0.015 per actor run
        st.metric("Est. Apify Credits Used", f"~${est_credits:.3f}", f"{apify_call_count} pipeline runs")

with col_a2:
    if apify_usage:
        st.metric("Compute Units", f"{apify_usage['compute_units_used']:.2f} CU")
    else:
        est_cu = apify_call_count * 0.1  # ~0.1 CU per run
        st.metric("Est. Compute Units", f"~{est_cu:.1f} CU")

with col_a3:
    st.metric("Total Pipeline Runs", apify_call_count, "Apify calls made")

with col_a4:
    if apify_usage:
        st.metric("Plan", apify_usage["plan_id"].upper(), f"@{apify_usage['username']}")
    else:
        st.metric("Fallback Mode", "Binance API", "Free, no tokens")

# Detailed breakdown
with st.expander("📊 Apify Usage Details & Cost Breakdown", expanded=False):
    if apify_usage:
        st.success("✅ Live Apify usage data retrieved successfully")
        st.json(apify_usage)
    elif apify_err:
        st.warning(f"⚠️ Could not fetch live usage: {apify_err}")

    st.markdown("""
    **How Apify is used in this project:**
    - **Agent 2** calls the `dtrungtin/binance-ohlcv-scraper` Apify actor to fetch 1,000 × 5-min OHLCV bars per asset
    - Each pipeline run = 1 Apify actor execution per asset (e.g. 2 assets = 2 actor calls)
    - If Apify fails or quota is exceeded, the system **automatically falls back** to the free Binance public REST API
    
    **Cost Estimates (Apify Free Tier = $5 credits):**
    """)

    # Dynamic cost table
    n_runs = max(apify_call_count, 1)
    n_assets = len(ASSETS)
    total_actor_calls = n_runs * n_assets
    est_cu_total = total_actor_calls * 0.1
    est_cost_total = total_actor_calls * 0.015
    pct_of_free_tier = min(est_cost_total / 5.0 * 100, 100)

    cost_df = pd.DataFrame([
        {"Metric": "Pipeline runs", "Value": str(n_runs)},
        {"Metric": "Assets tracked", "Value": str(n_assets)},
        {"Metric": "Total Apify actor calls", "Value": str(total_actor_calls)},
        {"Metric": "Estimated compute units", "Value": f"~{est_cu_total:.1f} CU"},
        {"Metric": "Estimated cost", "Value": f"~${est_cost_total:.3f}"},
        {"Metric": "Free tier ($5) consumed", "Value": f"~{pct_of_free_tier:.1f}%"},
        {"Metric": "Binance fallback calls (free)", "Value": str(apify_fallback_count)},
    ])
    st.dataframe(cost_df, use_container_width=True, hide_index=True)

    st.info(
        "💡 **Cost optimization:** The Binance REST API fallback is free and "
        "activates automatically when Apify is unavailable, making this system "
        "resilient and cost-efficient for production use."
    )


# ── Latest Pipeline Run ────────────────────────────────────────────
st.subheader("🔮 Latest Predictions")

latest_run = None
if runs_path.exists():
    with open(runs_path) as f:
        lines = f.readlines()
    if lines:
        try:
            latest_run = json.loads(lines[-1])
        except Exception:
            pass

if latest_run:
    run_time = latest_run.get("run_time", "unknown")
    st.caption(f"Last run: {run_time}")

    pred_cols = st.columns(len(selected_assets))
    for i, asset in enumerate(selected_assets):
        pred = latest_run.get("predictions", {}).get(asset, {})
        pos = latest_run.get("positions", {}).get(asset, {})

        with pred_cols[i]:
            direction = pred.get("direction", "?")
            up_prob = pred.get("up_prob", 0.5)
            conf = pred.get("confidence", 0)
            action = pos.get("action", "?")
            bet = pos.get("bet_usd", 0)

            color = "up" if direction == "UP" else "down"
            arrow = "⬆️" if direction == "UP" else "⬇️"

            st.markdown(f"### {asset}")
            st.markdown(f'<span class="{color} big-metric">{arrow} {direction}</span>', unsafe_allow_html=True)
            st.metric("P(UP)", f"{up_prob:.3f}", delta=f"conf {conf:.3f}")
            if action == "BET":
                st.success(f"BET ${bet:.2f} on {pos.get('bet_on','?')}")
            else:
                st.info(f"SKIP — {pos.get('reason','')[:40]}")
else:
    st.info("No pipeline runs yet. Click '▶️ Run Pipeline Now' to start.")


# ── Arbitrage Section ──────────────────────────────────────────────
st.subheader("⚡ Arbitrage Opportunities")
arb_path = DATA_DIR / "arbitrage_opportunities.jsonl"
if arb_path.exists():
    with open(arb_path) as f:
        arb_ops = [json.loads(l) for l in f.readlines()[-20:]]
    if arb_ops:
        arb_df = pd.DataFrame(arb_ops)
        st.dataframe(
            arb_df[["detected_at", "asset", "direction", "kalshi_15min_prob",
                     "poly_chain_15min", "gap", "description"]],
            use_container_width=True,
        )
    else:
        st.info("No arbitrage opportunities logged yet.")
else:
    st.info("No arbitrage log found.")


# ── Prediction History ─────────────────────────────────────────────
st.subheader("📋 Prediction History")

all_preds = []
for asset in selected_assets:
    preds = load_predictions(asset=asset, limit=100)
    all_preds.extend(preds)

if all_preds:
    df = pd.DataFrame(all_preds)
    if "correct" in df.columns:
        df["result"] = df.apply(
            lambda r: "✅" if r.get("correct") else ("❌" if r.get("resolved") else "⏳"), axis=1
        )

    display_cols = [c for c in ["timestamp", "asset", "direction", "up_prob", "confidence",
                                  "current_price", "method", "resolved", "result"] if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values("timestamp", ascending=False).head(50),
        use_container_width=True,
    )
else:
    st.info("No predictions recorded yet.")


# ── OHLCV Chart ────────────────────────────────────────────────────
st.subheader("📊 Price Charts")

chart_asset = st.selectbox("Select asset for chart", selected_assets)
ohlcv_path = DATA_DIR / f"ohlcv_{chart_asset.lower()}.csv"
ohlcv_parquet = DATA_DIR / f"ohlcv_{chart_asset.lower()}.parquet"

ohlcv_df = None
if ohlcv_parquet.exists():
    try:
        ohlcv_df = pd.read_parquet(ohlcv_parquet)
    except Exception:
        pass
if ohlcv_df is None and ohlcv_path.exists():
    ohlcv_df = pd.read_csv(ohlcv_path)

if ohlcv_df is not None and not ohlcv_df.empty:
    if "open_time" in ohlcv_df.columns:
        ohlcv_df["open_time"] = pd.to_datetime(ohlcv_df["open_time"])
        ohlcv_df = ohlcv_df.set_index("open_time")

    chart_df = ohlcv_df.tail(200)

    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name=chart_asset,
        )])
        fig.update_layout(
            title=f"{chart_asset} — Last 200 × 5-min Candles",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            height=400,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(chart_df["close"])
else:
    st.info(f"No OHLCV data for {chart_asset}. Run the pipeline to fetch data.")


# ── Pipeline Run Log ───────────────────────────────────────────────
st.subheader("📝 Pipeline Run Log")

if runs_path.exists():
    with open(runs_path) as f:
        run_lines = f.readlines()

    runs = []
    for line in run_lines[-30:]:
        try:
            r = json.loads(line)
            runs.append({
                "run_time": r.get("run_time", ""),
                "assets": ", ".join(r.get("assets", [])),
                "duration_s": r.get("duration_seconds", ""),
                "ohlcv_bars": str(r.get("ohlcv_bars", "")),
            })
        except Exception:
            pass

    if runs:
        st.dataframe(pd.DataFrame(runs[::-1]), use_container_width=True)
else:
    st.info("No run log found.")

# ── Auto-refresh ───────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
