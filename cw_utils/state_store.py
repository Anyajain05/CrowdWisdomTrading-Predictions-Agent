from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

BANKROLL_PATH = DATA_DIR / "bankroll_state.json"
PREDICTIONS_PATH = DATA_DIR / "prediction_history.jsonl"
LEDGER_PATH = DATA_DIR / "bankroll_ledger.jsonl"

DEFAULT_BANKROLL = float(os.getenv("STARTING_BANKROLL", 1000.0))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _ensure_bankroll_state() -> dict:
    state = _read_json(BANKROLL_PATH, {})
    if not state:
        state = {
            "bankroll": round(DEFAULT_BANKROLL, 2),
            "starting_bankroll": round(DEFAULT_BANKROLL, 2),
            "updated_at": _utc_now(),
        }
        _write_json(BANKROLL_PATH, state)
    return state


def get_bankroll() -> float:
    state = _ensure_bankroll_state()
    return float(state.get("bankroll", DEFAULT_BANKROLL))


def update_bankroll(amount, reason=None):
    state = _ensure_bankroll_state()
    current = float(state.get("bankroll", DEFAULT_BANKROLL))
    new_value = round(current + float(amount), 2)
    state["bankroll"] = new_value
    state["updated_at"] = _utc_now()
    _write_json(BANKROLL_PATH, state)

    ledger_entry = {
        "timestamp": _utc_now(),
        "delta": round(float(amount), 2),
        "bankroll_after": new_value,
        "reason": reason or "",
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger_entry) + "\n")
    return new_value


def compute_accuracy(asset=None):
    rows = load_predictions(asset=asset, limit=None)
    resolved = [row for row in rows if row.get("resolved")]
    total = len(resolved)
    correct = sum(1 for row in resolved if row.get("was_correct") is True)
    accuracy = (correct / total) if total else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
    }


def load_predictions(asset, limit=50):
    rows = _read_jsonl(PREDICTIONS_PATH)
    asset = asset.upper() if asset else None
    if asset:
        rows = [row for row in rows if row.get("asset") == asset]
    rows = sorted(rows, key=lambda row: row.get("timestamp", ""), reverse=True)
    if limit is None:
        return rows
    return rows[:limit]


def mark_resolved(pred_id, actual):
    rows = _read_jsonl(PREDICTIONS_PATH)
    changed = False
    for row in rows:
        if row.get("id") != pred_id:
            continue
        row["resolved"] = True
        row["actual_direction"] = actual
        row["resolved_at"] = _utc_now()
        row["was_correct"] = row.get("direction") == actual
        changed = True
        break
    if changed:
        _write_jsonl(PREDICTIONS_PATH, rows)


def save_prediction(asset, pred):
    row = {
        "id": pred.get("id"),
        "asset": asset.upper(),
        "direction": pred.get("direction"),
        "up_prob": pred.get("up_prob"),
        "confidence": pred.get("confidence"),
        "current_price": pred.get("current_price"),
        "method": pred.get("method"),
        "backend": pred.get("backend"),
        "forecast_close": pred.get("forecast_close"),
        "forecast_delta_pct": pred.get("forecast_delta_pct"),
        "market_prob": pred.get("market_prob"),
        "bet_usd": pred.get("bet_usd", 0.0),
        "bet_pct": pred.get("bet_pct", 0.0),
        "bet_on": pred.get("bet_on"),
        "action": pred.get("action"),
        "timestamp": pred.get("timestamp", _utc_now()),
        "resolved": bool(pred.get("resolved", False)),
        "actual_direction": pred.get("actual_direction"),
        "resolved_at": pred.get("resolved_at"),
        "was_correct": pred.get("was_correct"),
    }
    with open(PREDICTIONS_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def update_prediction(pred_id: str, updates: dict[str, Any]) -> None:
    rows = _read_jsonl(PREDICTIONS_PATH)
    changed = False
    for row in rows:
        if row.get("id") != pred_id:
            continue
        row.update(updates)
        changed = True
        break
    if changed:
        _write_jsonl(PREDICTIONS_PATH, rows)
