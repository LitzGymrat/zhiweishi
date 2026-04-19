from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
REAL_DATA_FILE = BASE_DIR / "data" / "real_data" / "evidence_points.json"


def _load_payload() -> dict[str, Any]:
    if not REAL_DATA_FILE.exists():
        return {"summary_cards": [], "records": []}
    return json.loads(REAL_DATA_FILE.read_text(encoding="utf-8"))


def get_real_data_summary_cards() -> list[dict[str, str]]:
    payload = _load_payload()
    cards = payload.get("summary_cards", [])
    return [card for card in cards if isinstance(card, dict) and card.get("label") and card.get("value")]


def get_real_data_records() -> list[dict[str, Any]]:
    payload = _load_payload()
    records = payload.get("records", [])
    return [record for record in records if isinstance(record, dict) and record.get("title")]