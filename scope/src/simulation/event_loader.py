from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .agent_loader import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

DEFAULT_EVENTS_PATH = PROJECT_ROOT / "scope" / "data" / "inputs" / "events.jsonl"


def load_events(events_path: Path = DEFAULT_EVENTS_PATH) -> list[dict[str, Any]]:
    """Load event records from UTF-8 JSONL."""

    if not events_path.exists():
        raise FileNotFoundError(f"Events JSONL not found: {events_path}")

    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid event JSON at {events_path}:{line_number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Event record at {events_path}:{line_number} is not an object")
            events.append(event)

    LOGGER.info("Loaded %d events from %s", len(events), events_path)
    return events


def get_event_by_id(event_id: str, events_path: Path = DEFAULT_EVENTS_PATH) -> dict[str, Any]:
    """Return one event by event_id."""

    for event in load_events(events_path):
        if str(event.get("event_id", "")).strip() == event_id:
            return event
    raise ValueError(f"event_id not found: {event_id}")
