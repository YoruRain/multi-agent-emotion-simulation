from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable

from .agent_state import AgentState, agent_state_to_dict

LOGGER = logging.getLogger(__name__)


def _state_value(state: AgentState | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(state, AgentState):
        return getattr(state, key, default)
    return state.get(key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _avg(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


def _stddev(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    mean = sum(items) / len(items)
    variance = sum((item - mean) ** 2 for item in items) / len(items)
    return round(variance ** 0.5, 4)


def _dominant_label(counts: dict[str, int], default: str) -> str:
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]


def compute_round_metrics(
    states: list[AgentState],
    round_id: int,
    interaction_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute aggregate state metrics for one simulation round."""

    total_agents = len(states)
    active_states = [state for state in states if _as_bool(_state_value(state, "is_active", False))]
    positive_count = sum(1 for state in states if _state_value(state, "emotion_label") == "positive")
    neutral_emotion_count = sum(1 for state in states if _state_value(state, "emotion_label") == "neutral")
    negative_count = sum(1 for state in states if _state_value(state, "emotion_label") == "negative")
    support_count = sum(1 for state in states if _state_value(state, "stance_label") == "support")
    neutral_stance_count = sum(1 for state in states if _state_value(state, "stance_label") == "neutral")
    oppose_count = sum(1 for state in states if _state_value(state, "stance_label") == "against")
    emotion_scores = [_as_float(_state_value(state, "emotion_score")) for state in states]
    stance_scores = [_as_float(_state_value(state, "stance_score")) for state in states]
    abs_emotion_deltas = [abs(_as_float(_state_value(state, "emotion_delta"))) for state in states]
    abs_stance_deltas = [abs(_as_float(_state_value(state, "stance_delta"))) for state in states]
    neighbor_counts = [_as_float(_state_value(state, "neighbor_count")) for state in states]
    neighbor_weight_sums = [_as_float(_state_value(state, "neighbor_influence_weight_sum")) for state in states]
    emotion_label_counts = {
        "positive": positive_count,
        "neutral": neutral_emotion_count,
        "negative": negative_count,
    }
    stance_label_counts = {
        "support": support_count,
        "neutral": neutral_stance_count,
        "against": oppose_count,
    }

    first = states[0] if states else {}
    summary = interaction_summary or {}
    metrics = {
        "run_id": _state_value(first, "run_id", ""),
        "event_id": _state_value(first, "event_id", ""),
        "topic": _state_value(first, "topic", ""),
        "round_id": round_id,
        "total_agents": total_agents,
        "active_agent_count": len(active_states),
        "participation_rate": _ratio(len(active_states), total_agents),
        "avg_emotion_score": _avg(_as_float(_state_value(state, "emotion_score")) for state in states),
        "avg_stance_score": _avg(_as_float(_state_value(state, "stance_score")) for state in states),
        "positive_count": positive_count,
        "neutral_emotion_count": neutral_emotion_count,
        "negative_count": negative_count,
        "positive_ratio": _ratio(positive_count, total_agents),
        "neutral_ratio": _ratio(neutral_emotion_count, total_agents),
        "negative_ratio": _ratio(negative_count, total_agents),
        "support_count": support_count,
        "neutral_stance_count": neutral_stance_count,
        "oppose_count": oppose_count,
        "support_ratio": _ratio(support_count, total_agents),
        "neutral_stance_ratio": _ratio(neutral_stance_count, total_agents),
        "oppose_ratio": _ratio(oppose_count, total_agents),
        "avg_influence_score_active": _avg(
            _as_float(_state_value(state, "influence_score")) for state in active_states
        ),
        "avg_activity_score_active": _avg(
            _as_float(_state_value(state, "activity_score")) for state in active_states
        ),
        "kol_speaker_count": int(summary.get("kol_speaker_count", 0) or 0),
        "regular_active_count": int(summary.get("regular_active_count", 0) or 0),
        "interaction_count": int(summary.get("interaction_count", 0) or 0),
        "avg_interaction_weight": round(_as_float(summary.get("avg_interaction_weight", 0.0)), 4),
        "high_influence_interaction_count": int(summary.get("high_influence_interaction_count", 0) or 0),
        "agents_with_context_count": int(summary.get("agents_with_context_count", 0) or 0),
        "avg_context_comment_count": round(_as_float(summary.get("avg_context_comment_count", 0.0)), 4),
        "emotion_volatility": _stddev(emotion_scores),
        "stance_volatility": _stddev(stance_scores),
        "avg_abs_emotion_delta": _avg(abs_emotion_deltas),
        "avg_abs_stance_delta": _avg(abs_stance_deltas),
        "max_abs_emotion_delta": round(max(abs_emotion_deltas), 4) if abs_emotion_deltas else 0.0,
        "max_abs_stance_delta": round(max(abs_stance_deltas), 4) if abs_stance_deltas else 0.0,
        "dominant_emotion_state": _dominant_label(emotion_label_counts, "neutral"),
        "dominant_stance_state": _dominant_label(stance_label_counts, "neutral"),
        "polarization_score": _stddev(stance_scores),
        "avg_neighbor_count": _avg(neighbor_counts),
        "agents_affected_by_neighbors": sum(1 for count in neighbor_counts if count > 0),
        "avg_neighbor_influence_weight": _avg(neighbor_weight_sums),
        "dynamics_enabled": any(_as_bool(_state_value(state, "dynamics_enabled", False)) for state in states),
    }
    return metrics


def save_round_metrics(metrics_list: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics_list[0].keys()) if metrics_list else [
        "run_id",
        "event_id",
        "topic",
        "round_id",
        "total_agents",
        "active_agent_count",
        "participation_rate",
        "avg_emotion_score",
        "avg_stance_score",
        "positive_count",
        "neutral_emotion_count",
        "negative_count",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "support_count",
        "neutral_stance_count",
        "oppose_count",
        "support_ratio",
        "neutral_stance_ratio",
        "oppose_ratio",
        "avg_influence_score_active",
        "avg_activity_score_active",
        "kol_speaker_count",
        "regular_active_count",
        "interaction_count",
        "avg_interaction_weight",
        "high_influence_interaction_count",
        "agents_with_context_count",
        "avg_context_comment_count",
        "emotion_volatility",
        "stance_volatility",
        "avg_abs_emotion_delta",
        "avg_abs_stance_delta",
        "max_abs_emotion_delta",
        "max_abs_stance_delta",
        "dominant_emotion_state",
        "dominant_stance_state",
        "polarization_score",
        "avg_neighbor_count",
        "agents_affected_by_neighbors",
        "avg_neighbor_influence_weight",
        "dynamics_enabled",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    LOGGER.info("Wrote round metrics to %s", output_path)


def summarize_run(output_dir: Path) -> dict[str, Any]:
    """Reload agent state CSV, recompute round metrics and return a compact summary."""

    states_path = output_dir / "agent_states_by_round.csv"
    if not states_path.exists():
        raise FileNotFoundError(f"agent_states_by_round.csv not found: {states_path}")

    rows_by_round: dict[int, list[dict[str, Any]]] = {}
    with states_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                round_id = int(row.get("round_id", 0))
            except (TypeError, ValueError):
                round_id = 0
            rows_by_round.setdefault(round_id, []).append(row)

    metrics = [compute_round_metrics(rows, round_id) for round_id, rows in sorted(rows_by_round.items())]  # type: ignore[arg-type]
    save_round_metrics(metrics, output_dir / "round_metrics.csv")
    final_metrics = metrics[-1] if metrics else {}
    return {
        "output_dir": str(output_dir),
        "round_count": len(metrics),
        "agent_count": int(final_metrics.get("total_agents", 0)) if final_metrics else 0,
        "final_round_id": int(final_metrics.get("round_id", 0)) if final_metrics else 0,
        "final_avg_emotion_score": final_metrics.get("avg_emotion_score", 0.0),
        "final_avg_stance_score": final_metrics.get("avg_stance_score", 0.0),
    }


def states_to_csv(states: list[AgentState], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [agent_state_to_dict(state) for state in states]
    fieldnames = list(rows[0].keys()) if rows else list(AgentState.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
