from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


INTERACTION_FIELDNAMES = [
    "run_id",
    "event_id",
    "topic",
    "round_id",
    "source_agent_id",
    "target_agent_id",
    "source_user_id",
    "target_user_id",
    "interaction_type",
    "weight",
    "source_action_type",
    "source_reaction_text",
    "target_action_type",
    "target_reaction_text",
    "context_rank",
    "source_emotion_score",
    "target_emotion_score_before",
    "target_emotion_score_after",
    "source_stance_score",
    "target_stance_score_before",
    "target_stance_score_after",
    "source_influence_score",
    "target_susceptibility_score",
    "target_kol_sensitivity_score",
    "target_media_dependency_score",
    "source_verified_type_name",
    "source_propagation_role",
    "target_propagation_role",
    "reason",
    "source",
    "created_at",
]


def _short_text(value: Any, max_length: int = 120) -> str:
    text = "" if value is None else str(value).strip()
    return text[:max_length]


@dataclass
class InteractionRecord:
    run_id: str
    event_id: str
    topic: str | None
    round_id: int
    source_agent_id: str
    target_agent_id: str
    source_user_id: str | None
    target_user_id: str | None
    interaction_type: str
    weight: float
    source_action_type: str
    source_reaction_text: str
    target_action_type: str | None
    target_reaction_text: str | None
    context_rank: int | None
    source_emotion_score: float
    target_emotion_score_before: float
    target_emotion_score_after: float | None
    source_stance_score: float
    target_stance_score_before: float
    target_stance_score_after: float | None
    source_influence_score: float
    target_susceptibility_score: float
    target_kol_sensitivity_score: float
    target_media_dependency_score: float
    source_verified_type_name: str | None
    source_propagation_role: str | None
    target_propagation_role: str | None
    reason: str
    source: str
    created_at: str = ""

    def __post_init__(self) -> None:
        self.weight = round(max(0.01, min(1.0, float(self.weight))), 4)
        self.source_reaction_text = _short_text(self.source_reaction_text)
        self.target_reaction_text = _short_text(self.target_reaction_text)
        if not self.interaction_type:
            self.interaction_type = "influence_candidate"
        if self.created_at == "":
            self.created_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_interactions_csv(records: list[InteractionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INTERACTION_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
