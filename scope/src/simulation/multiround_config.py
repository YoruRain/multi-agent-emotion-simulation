from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agent_loader import PROJECT_ROOT

DEFAULT_MULTIROUND_OUTPUT_DIR = PROJECT_ROOT / "scope" / "data" / "outputs" / "simulation" / "multiround"


@dataclass(frozen=True)
class MultiRoundSimulationConfig:
    event_id: str
    output_dir: Path = field(default_factory=lambda: DEFAULT_MULTIROUND_OUTPUT_DIR)
    max_agents: int | None = None
    memory_user_level: str | None = None
    rounds: int = 5
    active_agent_limit: int | None = None
    use_llm: bool = False
    max_llm_agents_per_round: int | None = None
    seed: int = 42
    overwrite: bool = False
    resume: bool = True
    dry_run: bool = False
    enable_interactions: bool = False
    enable_emotion_dynamics: bool = False
    interaction_mode: str = "none"

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be >= 0.")
        if self.max_agents is not None and self.max_agents < 1:
            raise ValueError("max_agents must be >= 1 when provided.")
        if self.active_agent_limit is not None and self.active_agent_limit < 1:
            raise ValueError("active_agent_limit must be >= 1 when provided.")
        if self.max_llm_agents_per_round is not None and self.max_llm_agents_per_round < 0:
            raise ValueError("max_llm_agents_per_round must be >= 0 when provided.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload
