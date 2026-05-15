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
    llm_concurrency: int = 3
    model_name: str | None = None
    base_url: str | None = None
    seed: int = 42
    overwrite: bool = False
    resume: bool = True
    dry_run: bool = False
    enable_interactions: bool = False
    enable_emotion_dynamics: bool = False
    self_retention: float = 0.65
    social_influence_strength: float = 0.25
    event_influence_strength: float = 0.10
    reaction_influence_strength: float = 0.15
    stance_retention: float = 0.75
    social_stance_strength: float = 0.20
    event_stance_strength: float = 0.10
    reaction_stance_strength: float = 0.15
    enable_saturation_damping: bool = True
    saturation_damping_strength: float = 0.6
    min_delta_threshold_for_reason: float = 0.03
    interaction_mode: str = "none"
    kol_speaker_limit: int = 5
    top_k_context_comments: int = 3
    allow_previous_round_context: bool = False
    max_context_comment_length: int = 80

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be >= 0.")
        if self.max_agents is not None and self.max_agents < 1:
            raise ValueError("max_agents must be >= 1 when provided.")
        if self.active_agent_limit is not None and self.active_agent_limit < 1:
            raise ValueError("active_agent_limit must be >= 1 when provided.")
        if self.max_llm_agents_per_round is not None and self.max_llm_agents_per_round < 0:
            raise ValueError("max_llm_agents_per_round must be >= 0 when provided.")
        if self.llm_concurrency < 1:
            raise ValueError("llm_concurrency must be >= 1.")
        if self.kol_speaker_limit < 0:
            raise ValueError("kol_speaker_limit must be >= 0.")
        if self.top_k_context_comments < 0:
            raise ValueError("top_k_context_comments must be >= 0.")
        if self.max_context_comment_length < 1:
            raise ValueError("max_context_comment_length must be >= 1.")
        for name in (
            "self_retention",
            "social_influence_strength",
            "event_influence_strength",
            "reaction_influence_strength",
            "stance_retention",
            "social_stance_strength",
            "event_stance_strength",
            "reaction_stance_strength",
            "saturation_damping_strength",
            "min_delta_threshold_for_reason",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")
        if self.interaction_mode not in {"none", "kol_first"}:
            raise ValueError("interaction_mode must be 'none' or 'kol_first'.")
        if self.enable_interactions and self.interaction_mode == "none":
            object.__setattr__(self, "interaction_mode", "kol_first")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload
