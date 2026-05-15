from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .agent_loader import AgentRecord

LOGGER = logging.getLogger(__name__)


def clamp(value: Any, min_value: float, max_value: float) -> float:
    """Convert value to float and clip it to the target interval."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = min_value
    return max(min_value, min(max_value, number))


def score_to_emotion_label(score: float) -> str:
    if score >= 0.25:
        return "positive"
    if score <= -0.25:
        return "negative"
    return "neutral"


def score_to_stance_label(score: float) -> str:
    if score >= 0.25:
        return "support"
    if score <= -0.25:
        return "against"
    return "neutral"


def agent_state_to_dict(state: "AgentState") -> dict[str, Any]:
    return asdict(state)


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _nested_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _optional_clamped_float(value: Any, min_value: float, max_value: float) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return clamp(value, min_value, max_value)
    except (TypeError, ValueError):
        return None


@dataclass
class AgentState:
    run_id: str
    event_id: str
    weibo_id: str | None
    topic: str | None
    agent_id: str
    user_id: str | None
    round_id: int
    memory_user_level: str | None
    verified_type_name: str | None
    propagation_role: str | None
    influence_level: str | None
    influence_score: float
    susceptibility_score: float
    activity_score: float
    kol_sensitivity_score: float
    media_dependency_score: float
    repost_tendency_score: float
    emotion_score: float
    stance_score: float
    emotion_label: str = ""
    stance_label: str = ""
    is_active: bool = False
    last_action_type: str = "ignore"
    last_reaction_text: str = ""
    last_reason: str = ""
    old_emotion_score: float | None = None
    new_emotion_score: float | None = None
    emotion_delta: float | None = None
    old_stance_score: float | None = None
    new_stance_score: float | None = None
    stance_delta: float | None = None
    neighbor_emotion_score: float | None = None
    neighbor_stance_score: float | None = None
    neighbor_influence_weight_sum: float | None = None
    neighbor_count: int | None = None
    high_influence_neighbor_count: int | None = None
    media_neighbor_count: int | None = None
    kol_neighbor_count: int | None = None
    event_emotion_score: float | None = None
    event_stance_score: float | None = None
    own_reaction_emotion_score: float | None = None
    own_reaction_stance_score: float | None = None
    dynamics_enabled: bool = False
    state_update_reason: str = "根据用户长期画像和事件基本倾向初始化状态"
    source: str = "initial_profile"
    created_at: str = ""

    def __post_init__(self) -> None:
        self.influence_score = clamp(self.influence_score, 0.0, 1.0)
        self.susceptibility_score = clamp(self.susceptibility_score, 0.0, 1.0)
        self.activity_score = clamp(self.activity_score, 0.0, 1.0)
        self.kol_sensitivity_score = clamp(self.kol_sensitivity_score, 0.0, 1.0)
        self.media_dependency_score = clamp(self.media_dependency_score, 0.0, 1.0)
        self.repost_tendency_score = clamp(self.repost_tendency_score, 0.0, 1.0)
        self.emotion_score = clamp(self.emotion_score, -1.0, 1.0)
        self.stance_score = clamp(self.stance_score, -1.0, 1.0)
        self.emotion_label = score_to_emotion_label(self.emotion_score)
        self.stance_label = score_to_stance_label(self.stance_score)
        if not self.last_action_type:
            self.last_action_type = "ignore"
        if self.created_at == "":
            self.created_at = datetime.now().isoformat(timespec="seconds")


def _estimate_susceptibility(behavior: dict[str, Any]) -> float:
    kol = clamp(behavior.get("kol_sensitivity_score", 0.0), 0.0, 1.0)
    media = clamp(behavior.get("media_dependency_score", 0.0), 0.0, 1.0)
    susceptibility_score = 0.35 + 0.35 * kol + 0.25 * media
    return clamp(susceptibility_score, 0.05, 0.95)


def _estimate_activity(memory_user_level: str, propagation_role: str) -> float:
    level_scores = {
        "core": 0.75,
        "normal": 0.55,
        "background": 0.35,
    }
    score = level_scores.get(memory_user_level.lower(), 0.5)
    if any(token in propagation_role for token in ["原创表达者", "高活跃", "转发评论者"]):
        score += 0.10
    if "低活跃观察者" in propagation_role:
        score -= 0.15
    return clamp(score, 0.05, 0.95)


def _estimate_initial_stance(behavior: dict[str, Any], event: dict[str, Any]) -> float:
    dominant_stance = _safe_text(event.get("dominant_stance_label")).lower()
    if dominant_stance in {"joy", "sympathy", "admiration"}:
        bias = 0.2
    elif dominant_stance in {"anger", "sadness", "fear", "disgust", "disappointment", "confusion"}:
        bias = -0.2
    else:
        return 0.0

    topic_ratio = clamp(
        behavior.get("public_issue_topic_ratio", behavior.get("final_public_issue_topic_ratio", 0.0)),
        0.0,
        1.0,
    )
    if _safe_text(event.get("event_type")) == "public_issue":
        bias += (0.1 * topic_ratio) if bias > 0 else (-0.1 * topic_ratio)
    return clamp(bias, -1.0, 1.0)


def build_initial_agent_state(agent_record: AgentRecord, event: dict[str, Any], run_id: str) -> AgentState:
    profile = agent_record.profile
    base_identity = profile.get("base_identity") if isinstance(profile.get("base_identity"), dict) else {}
    behavior = profile.get("behavior_parameters") if isinstance(profile.get("behavior_parameters"), dict) else {}
    if not behavior:
        LOGGER.warning("Agent %s has no behavior_parameters; using default state scores.", agent_record.agent_id)

    memory_user_level = _safe_text(base_identity.get("memory_user_level"), agent_record.memory_user_level)
    propagation_role = (
        _safe_text(base_identity.get("propagation_role"))
        or _safe_text(behavior.get("propagation_role"))
        or _safe_text(_nested_get(profile, "prompt_profile", "propagation_role"))
    )
    pos_ratio = clamp(behavior.get("pos_ratio", 0.0), 0.0, 1.0)
    neg_ratio = clamp(behavior.get("neg_ratio", 0.0), 0.0, 1.0)

    return AgentState(
        run_id=run_id,
        event_id=_safe_text(event.get("event_id")),
        weibo_id=_safe_text(event.get("weibo_id")) or None,
        topic=_safe_text(event.get("topic")) or None,
        agent_id=agent_record.agent_id,
        user_id=agent_record.user_id or None,
        round_id=0,
        memory_user_level=memory_user_level or None,
        verified_type_name=_safe_text(base_identity.get("verified_type_name")) or None,
        propagation_role=propagation_role or None,
        influence_level=_safe_text(base_identity.get("influence_level")) or None,
        influence_score=clamp(behavior.get("influence_score", 0.0), 0.0, 1.0),
        susceptibility_score=_estimate_susceptibility(behavior),
        activity_score=_estimate_activity(memory_user_level, propagation_role),
        kol_sensitivity_score=clamp(behavior.get("kol_sensitivity_score", 0.0), 0.0, 1.0),
        media_dependency_score=clamp(behavior.get("media_dependency_score", 0.0), 0.0, 1.0),
        repost_tendency_score=clamp(behavior.get("repost_ratio", behavior.get("repost_tendency_score", 0.0)), 0.0, 1.0),
        emotion_score=clamp(pos_ratio - neg_ratio, -1.0, 1.0),
        stance_score=_estimate_initial_stance(behavior, event),
        state_update_reason="根据用户长期画像和事件基本倾向初始化状态",
    )
