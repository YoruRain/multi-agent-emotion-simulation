from __future__ import annotations

from dataclasses import replace
from typing import Any

from .agent_state import AgentState, clamp
from .interaction_schema import InteractionRecord
from .multiround_config import MultiRoundSimulationConfig


REACTION_EMOTION_SCORES = {
    "anger": -0.85,
    "disgust": -0.80,
    "fear": -0.70,
    "sadness": -0.70,
    "disappointment": -0.65,
    "confusion": -0.25,
    "mixed": 0.00,
    "surprise": 0.00,
    "sympathy": 0.35,
    "admiration": 0.70,
    "joy": 0.75,
}


def normalize_emotion_label(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"", "unknown", "none", "null", "nan"}:
        return "mixed"
    legacy_labels = {
        "positive": "joy",
        "neutral": "mixed",
        "negative": "disappointment",
    }
    normalized = legacy_labels.get(normalized, normalized)
    return normalized if normalized in REACTION_EMOTION_SCORES else "mixed"


def emotion_label_to_score(emotion_label: str | None, intensity: int | float = 1) -> float:
    try:
        intensity_value = float(intensity)
    except (TypeError, ValueError):
        intensity_value = 1.0
    if intensity_value <= 0:
        return 0.0
    base_score = REACTION_EMOTION_SCORES.get(normalize_emotion_label(emotion_label), 0.0)
    if intensity_value >= 2:
        base_score *= 1.15
    return round(clamp(base_score, -1.0, 1.0), 4)


def stance_label_to_score(stance_label: str | None, intensity: int | float = 1) -> float:
    try:
        intensity_value = float(intensity)
    except (TypeError, ValueError):
        intensity_value = 1.0
    if intensity_value <= 0:
        return 0.0

    label = str(stance_label or "").strip().lower()
    if label in {"favor", "support"}:
        return 0.85 if intensity_value >= 2 else 0.50
    if label in {"against", "oppose", "opposition"}:
        return -0.85 if intensity_value >= 2 else -0.50
    return 0.0


def score_to_emotion_state_label(score: float) -> str:
    if score >= 0.25:
        return "positive"
    if score <= -0.25:
        return "negative"
    return "neutral"


def score_to_stance_state_label(score: float) -> str:
    if score >= 0.25:
        return "support"
    if score <= -0.25:
        return "against"
    return "neutral"


def build_event_influence_scores(event: dict[str, Any]) -> dict[str, Any]:
    emotion_label = normalize_emotion_label(event.get("dominant_emotion_label"))
    emotion_score = emotion_label_to_score(emotion_label, 1)
    emotion_reason = f"事件主导情绪为{emotion_label}"

    stance_label = str(event.get("dominant_stance_label") or "").strip().lower()
    stance_reason = "事件主导立场缺失"
    if stance_label:
        stance_score = stance_label_to_score(stance_label, 1)
        stance_reason = f"事件主导立场为{stance_label}"
    else:
        focus = str(event.get("event_stance_focus") or "").strip().lower()
        if any(token in focus for token in ("support", "favor", "支持", "认可", "正向")):
            stance_score = 0.30
            stance_reason = "根据事件立场焦点弱判为支持"
        elif any(token in focus for token in ("against", "oppose", "反对", "质疑", "负向")):
            stance_score = -0.30
            stance_reason = "根据事件立场焦点弱判为反对"
        else:
            stance_score = 0.0

    return {
        "event_emotion_score": round(clamp(emotion_score, -1.0, 1.0), 4),
        "event_stance_score": round(clamp(stance_score, -1.0, 1.0), 4),
        "event_emotion_reason": emotion_reason,
        "event_stance_reason": stance_reason,
    }


def aggregate_neighbor_influence(
    target_agent_id: str,
    interactions_for_round: list[InteractionRecord | dict[str, Any]],
    state_by_agent: dict[str, AgentState],
) -> dict[str, Any]:
    weighted_emotion_sum = 0.0
    weighted_stance_sum = 0.0
    weight_sum = 0.0
    neighbor_ids: set[str] = set()
    high_influence_count = 0
    media_count = 0
    kol_count = 0

    for interaction in interactions_for_round:
        target_id = _interaction_value(interaction, "target_agent_id")
        if str(target_id) != str(target_agent_id):
            continue
        source_agent_id = str(_interaction_value(interaction, "source_agent_id") or "")
        source_state = state_by_agent.get(source_agent_id)
        if source_state is None:
            continue
        weight = clamp(_interaction_value(interaction, "weight", 0.0), 0.0, 1.0)
        if weight <= 0:
            continue

        neighbor_ids.add(source_agent_id)
        weighted_emotion_sum += source_state.emotion_score * weight
        weighted_stance_sum += source_state.stance_score * weight
        weight_sum += weight
        if source_state.influence_score >= 0.75:
            high_influence_count += 1
        verified = source_state.verified_type_name or ""
        if any(token in verified for token in ("媒体", "政府", "机构")):
            media_count += 1
        role = source_state.propagation_role or ""
        if any(token in role for token in ("KOL", "潜在影响者", "高影响力")):
            kol_count += 1

    if weight_sum <= 0:
        return {
            "neighbor_emotion_score": 0.0,
            "neighbor_stance_score": 0.0,
            "neighbor_influence_weight_sum": 0.0,
            "neighbor_count": 0,
            "high_influence_neighbor_count": 0,
            "media_neighbor_count": 0,
            "kol_neighbor_count": 0,
        }

    return {
        "neighbor_emotion_score": round(weighted_emotion_sum / weight_sum, 4),
        "neighbor_stance_score": round(weighted_stance_sum / weight_sum, 4),
        "neighbor_influence_weight_sum": round(weight_sum, 4),
        "neighbor_count": len(neighbor_ids),
        "high_influence_neighbor_count": high_influence_count,
        "media_neighbor_count": media_count,
        "kol_neighbor_count": kol_count,
    }


def update_agent_state_with_dynamics(
    old_state: AgentState,
    own_reaction: dict[str, Any] | None,
    neighbor_influence: dict[str, Any],
    event_influence: dict[str, Any],
    config: MultiRoundSimulationConfig,
) -> AgentState:
    old_emotion = old_state.emotion_score
    old_stance = old_state.stance_score
    own_emotion_score = _reaction_emotion_score(own_reaction)
    own_stance_score = _reaction_stance_score(own_reaction)
    susceptibility_score = old_state.susceptibility_score

    raw_emotion = (
        config.self_retention * old_emotion
        + config.social_influence_strength * susceptibility_score * _as_float(neighbor_influence.get("neighbor_emotion_score"))
        + config.event_influence_strength * _as_float(event_influence.get("event_emotion_score"))
        + config.reaction_influence_strength * own_emotion_score
    )
    raw_stance = (
        config.stance_retention * old_stance
        + config.social_stance_strength * susceptibility_score * _as_float(neighbor_influence.get("neighbor_stance_score"))
        + config.event_stance_strength * _as_float(event_influence.get("event_stance_score"))
        + config.reaction_stance_strength * own_stance_score
    )

    if config.enable_saturation_damping:
        new_emotion = apply_saturation_damping(old_emotion, raw_emotion, config.saturation_damping_strength)
        new_stance = apply_saturation_damping(old_stance, raw_stance, config.saturation_damping_strength)
    else:
        new_emotion = clamp(raw_emotion, -1.0, 1.0)
        new_stance = clamp(raw_stance, -1.0, 1.0)

    new_emotion = round(clamp(new_emotion, -1.0, 1.0), 4)
    new_stance = round(clamp(new_stance, -1.0, 1.0), 4)
    new_state = replace(
        old_state,
        emotion_score=new_emotion,
        stance_score=new_stance,
        emotion_label=score_to_emotion_state_label(new_emotion),
        stance_label=score_to_stance_state_label(new_stance),
        old_emotion_score=round(old_emotion, 4),
        new_emotion_score=new_emotion,
        emotion_delta=round(new_emotion - old_emotion, 4),
        old_stance_score=round(old_stance, 4),
        new_stance_score=new_stance,
        stance_delta=round(new_stance - old_stance, 4),
        neighbor_emotion_score=_as_float(neighbor_influence.get("neighbor_emotion_score")),
        neighbor_stance_score=_as_float(neighbor_influence.get("neighbor_stance_score")),
        neighbor_influence_weight_sum=_as_float(neighbor_influence.get("neighbor_influence_weight_sum")),
        neighbor_count=int(neighbor_influence.get("neighbor_count", 0) or 0),
        high_influence_neighbor_count=int(neighbor_influence.get("high_influence_neighbor_count", 0) or 0),
        media_neighbor_count=int(neighbor_influence.get("media_neighbor_count", 0) or 0),
        kol_neighbor_count=int(neighbor_influence.get("kol_neighbor_count", 0) or 0),
        event_emotion_score=_as_float(event_influence.get("event_emotion_score")),
        event_stance_score=_as_float(event_influence.get("event_stance_score")),
        own_reaction_emotion_score=own_emotion_score,
        own_reaction_stance_score=own_stance_score,
        dynamics_enabled=True,
    )
    return replace(
        new_state,
        state_update_reason=build_dynamics_update_reason(
            old_state,
            new_state,
            neighbor_influence,
            event_influence,
            own_reaction,
            config.min_delta_threshold_for_reason,
        ),
    )


def apply_saturation_damping(old_score: float, raw_new_score: float, damping_strength: float = 0.5) -> float:
    old_score = clamp(old_score, -1.0, 1.0)
    raw_new_score = clamp(raw_new_score, -1.0, 1.0)
    delta = raw_new_score - old_score
    damping_factor = 1 - clamp(damping_strength, 0.0, 1.0) * abs(old_score)
    return clamp(old_score + delta * damping_factor, -1.0, 1.0)


def build_dynamics_update_reason(
    old_state: AgentState,
    new_state: AgentState,
    neighbor_influence: dict[str, Any],
    event_influence: dict[str, Any],
    own_reaction: dict[str, Any] | None,
    min_delta_threshold: float = 0.03,
) -> str:
    emotion_delta = new_state.emotion_score - old_state.emotion_score
    stance_delta = new_state.stance_score - old_state.stance_score
    neighbor_count = int(neighbor_influence.get("neighbor_count", 0) or 0)

    if abs(emotion_delta) < min_delta_threshold and abs(stance_delta) < min_delta_threshold:
        return "本轮状态变化较小，整体保持稳定"
    if neighbor_count > 0 and emotion_delta < -0.05:
        return "受评论区负向观点影响，情绪分数下降"
    if neighbor_count > 0 and emotion_delta > 0.05:
        return "受较积极的上下文评论影响，情绪分数上升"
    if stance_delta < -0.05:
        return "受邻近评论和自身表达影响，立场更偏反对"
    if stance_delta > 0.05:
        return "受邻近评论和自身表达影响，立场更偏支持"
    if neighbor_count == 0 and own_reaction:
        return "本轮主要根据自身表达更新状态"
    if neighbor_count == 0 and not own_reaction:
        return "本轮未参与互动，状态主要保持稳定"
    if _as_float(event_influence.get("event_emotion_score")) or _as_float(event_influence.get("event_stance_score")):
        return "受事件刺激与互动影响，状态轻微变化"
    return "受互动上下文影响，状态轻微变化"


def _reaction_emotion_score(own_reaction: dict[str, Any] | None) -> float:
    if not own_reaction:
        return 0.0
    return emotion_label_to_score(own_reaction.get("emotion_label"), own_reaction.get("emotion_intensity", 1))


def _reaction_stance_score(own_reaction: dict[str, Any] | None) -> float:
    if not own_reaction:
        return 0.0
    return stance_label_to_score(own_reaction.get("stance_label"), own_reaction.get("stance_intensity", 1))


def _interaction_value(interaction: InteractionRecord | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(interaction, InteractionRecord):
        return getattr(interaction, key, default)
    return interaction.get(key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return default
