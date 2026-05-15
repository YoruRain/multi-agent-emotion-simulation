from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from .agent_state import AgentState, clamp
from .interaction_schema import InteractionRecord
from .multiround_config import MultiRoundSimulationConfig


ContextComment = dict[str, Any]


class InteractionEngine:
    """Select speakers, build visible comment contexts, and record candidate influence edges."""

    REPLY_HINTS = ("确实", "同意", "不是吧", "别急", "这说法", "我觉得", "前面", "有道理")

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def select_kol_speakers(
        self,
        agent_states: list[AgentState],
        config: MultiRoundSimulationConfig,
    ) -> list[AgentState]:
        candidates: list[tuple[float, AgentState]] = []
        for state in agent_states:
            score = self.speaker_score(state)
            participation_prob = clamp(0.15 + score, 0.05, 0.95)
            if self.rng.random() < participation_prob:
                candidates.append((score, state))

        candidates.sort(key=lambda item: (item[0], item[1].influence_score, item[1].agent_id), reverse=True)
        total_agents = len(agent_states)
        population_limit = max(1, int(total_agents * 0.3))
        limit = min(max(0, config.kol_speaker_limit), population_limit)
        if config.active_agent_limit is not None:
            limit = min(limit, config.active_agent_limit)
        return [state for _, state in candidates[:limit]]

    def select_regular_candidates(
        self,
        agent_states: list[AgentState],
        kol_speakers: list[AgentState],
        config: MultiRoundSimulationConfig,
    ) -> list[AgentState]:
        kol_ids = {state.agent_id for state in kol_speakers}
        selected: list[AgentState] = []
        for state in agent_states:
            if state.agent_id in kol_ids:
                continue
            probability = clamp(state.activity_score + 0.10 * state.influence_score, 0.05, 0.95)
            if self.rng.random() < probability:
                selected.append(state)

        selected.sort(key=lambda item: (item.activity_score + item.influence_score, item.agent_id), reverse=True)
        if config.active_agent_limit is not None:
            remaining = max(0, config.active_agent_limit - len(kol_speakers))
            selected = selected[:remaining]
        return selected

    def select_context_comments(
        self,
        target_agent: AgentState,
        candidate_comments: list[ContextComment],
        config: MultiRoundSimulationConfig,
    ) -> list[ContextComment]:
        scored: list[tuple[float, ContextComment]] = []
        for comment in candidate_comments:
            source_state = comment.get("source_state")
            if not isinstance(source_state, AgentState) or source_state.agent_id == target_agent.agent_id:
                continue
            score = (
                0.55 * source_state.influence_score
                + 0.25 * abs(source_state.emotion_score)
                + 0.20 * abs(source_state.stance_score)
            )
            score += 0.20 * target_agent.kol_sensitivity_score * source_state.influence_score
            verified = source_state.verified_type_name or ""
            if any(token in verified for token in ("媒体", "政府", "机构")):
                score += 0.15 * target_agent.media_dependency_score
            scored.append((score, comment))

        scored.sort(key=lambda item: (item[0], item[1].get("source_agent_id", "")), reverse=True)
        return [
            {**comment, "context_rank": index + 1}
            for index, (_, comment) in enumerate(scored[: max(0, config.top_k_context_comments)])
        ]

    def build_interaction_records(
        self,
        source_comments: list[ContextComment],
        target_agent_before: AgentState,
        target_agent_after: AgentState,
        target_reaction: dict[str, Any],
        config: MultiRoundSimulationConfig,
    ) -> list[InteractionRecord]:
        records: list[InteractionRecord] = []
        target_action = str(target_reaction.get("action_type", "comment"))
        target_text = str(target_reaction.get("reaction_text", ""))
        for comment in source_comments:
            source_state = comment.get("source_state")
            if not isinstance(source_state, AgentState):
                continue
            interaction_type = self._infer_interaction_type(target_action, target_text)
            weight = self.compute_interaction_weight(source_state, target_agent_before, interaction_type)
            records.append(
                InteractionRecord(
                    run_id=target_agent_before.run_id,
                    event_id=target_agent_before.event_id,
                    topic=target_agent_before.topic,
                    round_id=target_agent_after.round_id,
                    source_agent_id=source_state.agent_id,
                    target_agent_id=target_agent_before.agent_id,
                    source_user_id=source_state.user_id,
                    target_user_id=target_agent_before.user_id,
                    interaction_type=interaction_type,
                    weight=weight,
                    source_action_type=str(comment.get("action_type") or "comment"),
                    source_reaction_text=str(comment.get("reaction_text") or ""),
                    target_action_type=target_action,
                    target_reaction_text=target_text,
                    context_rank=comment.get("context_rank"),
                    source_emotion_score=source_state.emotion_score,
                    target_emotion_score_before=target_agent_before.emotion_score,
                    target_emotion_score_after=target_agent_after.emotion_score,
                    source_stance_score=source_state.stance_score,
                    target_stance_score_before=target_agent_before.stance_score,
                    target_stance_score_after=target_agent_after.stance_score,
                    source_influence_score=source_state.influence_score,
                    target_susceptibility_score=target_agent_before.susceptibility_score,
                    target_kol_sensitivity_score=target_agent_before.kol_sensitivity_score,
                    target_media_dependency_score=target_agent_before.media_dependency_score,
                    source_verified_type_name=source_state.verified_type_name,
                    source_propagation_role=source_state.propagation_role,
                    target_propagation_role=target_agent_before.propagation_role,
                    reason="本轮评论被纳入目标 Agent 的可见上下文，记录为候选影响边",
                    source="interaction_engine",
                    created_at=datetime.now().isoformat(timespec="seconds"),
                )
            )
        return records

    def compute_interaction_weight(
        self,
        source_agent: AgentState,
        target_agent: AgentState,
        interaction_type: str,
    ) -> float:
        weight = source_agent.influence_score * target_agent.susceptibility_score
        role = source_agent.propagation_role or ""
        verified = source_agent.verified_type_name or ""
        if source_agent.influence_score >= 0.75:
            weight *= 1.15
        if any(token in role for token in ("KOL", "潜在影响者", "高影响力")):
            weight *= 1.15
        if any(token in verified for token in ("媒体", "政府", "机构", "企业")):
            weight *= 1 + 0.30 * target_agent.media_dependency_score
        elif source_agent.influence_score >= 0.60:
            weight *= 1 + 0.30 * target_agent.kol_sensitivity_score
        if interaction_type == "repost":
            weight *= 1.20
        elif interaction_type == "reply":
            weight *= 1.10
        return round(clamp(weight, 0.01, 1.0), 4)

    @staticmethod
    def speaker_score(state: AgentState) -> float:
        return clamp(
            0.50 * state.influence_score
            + 0.30 * state.activity_score
            + 0.10 * state.media_dependency_score
            + 0.10 * state.repost_tendency_score
            + InteractionEngine._role_bonus(state.propagation_role),
            0.0,
            1.0,
        )

    @staticmethod
    def regular_participation_probability(
        state: AgentState,
        context_comments: list[ContextComment],
    ) -> float:
        context_bonus = min(0.20, 0.05 * len(context_comments))
        probability = state.activity_score + 0.10 * state.influence_score + context_bonus
        if any(comment.get("source_influence_score", 0.0) >= 0.75 for comment in context_comments):
            probability += 0.10 * state.kol_sensitivity_score
        return clamp(probability, 0.05, 0.95)

    @staticmethod
    def comment_from_state(
        state: AgentState,
        reaction: dict[str, Any],
        round_id: int,
        max_length: int,
    ) -> ContextComment:
        text = str(reaction.get("reaction_text", "")).strip()[:max_length]
        return {
            "round_id": round_id,
            "source_agent_id": state.agent_id,
            "source_user_id": state.user_id,
            "source_state": state,
            "source_influence_score": state.influence_score,
            "source_emotion_score": state.emotion_score,
            "source_stance_score": state.stance_score,
            "source_verified_type_name": state.verified_type_name,
            "source_propagation_role": state.propagation_role,
            "action_type": reaction.get("action_type", "comment"),
            "reaction_text": text,
        }

    @staticmethod
    def _role_bonus(propagation_role: str | None) -> float:
        role = propagation_role or ""
        if any(token in role for token in ("潜在影响者", "KOL", "高影响力")):
            return 0.15
        if "媒体信息跟随者" in role:
            return 0.08
        if "转发评论者" in role:
            return 0.05
        if "低活跃观察者" in role:
            return -0.10
        return 0.0

    @classmethod
    def _infer_interaction_type(cls, action_type: str, reaction_text: str) -> str:
        if action_type in {"repost", "repost_with_comment"}:
            return "repost"
        if any(token in reaction_text for token in cls.REPLY_HINTS):
            return "reply"
        return "same_round_context"
