from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_loader import (
    DEFAULT_AGENT_MEMORIES_PATH,
    DEFAULT_AGENT_PROFILES_PATH,
    DEFAULT_AGENT_SYS_PROMPTS_PATH,
    AgentRecord,
    load_agent_records,
)
from .agent_state import (
    AgentState,
    agent_state_to_dict,
    build_initial_agent_state,
    clamp,
    score_to_emotion_label,
    score_to_stance_label,
)
from .emotion_dynamics import (
    aggregate_neighbor_influence,
    build_event_influence_scores,
    update_agent_state_with_dynamics,
)
from .event_loader import DEFAULT_EVENTS_PATH, get_event_by_id
from .interaction_engine import ContextComment, InteractionEngine
from .interaction_schema import InteractionRecord, save_interactions_csv
from .llm_reaction_generator import LLMReactionGenerator, ReactionParseError
from .multiround_analyzer import compute_round_metrics, save_round_metrics, states_to_csv
from .multiround_config import MultiRoundSimulationConfig
from .network_builder import build_interaction_graph, save_graphml
from .single_event_simulator import build_event_message, estimate_participation_tendency, select_memories

LOGGER = logging.getLogger(__name__)

REACTION_EMOTION_LABELS = {
    "anger",
    "sadness",
    "fear",
    "joy",
    "disgust",
    "disappointment",
    "surprise",
    "sympathy",
    "confusion",
    "admiration",
    "mixed",
}
NEGATIVE_REACTION_EMOTION_LABELS = {"anger", "sadness", "fear", "disgust", "disappointment"}


@dataclass(frozen=True)
class ReactionGenerationResult:
    reaction: dict[str, Any]
    source: str
    llm_attempted: bool = False
    parse_status: str = "not_attempted"
    error_message: str = ""
    raw_output: str = ""


class MultiRoundSimulator:
    """Run a multi-round state simulation skeleton for one event."""

    def __init__(
        self,
        config: MultiRoundSimulationConfig,
        profiles_path: Path = DEFAULT_AGENT_PROFILES_PATH,
        memories_path: Path = DEFAULT_AGENT_MEMORIES_PATH,
        sys_prompts_path: Path = DEFAULT_AGENT_SYS_PROMPTS_PATH,
        events_path: Path = DEFAULT_EVENTS_PATH,
    ) -> None:
        self.config = config
        self.profiles_path = profiles_path
        self.memories_path = memories_path
        self.sys_prompts_path = sys_prompts_path
        self.events_path = events_path
        self.rng = random.Random(config.seed)
        self.interaction_engine = InteractionEngine(self.rng)
        self.llm_generator = LLMReactionGenerator(
            model_name=config.model_name,
            base_url=config.base_url,
            seed=config.seed,
        )
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    def run(self) -> dict[str, Any]:
        event = get_event_by_id(self.config.event_id, self.events_path)
        agents = self._load_agents()
        if not agents:
            raise ValueError(
                "No agent records matched the requested filters. "
                f"memory_user_level={self.config.memory_user_level!r} max_agents={self.config.max_agents!r}"
            )

        initial_states = [build_initial_agent_state(agent, event, self.run_id) for agent in agents]
        initial_states = [
            replace(
                state,
                round_id=0,
                is_active=False,
                last_action_type="ignore",
                last_reaction_text="",
                last_reason="",
                state_update_reason="初始状态",
                source="initial_profile",
            )
            for state in initial_states
        ]

        if self.config.dry_run:
            self._log_dry_run(event, initial_states)
            return {
                "run_id": self.run_id,
                "output_dir": "",
                "agent_count": len(initial_states),
                "rounds": self.config.rounds,
                "interaction_mode": self.config.interaction_mode,
                "dry_run": True,
            }

        if self.config.use_llm:
            LOGGER.info(
                "LLM enabled for active agents: model=%s base_url=%s max_llm_agents_per_round=%s llm_concurrency=%d",
                self.llm_generator.model_name,
                self.llm_generator.base_url,
                self.config.max_llm_agents_per_round,
                self.config.llm_concurrency,
            )
        if self.config.enable_emotion_dynamics and not self.config.enable_interactions:
            LOGGER.info("emotion dynamics enabled, neighbor influence disabled because interactions are disabled")

        run_output_dir = self.config.output_dir / self.run_id
        if run_output_dir.exists() and not self.config.overwrite:
            raise FileExistsError(f"Output directory already exists: {run_output_dir}")
        run_output_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(run_output_dir / "config.json", {"run_id": self.run_id, **self.config.to_dict()})
        self._write_json(run_output_dir / "selected_event.json", event)
        states_to_csv(initial_states, run_output_dir / "agent_initial_states.csv")

        all_states: list[AgentState] = list(initial_states)
        active_reactions: list[dict[str, Any]] = []
        all_interactions: list[InteractionRecord] = []
        metrics_list: list[dict[str, Any]] = [compute_round_metrics(initial_states, 0)]

        current_states = list(initial_states)
        agent_by_id = {agent.agent_id: agent for agent in agents}
        previous_round_comments: list[ContextComment] = []
        for round_id in range(1, self.config.rounds + 1):
            next_states, round_reactions, round_interactions, round_summary, previous_round_comments = self._run_round(
                current_states,
                agent_by_id,
                event,
                round_id,
                previous_round_comments,
            )
            all_states.extend(next_states)
            active_reactions.extend(round_reactions)
            all_interactions.extend(round_interactions)
            metrics_list.append(compute_round_metrics(next_states, round_id, round_summary))
            current_states = next_states

        states_to_csv(all_states, run_output_dir / "agent_states_by_round.csv")
        save_round_metrics(metrics_list, run_output_dir / "round_metrics.csv")
        self._write_active_reactions(active_reactions, run_output_dir / "active_reactions.jsonl")
        interactions_path = ""
        graphml_path = ""
        if self.config.enable_interactions:
            interactions_path = str(run_output_dir / "interactions.csv")
            save_interactions_csv(all_interactions, run_output_dir / "interactions.csv")
            graphml_path = str(run_output_dir / "network.graphml")
            try:
                graph = build_interaction_graph(all_states, all_interactions)
                save_graphml(graph, run_output_dir / "network.graphml")
            except RuntimeError as exc:
                graphml_path = ""
                LOGGER.warning("%s", exc)

        dynamics_summary = self._build_dynamics_summary(
            event,
            initial_states,
            current_states,
            metrics_list,
            all_interactions,
        )
        self._write_json(run_output_dir / "dynamics_summary.json", dynamics_summary)

        final_metrics = metrics_list[-1] if metrics_list else {}
        LOGGER.info("Multi-round simulation output: %s", run_output_dir)
        return {
            "run_id": self.run_id,
            "output_dir": str(run_output_dir),
            "agent_count": len(initial_states),
            "rounds": self.config.rounds,
            "final_avg_emotion_score": final_metrics.get("avg_emotion_score", 0.0),
            "final_avg_stance_score": final_metrics.get("avg_stance_score", 0.0),
            "round_metrics_path": str(run_output_dir / "round_metrics.csv"),
            "interaction_mode": self.config.interaction_mode,
            "interactions_path": interactions_path,
            "graphml_path": graphml_path,
            "interaction_count": len(all_interactions),
            "avg_abs_emotion_delta": final_metrics.get("avg_abs_emotion_delta", 0.0),
            "avg_abs_stance_delta": final_metrics.get("avg_abs_stance_delta", 0.0),
            "dynamics_summary_path": str(run_output_dir / "dynamics_summary.json"),
            "dry_run": False,
        }

    def _load_agents(self) -> list[AgentRecord]:
        return load_agent_records(
            profiles_path=self.profiles_path,
            memories_path=self.memories_path,
            sys_prompts_path=self.sys_prompts_path,
            memory_user_level=self.config.memory_user_level,
            max_agents=self.config.max_agents,
            random_seed=self.config.seed,
        )

    def _run_round(
        self,
        current_states: list[AgentState],
        agent_by_id: dict[str, AgentRecord],
        event: dict[str, Any],
        round_id: int,
        previous_round_comments: list[ContextComment] | None = None,
    ) -> tuple[list[AgentState], list[dict[str, Any]], list[InteractionRecord], dict[str, Any], list[ContextComment]]:
        if self.config.enable_interactions and self.config.interaction_mode == "kol_first":
            return self._run_interaction_round(
                current_states,
                agent_by_id,
                event,
                round_id,
                previous_round_comments or [],
            )

        active_agent_ids = self._select_active_agents(current_states, agent_by_id, event)
        active_states = [state for state in current_states if state.agent_id in active_agent_ids]
        llm_agent_ids = self._select_llm_agent_ids(
            [(state, "regular_agent") for state in active_states],
        )
        generated_by_id = self._run_async(
            self._generate_reactions_for_active_states(
                active_states,
                agent_by_id,
                event,
                round_id,
                llm_agent_ids,
                speaker_type="regular_agent",
                context_comments_by_id={},
            ),
        )
        next_states: list[AgentState] = []
        active_reactions: list[dict[str, Any]] = []
        round_comments: list[ContextComment] = []
        event_influence = build_event_influence_scores(event)

        for state in current_states:
            is_active = state.agent_id in active_agent_ids
            generation = generated_by_id.get(state.agent_id)
            if generation is None:
                generation = self._fallback_generation_result(state, event, round_id, is_active)
            reaction = generation.reaction
            next_state = self._update_state_from_reaction(
                state,
                reaction,
                round_id,
                is_active,
                source=generation.source,
            )
            if self.config.enable_emotion_dynamics:
                own_reaction = reaction if is_active else None
                next_state = self._apply_emotion_dynamics(
                    state,
                    next_state,
                    own_reaction,
                    {},
                    event_influence,
                )
            next_states.append(next_state)
            if is_active:
                active_reactions.append(
                    self._build_active_reaction_row(next_state, reaction, round_id, generation=generation),
                )
                round_comments.append(
                    self.interaction_engine.comment_from_state(
                        next_state,
                        reaction,
                        round_id,
                        self.config.max_context_comment_length,
                    )
                )

        LOGGER.info(
            "Round %d finished: active_agents=%d total_agents=%d",
            round_id,
            len(active_reactions),
            len(next_states),
        )
        summary = self._build_interaction_metric_summary(
            [],
            kol_speaker_count=0,
            regular_active_count=0,
            context_counts=[],
        )
        return next_states, active_reactions, [], summary, round_comments

    def _run_interaction_round(
        self,
        current_states: list[AgentState],
        agent_by_id: dict[str, AgentRecord],
        event: dict[str, Any],
        round_id: int,
        previous_round_comments: list[ContextComment],
    ) -> tuple[list[AgentState], list[dict[str, Any]], list[InteractionRecord], dict[str, Any], list[ContextComment]]:
        kol_speakers = self.interaction_engine.select_kol_speakers(current_states, self.config)
        regular_candidates = self.interaction_engine.select_regular_candidates(current_states, kol_speakers, self.config)
        next_state_by_id: dict[str, AgentState] = {}
        reaction_by_id: dict[str, dict[str, Any]] = {}
        active_reactions: list[dict[str, Any]] = []
        round_interactions: list[InteractionRecord] = []
        round_comments: list[ContextComment] = []
        context_counts: list[int] = []
        event_influence = build_event_influence_scores(event)
        llm_agent_ids = self._select_llm_agent_ids(
            [(state, "kol_speaker") for state in kol_speakers]
            + [(state, "regular_agent") for state in regular_candidates],
        )
        kol_generation_by_id = self._run_async(
            self._generate_reactions_for_active_states(
                kol_speakers,
                agent_by_id,
                event,
                round_id,
                llm_agent_ids,
                speaker_type="kol_speaker",
                context_comments_by_id={},
            ),
        )

        for state in kol_speakers:
            generation = kol_generation_by_id.get(state.agent_id)
            if generation is None:
                generation = self._fallback_generation_result(state, event, round_id, is_active=True)
            reaction = generation.reaction
            next_state = self._update_state_from_reaction(
                state,
                reaction,
                round_id,
                is_active=True,
                state_update_reason="本轮作为高影响力用户优先发声，状态根据自身表达轻微更新",
                source=generation.source,
            )
            next_state_by_id[state.agent_id] = next_state
            reaction_by_id[state.agent_id] = reaction
            active_reactions.append(
                self._build_active_reaction_row(
                    next_state,
                    reaction,
                    round_id,
                    speaker_type="kol_speaker",
                    context_comments=[],
                    generation=generation,
                )
            )
            round_comments.append(
                self.interaction_engine.comment_from_state(
                    next_state,
                    reaction,
                    round_id,
                    self.config.max_context_comment_length,
                )
            )

        for state in regular_candidates:
            candidate_comments = list(round_comments)
            if self.config.allow_previous_round_context:
                candidate_comments.extend(previous_round_comments)
            context_comments = self.interaction_engine.select_context_comments(state, candidate_comments, self.config)
            context_counts.append(len(context_comments))
            generation = self._run_async(
                self._generate_reaction_for_state(
                    state,
                    agent_by_id.get(state.agent_id),
                    event,
                    round_id,
                    is_active=True,
                    use_llm_for_agent=state.agent_id in llm_agent_ids,
                    context_comments=context_comments,
                    speaker_type="regular_agent",
                ),
            )
            reaction = generation.reaction
            reason = (
                "本轮参考评论区已有观点后参与表达，状态根据自身表达轻微更新"
                if context_comments
                else "本轮主动参与评论，状态根据自身表达轻微更新"
            )
            next_state = self._update_state_from_reaction(
                state,
                reaction,
                round_id,
                is_active=True,
                state_update_reason=reason,
                source=generation.source,
            )
            next_state_by_id[state.agent_id] = next_state
            reaction_by_id[state.agent_id] = reaction
            active_reactions.append(
                self._build_active_reaction_row(
                    next_state,
                    reaction,
                    round_id,
                    speaker_type="regular_agent",
                    context_comments=context_comments,
                    generation=generation,
                )
            )
            round_interactions.extend(
                self.interaction_engine.build_interaction_records(
                    context_comments,
                    state,
                    next_state,
                    reaction,
                    self.config,
                )
            )
            round_comments.append(
                self.interaction_engine.comment_from_state(
                    next_state,
                    reaction,
                    round_id,
                    self.config.max_context_comment_length,
                )
            )

        next_states: list[AgentState] = []
        influence_state_by_id = {state.agent_id: state for state in current_states}
        influence_state_by_id.update(next_state_by_id)
        for state in current_states:
            next_state = next_state_by_id.get(state.agent_id)
            if next_state is None:
                generation = self._fallback_generation_result(state, event, round_id, is_active=False)
                reaction = generation.reaction
                next_state = self._update_state_from_reaction(
                    state,
                    reaction,
                    round_id,
                    is_active=False,
                    source=generation.source,
                )
            if self.config.enable_emotion_dynamics:
                neighbor_influence = aggregate_neighbor_influence(
                    state.agent_id,
                    round_interactions,
                    influence_state_by_id,
                )
                next_state = self._apply_emotion_dynamics(
                    state,
                    next_state,
                    reaction_by_id.get(state.agent_id),
                    neighbor_influence,
                    event_influence,
                )
            next_states.append(next_state)

        if self.config.enable_emotion_dynamics:
            final_state_by_id = {state.agent_id: state for state in next_states}
            for interaction in round_interactions:
                source_state = final_state_by_id.get(interaction.source_agent_id)
                target_state = final_state_by_id.get(interaction.target_agent_id)
                if source_state is not None:
                    interaction.source_emotion_score = source_state.emotion_score
                    interaction.source_stance_score = source_state.stance_score
                if target_state is not None:
                    interaction.target_emotion_score_after = target_state.emotion_score
                    interaction.target_stance_score_after = target_state.stance_score

        LOGGER.info(
            "Round %d finished: kol_speakers=%d regular_active=%d interactions=%d total_agents=%d",
            round_id,
            len(kol_speakers),
            len(regular_candidates),
            len(round_interactions),
            len(next_states),
        )
        summary = self._build_interaction_metric_summary(
            round_interactions,
            kol_speaker_count=len(kol_speakers),
            regular_active_count=len(regular_candidates),
            context_counts=context_counts,
        )
        return next_states, active_reactions, round_interactions, summary, round_comments

    @staticmethod
    def _run_async(coro: Any) -> Any:
        return asyncio.run(coro)

    def _select_llm_agent_ids(self, candidates: list[tuple[AgentState, str]]) -> set[str]:
        if not self.config.use_llm or not candidates:
            return set()
        if self.config.max_llm_agents_per_round is None:
            return {state.agent_id for state, _speaker_type in candidates}
        if self.config.max_llm_agents_per_round <= 0:
            return set()

        ranked = sorted(
            candidates,
            key=lambda item: self._llm_priority_key(item[0], item[1]),
            reverse=True,
        )
        return {state.agent_id for state, _speaker_type in ranked[: self.config.max_llm_agents_per_round]}

    def _llm_priority_key(self, state: AgentState, speaker_type: str) -> tuple[float, float, float, float, str]:
        kol_rank = 1.0 if speaker_type == "kol_speaker" else 0.0
        return (
            kol_rank,
            self.interaction_engine.speaker_score(state),
            state.influence_score,
            state.activity_score,
            state.agent_id,
        )

    async def _generate_reactions_for_active_states(
        self,
        states: list[AgentState],
        agent_by_id: dict[str, AgentRecord],
        event: dict[str, Any],
        round_id: int,
        llm_agent_ids: set[str],
        speaker_type: str,
        context_comments_by_id: dict[str, list[ContextComment]],
    ) -> dict[str, ReactionGenerationResult]:
        semaphore = asyncio.Semaphore(self.config.llm_concurrency)

        async def generate_one(state: AgentState) -> tuple[str, ReactionGenerationResult]:
            async with semaphore:
                result = await self._generate_reaction_for_state(
                    state,
                    agent_by_id.get(state.agent_id),
                    event,
                    round_id,
                    is_active=True,
                    use_llm_for_agent=state.agent_id in llm_agent_ids,
                    context_comments=context_comments_by_id.get(state.agent_id, []),
                    speaker_type=speaker_type,
                )
                return state.agent_id, result

        pairs = await asyncio.gather(*(generate_one(state) for state in states))
        return dict(pairs)

    async def _generate_reaction_for_state(
        self,
        state: AgentState,
        agent_record: AgentRecord | None,
        event: dict[str, Any],
        round_id: int,
        is_active: bool,
        use_llm_for_agent: bool,
        context_comments: list[ContextComment] | None = None,
        speaker_type: str = "regular_agent",
    ) -> ReactionGenerationResult:
        if not is_active:
            return self._fallback_generation_result(state, event, round_id, is_active=False, context_comments=context_comments)
        if not use_llm_for_agent:
            return self._fallback_generation_result(state, event, round_id, is_active=True, context_comments=context_comments)

        fallback_state = replace(state, is_active=True)
        fallback = generate_fallback_reaction(fallback_state, event, round_id, context_comments=context_comments)
        if agent_record is None:
            return self._llm_fallback_result(
                fallback,
                parse_status="failed",
                error_message="Missing agent record for LLM generation.",
            )
        if not self.llm_generator.api_key:
            return self._llm_fallback_result(
                fallback,
                parse_status="failed",
                error_message="Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY before running live simulation.",
            )

        selected_memories = select_memories(event, agent_record.memories)
        event_message = self._build_multiround_event_message(
            event,
            agent_record,
            state,
            selected_memories,
            round_id,
            speaker_type,
            context_comments or [],
        )
        try:
            raw_output, reaction = await self.llm_generator.generate(agent_record, event_message, selected_memories)
        except ReactionParseError as exc:
            return self._llm_fallback_result(
                fallback,
                parse_status="parse_failed",
                error_message=str(exc),
                raw_output=exc.raw_output,
            )
        except Exception as exc:  # noqa: BLE001 - multi-round simulations should continue on LLM failure
            LOGGER.exception("LLM generation failed for agent %s in round %d: %s", state.agent_id, round_id, exc)
            return self._llm_fallback_result(
                fallback,
                parse_status="failed",
                error_message=str(exc),
            )

        return ReactionGenerationResult(
            reaction=reaction.model_dump(),
            source="llm",
            llm_attempted=True,
            parse_status="success",
            error_message="",
            raw_output=raw_output,
        )

    def _fallback_generation_result(
        self,
        state: AgentState,
        event: dict[str, Any],
        round_id: int,
        is_active: bool,
        context_comments: list[ContextComment] | None = None,
    ) -> ReactionGenerationResult:
        return ReactionGenerationResult(
            reaction=generate_fallback_reaction(
                replace(state, is_active=is_active),
                event,
                round_id,
                context_comments=context_comments,
            ),
            source="fallback_rule",
            llm_attempted=False,
            parse_status="not_attempted",
            error_message="",
            raw_output="",
        )

    @staticmethod
    def _llm_fallback_result(
        fallback: dict[str, Any],
        parse_status: str,
        error_message: str,
        raw_output: str = "",
    ) -> ReactionGenerationResult:
        return ReactionGenerationResult(
            reaction={
                **fallback,
                "reason": f"{fallback.get('reason', '')}；LLM调用失败，已降级为规则生成。",
            },
            source="llm_fallback",
            llm_attempted=True,
            parse_status=parse_status,
            error_message=error_message,
            raw_output=raw_output,
        )

    def _build_multiround_event_message(
        self,
        event: dict[str, Any],
        agent_record: AgentRecord,
        state: AgentState,
        selected_memories: list[dict[str, Any]],
        round_id: int,
        speaker_type: str,
        context_comments: list[ContextComment],
    ) -> str:
        participation_tendency, _score = estimate_participation_tendency(agent_record.profile)
        base_message = build_event_message(event, selected_memories, participation_tendency)
        context_text = self._format_context_comments(context_comments)
        speaker_label = "高影响力用户优先发声" if speaker_type == "kol_speaker" else "普通活跃用户参与讨论"
        return (
            f"{base_message}\n\n"
            "【多轮仿真上下文】\n"
            f"当前轮次：第 {round_id} 轮。\n"
            f"本轮角色：{speaker_label}。\n"
            f"上一轮自身情绪状态：{state.emotion_label}（分数 {state.emotion_score:.3f}）。\n"
            f"上一轮自身立场状态：{state.stance_label}（分数 {state.stance_score:.3f}）。\n"
            f"上一轮可见行为：{state.last_action_type}；上一轮表达：{state.last_reaction_text or '无'}。\n\n"
            "【本轮可见评论上下文】\n"
            f"{context_text}\n\n"
            "【多轮输出要求】\n"
            "本轮程序已判定该用户会参与讨论，participate 必须为 true。\n"
            "请结合可见评论上下文生成自然的微博式表达；如果没有可见评论，则只基于事件和个人历史风格表达。\n"
            "请根据用户画像生成差异化反应，不同用户可表现为愤怒、厌恶、困惑、失望、观望或同情，不要让所有用户都使用同一种情绪表达。\n"
            "如果前几轮或上下文中已经多次出现相似表达，本轮应避免重复相同句式，可以从补充细节、回应评论区分歧、表达观望或提出疑问等角度生成。"
        )

    @staticmethod
    def _format_context_comments(context_comments: list[ContextComment]) -> str:
        if not context_comments:
            return "本轮暂无可见评论上下文。"
        lines: list[str] = []
        for index, comment in enumerate(context_comments, start=1):
            text = str(comment.get("reaction_text") or "").strip()
            source_agent_id = str(comment.get("source_agent_id") or "unknown")
            source_role = str(comment.get("source_propagation_role") or "")
            source_verified = str(comment.get("source_verified_type_name") or "")
            lines.append(
                f"{index}. agent_id={source_agent_id} role={source_role or 'unknown'} "
                f"verified={source_verified or 'unknown'}：{text or '无文本'}"
            )
        return "\n".join(lines)

    def _select_active_agents(
        self,
        states: list[AgentState],
        agent_by_id: dict[str, AgentRecord],
        event: dict[str, Any],
    ) -> set[str]:
        selected: list[AgentState] = []
        for state in states:
            agent = agent_by_id.get(state.agent_id)
            prob = self._participation_probability(state, event, agent.profile if agent else {})
            if self.rng.random() < prob:
                selected.append(state)

        if self.config.active_agent_limit and len(selected) > self.config.active_agent_limit:
            selected.sort(key=lambda item: (item.activity_score + item.influence_score, item.agent_id), reverse=True)
            selected = selected[: self.config.active_agent_limit]
        return {state.agent_id for state in selected}

    @staticmethod
    def _participation_probability(state: AgentState, event: dict[str, Any], profile: dict[str, Any]) -> float:
        behavior = profile.get("behavior_parameters") if isinstance(profile.get("behavior_parameters"), dict) else {}
        public_issue_ratio = 0.0
        if event.get("event_type") == "public_issue":
            public_issue_ratio = clamp(
                behavior.get("public_issue_topic_ratio", behavior.get("final_public_issue_topic_ratio", 0.0)),
                0.0,
                1.0,
            )
        probability = state.activity_score + 0.1 * state.influence_score + 0.1 * public_issue_ratio
        return clamp(probability, 0.05, 0.95)

    def _update_state_from_reaction(
        self,
        previous: AgentState,
        reaction: dict[str, Any],
        round_id: int,
        is_active: bool,
        state_update_reason: str | None = None,
        source: str = "fallback_rule",
    ) -> AgentState:
        old_emotion = previous.emotion_score
        old_stance = previous.stance_score
        if not is_active:
            return replace(
                previous,
                round_id=round_id,
                is_active=False,
                last_action_type="ignore",
                last_reaction_text="",
                last_reason=reaction.get("reason", ""),
                old_emotion_score=old_emotion,
                new_emotion_score=old_emotion,
                emotion_delta=0.0,
                old_stance_score=old_stance,
                new_stance_score=old_stance,
                stance_delta=0.0,
                neighbor_emotion_score=0.0,
                neighbor_stance_score=0.0,
                neighbor_influence_weight_sum=0.0,
                neighbor_count=0,
                high_influence_neighbor_count=0,
                media_neighbor_count=0,
                kol_neighbor_count=0,
                event_emotion_score=0.0,
                event_stance_score=0.0,
                own_reaction_emotion_score=0.0,
                own_reaction_stance_score=0.0,
                dynamics_enabled=False,
                state_update_reason="本轮未参与，状态保持稳定",
                source=source,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )

        emotion_delta = self._score_delta(previous.emotion_label, positive_label="positive")
        stance_delta = self._score_delta(previous.stance_label, positive_label="support")
        new_emotion = clamp(old_emotion + emotion_delta, -1.0, 1.0)
        new_stance = clamp(old_stance + stance_delta, -1.0, 1.0)
        return replace(
            previous,
            round_id=round_id,
            is_active=True,
            last_action_type=str(reaction.get("action_type", "comment")),
            last_reaction_text=str(reaction.get("reaction_text", "")),
            last_reason=str(reaction.get("reason", "")),
            old_emotion_score=old_emotion,
            new_emotion_score=new_emotion,
            emotion_delta=round(new_emotion - old_emotion, 4),
            old_stance_score=old_stance,
            new_stance_score=new_stance,
            stance_delta=round(new_stance - old_stance, 4),
            neighbor_emotion_score=0.0,
            neighbor_stance_score=0.0,
            neighbor_influence_weight_sum=0.0,
            neighbor_count=0,
            high_influence_neighbor_count=0,
            media_neighbor_count=0,
            kol_neighbor_count=0,
            event_emotion_score=0.0,
            event_stance_score=0.0,
            own_reaction_emotion_score=0.0,
            own_reaction_stance_score=0.0,
            dynamics_enabled=False,
            emotion_score=new_emotion,
            stance_score=new_stance,
            emotion_label=score_to_emotion_label(new_emotion),
            stance_label=score_to_stance_label(new_stance),
            state_update_reason=state_update_reason or "本轮主动参与评论，状态根据自身表达轻微更新",
            source=source,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

    def _score_delta(self, label: str, positive_label: str) -> float:
        amount = self.rng.uniform(0.05, 0.10)
        if label == positive_label:
            return amount
        if label == "negative" or label == "against":
            return -amount
        return 0.0

    def _apply_emotion_dynamics(
        self,
        previous: AgentState,
        base_state: AgentState,
        own_reaction: dict[str, Any] | None,
        neighbor_influence: dict[str, Any],
        event_influence: dict[str, Any],
    ) -> AgentState:
        dynamic_state = update_agent_state_with_dynamics(
            previous,
            own_reaction,
            neighbor_influence,
            event_influence,
            self.config,
        )
        return replace(
            dynamic_state,
            round_id=base_state.round_id,
            is_active=base_state.is_active,
            last_action_type=base_state.last_action_type,
            last_reaction_text=base_state.last_reaction_text,
            last_reason=base_state.last_reason,
            source="emotion_dynamics",
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

    @staticmethod
    def _build_interaction_metric_summary(
        interactions: list[InteractionRecord],
        kol_speaker_count: int,
        regular_active_count: int,
        context_counts: list[int],
    ) -> dict[str, Any]:
        weights = [record.weight for record in interactions]
        agents_with_context = sum(1 for count in context_counts if count > 0)
        avg_context = round(sum(context_counts) / len(context_counts), 4) if context_counts else 0.0
        return {
            "kol_speaker_count": kol_speaker_count,
            "regular_active_count": regular_active_count,
            "interaction_count": len(interactions),
            "avg_interaction_weight": round(sum(weights) / len(weights), 4) if weights else 0.0,
            "high_influence_interaction_count": sum(
                1 for record in interactions if record.source_influence_score >= 0.75
            ),
            "agents_with_context_count": agents_with_context,
            "avg_context_comment_count": avg_context,
        }

    def _build_dynamics_summary(
        self,
        event: dict[str, Any],
        initial_states: list[AgentState],
        final_states: list[AgentState],
        metrics_list: list[dict[str, Any]],
        interactions: list[InteractionRecord],
    ) -> dict[str, Any]:
        initial_avg_emotion = self._avg_state_score(initial_states, "emotion_score")
        final_avg_emotion = self._avg_state_score(final_states, "emotion_score")
        initial_avg_stance = self._avg_state_score(initial_states, "stance_score")
        final_avg_stance = self._avg_state_score(final_states, "stance_score")
        final_metrics = metrics_list[-1] if metrics_list else {}
        return {
            "run_id": self.run_id,
            "event_id": self.config.event_id,
            "topic": event.get("topic", ""),
            "dynamics_enabled": self.config.enable_emotion_dynamics,
            "interaction_enabled": self.config.enable_interactions,
            "total_agents": len(initial_states),
            "rounds": self.config.rounds,
            "initial_avg_emotion_score": initial_avg_emotion,
            "final_avg_emotion_score": final_avg_emotion,
            "emotion_score_change": round(final_avg_emotion - initial_avg_emotion, 4),
            "initial_avg_stance_score": initial_avg_stance,
            "final_avg_stance_score": final_avg_stance,
            "stance_score_change": round(final_avg_stance - initial_avg_stance, 4),
            "final_emotion_distribution": self._label_distribution(final_states, "emotion_label"),
            "final_stance_distribution": self._label_distribution(final_states, "stance_label"),
            "max_emotion_volatility": max((float(item.get("emotion_volatility", 0.0) or 0.0) for item in metrics_list), default=0.0),
            "max_stance_volatility": max((float(item.get("stance_volatility", 0.0) or 0.0) for item in metrics_list), default=0.0),
            "final_polarization_score": final_metrics.get("polarization_score", 0.0),
            "total_interaction_count": len(interactions),
            "agents_ever_affected_by_neighbors": len(
                {
                    interaction.target_agent_id
                    for interaction in interactions
                    if interaction.target_agent_id
                }
            ),
            "parameter_config": {
                "self_retention": self.config.self_retention,
                "social_influence_strength": self.config.social_influence_strength,
                "event_influence_strength": self.config.event_influence_strength,
                "reaction_influence_strength": self.config.reaction_influence_strength,
                "stance_retention": self.config.stance_retention,
                "social_stance_strength": self.config.social_stance_strength,
                "event_stance_strength": self.config.event_stance_strength,
                "reaction_stance_strength": self.config.reaction_stance_strength,
                "enable_saturation_damping": self.config.enable_saturation_damping,
                "saturation_damping_strength": self.config.saturation_damping_strength,
            },
        }

    @staticmethod
    def _avg_state_score(states: list[AgentState], field_name: str) -> float:
        values = [float(getattr(state, field_name, 0.0) or 0.0) for state in states]
        return round(sum(values) / len(values), 4) if values else 0.0

    @staticmethod
    def _label_distribution(states: list[AgentState], field_name: str) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for state in states:
            label = str(getattr(state, field_name, "") or "unknown")
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def _build_active_reaction_row(
        self,
        state: AgentState,
        reaction: dict[str, Any],
        round_id: int,
        speaker_type: str = "regular_agent",
        context_comments: list[ContextComment] | None = None,
        generation: ReactionGenerationResult | None = None,
    ) -> dict[str, Any]:
        context_comments = context_comments or []
        generation = generation or ReactionGenerationResult(reaction=reaction, source="fallback_rule")
        context_agent_ids = [
            str(comment.get("source_agent_id"))
            for comment in context_comments
            if comment.get("source_agent_id")
        ]
        return {
            "run_id": state.run_id,
            "event_id": state.event_id,
            "round_id": round_id,
            "agent_id": state.agent_id,
            "user_id": state.user_id,
            "memory_user_level": state.memory_user_level,
            "propagation_role": state.propagation_role,
            "influence_score": round(state.influence_score, 4),
            "activity_score": round(state.activity_score, 4),
            "participate": reaction.get("participate", True),
            "action_type": reaction.get("action_type", "comment"),
            "emotion_label": reaction.get("emotion_label", state.emotion_label),
            "emotion_intensity": reaction.get("emotion_intensity", 1),
            "stance_label": reaction.get("stance_label", state.stance_label),
            "stance_intensity": reaction.get("stance_intensity", 1),
            "reaction_text": reaction.get("reaction_text", ""),
            "reason": reaction.get("reason", ""),
            "source": generation.source,
            "llm_attempted": generation.llm_attempted,
            "parse_status": generation.parse_status,
            "error_message": generation.error_message,
            "raw_output": generation.raw_output,
            "speaker_type": speaker_type,
            "context_agent_ids": ",".join(context_agent_ids),
            "context_comment_count": len(context_comments),
            "influenced_by_high_influence": any(
                float(comment.get("source_influence_score", 0.0) or 0.0) >= 0.75
                for comment in context_comments
            ),
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_active_reactions(records: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_dry_run(self, event: dict[str, Any], states: list[AgentState]) -> None:
        LOGGER.info("DRY RUN: no multi-round loop will be executed.")
        LOGGER.info("Event: %s | %s", event.get("event_id"), event.get("topic"))
        for state in states[:3]:
            LOGGER.info(
                "AgentState: agent_id=%s user_id=%s level=%s influence=%.3f activity=%.3f emotion=%s(%.3f) stance=%s(%.3f)",
                state.agent_id,
                state.user_id,
                state.memory_user_level,
                state.influence_score,
                state.activity_score,
                state.emotion_label,
                state.emotion_score,
                state.stance_label,
                state.stance_score,
            )
        if self.config.use_llm:
            speaker_candidates = [(state, "regular_agent") for state in states]
            if self.config.enable_interactions and self.config.interaction_mode == "kol_first":
                speaker_candidates = [(state, "kol_speaker") for state in states]
            llm_agent_ids = self._select_llm_agent_ids(speaker_candidates)
            LOGGER.info(
                "DRY RUN: no model calls will be made. LLM budget would allow %d candidate(s) per round.",
                len(llm_agent_ids),
            )
            ranked = sorted(
                speaker_candidates,
                key=lambda item: self._llm_priority_key(item[0], item[1]),
                reverse=True,
            )[:5]
            LOGGER.info("DRY RUN top LLM priority candidates:")
            for state, speaker_type in ranked:
                LOGGER.info(
                    "  agent_id=%s speaker_type=%s selected_by_budget=%s influence=%.3f activity=%.3f",
                    state.agent_id,
                    speaker_type,
                    state.agent_id in llm_agent_ids,
                    state.influence_score,
                    state.activity_score,
                )
        if self.config.enable_interactions:
            scored = sorted(
                states,
                key=lambda state: (
                    self.interaction_engine.speaker_score(state),
                    state.influence_score,
                    state.agent_id,
                ),
                reverse=True,
            )[:5]
            LOGGER.info("DRY RUN candidate KOL speakers:")
            for state in scored:
                LOGGER.info(
                    "  agent_id=%s influence=%.3f activity=%.3f role=%s verified=%s",
                    state.agent_id,
                    state.influence_score,
                    state.activity_score,
                    state.propagation_role,
                    state.verified_type_name,
                )
            if len(scored) >= 2:
                sample_comment = self.interaction_engine.comment_from_state(
                    scored[0],
                    {
                        "action_type": "comment",
                        "reaction_text": _reaction_text(scored[0].emotion_label, scored[0].stance_label, event, 1),
                    },
                    round_id=1,
                    max_length=self.config.max_context_comment_length,
                )
                context = self.interaction_engine.select_context_comments(scored[1], [sample_comment], self.config)
                LOGGER.info(
                    "DRY RUN sample context_comments for agent_id=%s: %s",
                    scored[1].agent_id,
                    [
                        {
                            "source_agent_id": item.get("source_agent_id"),
                            "reaction_text": item.get("reaction_text"),
                            "context_rank": item.get("context_rank"),
                        }
                        for item in context
                    ],
                )
            LOGGER.info(
                "DRY RUN expected outputs: config.json, selected_event.json, agent_initial_states.csv, "
                "agent_states_by_round.csv, round_metrics.csv, active_reactions.jsonl, interactions.csv, network.graphml"
            )


def generate_fallback_reaction(
    agent_state: AgentState,
    event: dict[str, Any],
    round_id: int,
    context_comments: list[ContextComment] | None = None,
) -> dict[str, Any]:
    reaction_emotion_label = _reaction_emotion_label(agent_state.emotion_label, event)
    reaction_stance_label = _reaction_stance_label(agent_state.stance_label)
    if not agent_state.is_active:
        return {
            "participate": False,
            "action_type": "ignore",
            "emotion_label": reaction_emotion_label,
            "emotion_intensity": 0,
            "stance_label": reaction_stance_label,
            "stance_intensity": 0,
            "reaction_text": "",
            "reason": "本轮未被参与概率选中，保持围观。",
        }

    action_type = "repost_with_comment" if agent_state.repost_tendency_score >= 0.65 else "comment"
    text = _reaction_text(agent_state.emotion_label, agent_state.stance_label, event, round_id, context_comments)
    return {
        "participate": True,
        "action_type": action_type,
        "emotion_label": reaction_emotion_label,
        "emotion_intensity": 1,
        "stance_label": reaction_stance_label,
        "stance_intensity": 1,
        "reaction_text": text,
        "reason": (
            "根据该用户活跃度、当前事件倾向和可见评论上下文生成的规则反应"
            if context_comments
            else "根据该用户活跃度和当前事件倾向生成的规则反应"
        ),
    }


def _reaction_emotion_label(internal_emotion_label: str, event: dict[str, Any]) -> str:
    """Map coarse AgentState emotion labels to ReactionSchema-compatible labels."""

    label = internal_emotion_label.strip().lower()
    dominant_emotion = str(event.get("dominant_emotion_label") or "").strip().lower()
    event_tendency = str(event.get("event_emotion_tendency") or "").strip().lower()

    if label == "positive":
        if dominant_emotion in {"joy", "sympathy", "admiration", "surprise"}:
            return dominant_emotion
        return "joy"
    if label == "neutral":
        return "mixed"
    if label == "negative":
        if dominant_emotion in NEGATIVE_REACTION_EMOTION_LABELS:
            return dominant_emotion
        if event_tendency in NEGATIVE_REACTION_EMOTION_LABELS:
            return event_tendency
        if str(event.get("dominant_norm_violation") or "").strip().lower() == "high":
            return "disgust"
        if str(event.get("event_type") or "").strip() == "public_issue":
            return "anger"
        # ReactionSchema currently has no "worry" label, so confusion is the schema-safe fallback.
        return "confusion"
    if label in REACTION_EMOTION_LABELS:
        return label
    return "mixed"


def _reaction_stance_label(internal_stance_label: str) -> str:
    if internal_stance_label == "support":
        return "favor"
    if internal_stance_label in {"against", "neutral", "mixed", "unclear"}:
        return internal_stance_label
    return "neutral"


def _reaction_text(
    emotion_label: str,
    stance_label: str,
    event: dict[str, Any],
    round_id: int,
    context_comments: list[ContextComment] | None = None,
) -> str:
    topic = str(event.get("topic") or "这件事").strip()
    context_comments = context_comments or []
    if context_comments:
        has_high_influence = any(
            float(comment.get("source_influence_score", 0.0) or 0.0) >= 0.75 for comment in context_comments
        )
        has_verified_source = any(
            any(token in str(comment.get("source_verified_type_name") or "") for token in ("媒体", "政府", "机构"))
            for comment in context_comments
        )
        if stance_label == "against" and has_high_influence:
            return "前面说得有道理，这个解释确实还不够让人信服。"
        if emotion_label == "negative":
            return "看了前面的说法，感觉这事还是得继续追问清楚。"
        if stance_label == "support" and has_verified_source:
            return "如果通报内容属实，那这个处理方向还算明确。"
        if stance_label == "neutral":
            return "评论区分歧挺大，还是等更完整的信息吧。"

    templates = {
        ("negative", "against"): [
            "这事看着真的有点离谱，希望后续能给个清楚说法。",
            "越看越觉得不太对，还是等一个更明确的回应吧。",
        ],
        ("negative", "neutral"): [
            "先观望吧，现在信息还不太完整。",
            "情绪先放一放，关键还是看后续有没有细节。",
        ],
        ("negative", "support"): [
            "能处理当然好，但现在还是有些地方让人不放心。",
            "支持继续查清楚，别让大家的疑问悬着。",
        ],
        ("positive", "support"): [
            "如果情况属实，这个处理还是比较及时的。",
            "看到有进展总归是好事，希望后面也能跟上。",
        ],
        ("positive", "neutral"): [
            "目前看还有不少细节，等更多信息出来再说。",
            "先保持关注吧，希望事情能往清楚的方向走。",
        ],
        ("positive", "against"): [
            "有回应是好事，但这个说法还是需要再解释清楚。",
            "至少现在有人关注了，后续别轻轻带过。",
        ],
        ("neutral", "support"): [
            "如果后续信息能对上，这个方向可以继续看。",
            "先看通报和证据，处理过程透明些就更好。",
        ],
        ("neutral", "against"): [
            "这个说法还需要更多证据支撑，先继续关注。",
            "目前疑点还是有的，希望后续能讲清楚。",
        ],
        ("neutral", "neutral"): [
            "继续看看后续通报，先不急着下结论。",
            "信息还在变化，先放一放等更多细节。",
        ],
    }
    choices = templates.get((emotion_label, stance_label), templates[("neutral", "neutral")])
    index = (round_id + len(topic)) % len(choices)
    return choices[index]
