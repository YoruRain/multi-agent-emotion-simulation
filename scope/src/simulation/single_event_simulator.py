from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .agent_loader import (
    DEFAULT_AGENT_MEMORIES_PATH,
    DEFAULT_AGENT_PROFILES_PATH,
    DEFAULT_AGENT_SYS_PROMPTS_PATH,
    PROJECT_ROOT,
    AgentRecord,
    load_agent_records,
)
from .event_loader import DEFAULT_EVENTS_PATH, get_event_by_id
from .reaction_schema import ReactionSchema, normalize_structured_output, parse_reaction_json
from .result_analyzer import analyze_results

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scope" / "data" / "outputs" / "simulation" / "single_event"
DEFAULT_MODEL_NAME = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
MEMORY_PRIORITY_PUBLIC_ISSUE = ["public_issue_memory", "style_memory", "propagation_memory", "general_memory"]
MEMORY_PRIORITY_DEFAULT = ["style_memory", "propagation_memory", "general_memory", "public_issue_memory"]
PARTICIPATION_TENDENCY_WEIGHTS = {
    "final_public_issue_topic_ratio": 0.30,
    "repost_ratio": 0.15,
    "repost_with_comment_ratio": 0.20,
    "kol_sensitivity_score": 0.20,
    "propagation_activity_level": 0.15,
}
PARTICIPATION_TENDENCY_LABELS = {
    "low": "低",
    "medium": "中等",
    "high": "高",
}


class ReactionParseError(ValueError):
    """Raised when model output cannot be parsed into ReactionSchema."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class SimulatorPaths:
    profiles_path: Path = DEFAULT_AGENT_PROFILES_PATH
    memories_path: Path = DEFAULT_AGENT_MEMORIES_PATH
    sys_prompts_path: Path = DEFAULT_AGENT_SYS_PROMPTS_PATH
    events_path: Path = DEFAULT_EVENTS_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class ParticipationGateDecision:
    probability: float
    draw: float
    passed: bool


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def _nested_get(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _summarize_distribution(distribution: Any) -> str:
    if not isinstance(distribution, dict) or not distribution:
        return "暂无分布信息"
    items: list[tuple[str, float]] = []
    for key, value in distribution.items():
        try:
            items.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    if not items:
        return "暂无分布信息"
    items.sort(key=lambda item: item[1], reverse=True)
    return "，".join(f"{key}: {value:.1%}" for key, value in items[:5])


def _propagation_activity_score(profile: dict[str, Any]) -> float | None:
    raw_value = None
    for candidate in [
        _nested_get(profile, "behavior_parameters", "propagation_activity_level", default=None),
        _nested_get(profile, "base_identity", "propagation_activity_level", default=None),
        _nested_get(profile, "prompt_profile", "propagation_activity_level", default=None),
    ]:
        if candidate is not None and str(candidate).strip():
            raw_value = candidate
            break
    numeric_value = _safe_float(raw_value)
    if numeric_value is not None:
        return numeric_value

    text = _safe_text(raw_value)
    if not text:
        text = " ".join(
            [
                _safe_text(_nested_get(profile, "prompt_profile", "identity_summary")),
                _safe_text(_nested_get(profile, "prompt_profile", "propagation_summary")),
            ],
        )
    lowered = text.lower()
    if any(token in lowered for token in ["high", "高活跃", "活跃程度高", "活跃度高"]):
        return 1.0
    if any(token in lowered for token in ["medium", "中等", "中活跃", "活跃程度中", "活跃度中"]):
        return 0.5
    if any(token in lowered for token in ["low", "低活跃", "活跃程度低", "活跃度低"]):
        return 0.15
    return None


def estimate_participation_tendency(profile: dict[str, Any]) -> tuple[str, float]:
    """Estimate a coarse prior participation tendency from behavior parameters."""

    weighted_score = 0.0
    total_weight = 0.0
    for field, weight in PARTICIPATION_TENDENCY_WEIGHTS.items():
        if field == "propagation_activity_level":
            value = _propagation_activity_score(profile)
        else:
            value = _safe_float(_nested_get(profile, "behavior_parameters", field, default=None))
        if value is None:
            continue
        weighted_score += value * weight
        total_weight += weight

    score = weighted_score / total_weight if total_weight else 0.5
    if score < 0.25:
        return "low", score
    if score < 0.55:
        return "medium", score
    return "high", score


def decide_participation(profile: dict[str, Any], rng: random.Random) -> ParticipationGateDecision:
    """Decide whether this agent participates before calling the LLM."""

    _tendency, score = estimate_participation_tendency(profile)
    probability = max(0.0, min(1.0, score))
    draw = rng.random()
    return ParticipationGateDecision(
        probability=probability,
        draw=draw,
        passed=draw < probability,
    )


def _is_participating_reaction(reaction: ReactionSchema) -> bool:
    return reaction.participate and reaction.action_type != "ignore" and bool(_safe_text(reaction.reaction_text))


def select_memories(event: dict[str, Any], memories: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Select a small set of memories according to event type and mark priority."""

    priority = MEMORY_PRIORITY_PUBLIC_ISSUE if event.get("event_type") == "public_issue" else MEMORY_PRIORITY_DEFAULT
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for mark in priority:
        for memory in memories:
            if _safe_text(memory.get("mark")) != mark:
                continue
            memory_id = _safe_text(memory.get("memory_id")) or _safe_text(memory.get("weibo_id")) or json.dumps(
                memory,
                ensure_ascii=False,
                sort_keys=True,
            )
            if memory_id in selected_ids:
                continue
            selected.append(memory)
            selected_ids.add(memory_id)
            if len(selected) >= limit:
                return selected
    return selected


def _format_memory_snippets(memories: list[dict[str, Any]], max_chars: int = 320) -> str:
    if not memories:
        return "暂无可参考历史记忆。"
    lines: list[str] = []
    for index, memory in enumerate(memories, start=1):
        mark = _safe_text(memory.get("mark"), "general_memory")
        content = _safe_text(memory.get("content"))
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        lines.append(f"{index}. [{mark}] {content}")
    return "\n".join(lines)


def build_event_message(
    event: dict[str, Any],
    memories: list[dict[str, Any]],
    participation_tendency: str | None = None,
) -> str:
    """Build the user message sent to a Weibo user Agent."""

    emotion_distribution = _summarize_distribution(event.get("emotion_distribution"))
    stance_distribution = _summarize_distribution(event.get("stance_distribution"))
    memory_snippets = _format_memory_snippets(memories)
    tendency_key = participation_tendency or "medium"
    tendency_label = PARTICIPATION_TENDENCY_LABELS.get(tendency_key, "中等")

    return (
        "程序已经根据用户行为参数完成是否参与决策：该用户本轮会参与讨论。"
        "请根据你的长期特征和历史表达习惯，只生成参与后的微博式行为与内容，并只输出严格 JSON。\n\n"
        "【热点事件】\n"
        f"话题：{_safe_text(event.get('topic'), '未知话题')}\n"
        f"事件背景：{_safe_text(event.get('event_context'), '暂无事件背景')}\n"
        f"事件类型：{_safe_text(event.get('event_type'), 'unknown')}\n"
        f"评论区情绪倾向：{_safe_text(event.get('event_emotion_tendency'), 'unknown')}\n"
        f"情绪摘要：{_safe_text(event.get('event_emotion_summary'), '暂无情绪摘要')}\n"
        f"立场焦点：{_safe_text(event.get('event_stance_focus'), '暂无立场焦点')}\n"
        f"主要评价对象：{_safe_text(event.get('dominant_stance_target_text'), '暂无明确对象')}\n"
        f"情绪分布摘要：{emotion_distribution}\n"
        f"立场分布摘要：{stance_distribution}\n\n"
        "【事实边界约束】\n"
        "请只基于上面提供的事件信息和下面的记忆样本生成反应，不要补充未给出的具体事实、人物、机构、调查细节或数字。"
        "如果事件信息不足，可以表达疑问、观望或低强度参与。\n\n"
        "【参与倾向参考】\n"
        f"系统预估该用户对此事件的参与倾向为：{tendency_label}（{tendency_key}）。"
        "程序门控已判定该用户会参与；请在此基础上调节表达强度、行为类型和措辞风格，不要脱离历史风格强行拔高。\n\n"
        "【可参考的历史记忆片段】\n"
        f"{memory_snippets}\n\n"
        "【输出要求】\n"
        "不要出现“根据画像”“作为 Agent”“模型认为”等元话语。\n"
        "participate 必须为 true，action_type 必须为 comment、repost 或 repost_with_comment 之一，reaction_text 不能为空。\n"
        "reaction_text 应控制在 10～80 个中文字符之间，除非该用户历史记忆明显具有长文表达风格。\n"
        "微博式表达可以包含疑问、讽刺、感叹或简短评价，但不要写成评论文章。\n"
        "所有字符串值都必须是合法 JSON 字符串；如需引用短语，请优先使用中文引号“”，不要直接使用未转义的英文双引号。"
    )


class SingleEventSimulator:
    """Run a single-event, single-turn Weibo user Agent simulation."""

    def __init__(
        self,
        paths: SimulatorPaths | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        temperature: float = 0.2,
    ) -> None:
        load_dotenv()
        self.paths = paths or SimulatorPaths()
        self.model_name = model_name or os.environ.get("MODEL_NAME") or DEFAULT_MODEL_NAME
        self.base_url = base_url or os.environ.get("BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.seed = seed
        self.temperature = temperature

    async def run_event(
        self,
        event_id: str,
        max_agents: int | None = None,
        memory_user_level: str | None = None,
        output_dir: Path | None = None,
        overwrite: bool = False,
        resume: bool = True,
        dry_run: bool = False,
        concurrency: int | None = None,
        run_id: str | None = None,
    ) -> Path | None:
        event = get_event_by_id(event_id, self.paths.events_path)
        agents = load_agent_records(
            profiles_path=self.paths.profiles_path,
            memories_path=self.paths.memories_path,
            sys_prompts_path=self.paths.sys_prompts_path,
            memory_user_level=memory_user_level,
            max_agents=max_agents,
            random_seed=self.seed,
        )
        if not agents:
            raise ValueError("No agent records matched the requested filters.")

        if dry_run:
            self._log_dry_run(event, agents[: min(3, len(agents))])
            return None

        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = output_dir or self.paths.output_dir
        run_output_dir = base_output_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        reactions_path = run_output_dir / "agent_reactions.jsonl"
        summary_report_path = run_output_dir / "summary_report.csv"

        completed = self._load_completed_keys(reactions_path) if resume and not overwrite else set()
        if overwrite and reactions_path.exists():
            reactions_path.unlink()

        pending_agents: list[tuple[int, AgentRecord]] = []
        for index, agent in enumerate(agents, start=1):
            key = (event_id, agent.agent_id)
            if key in completed:
                LOGGER.info("Skipping completed result for event=%s agent=%s", event_id, agent.agent_id)
                continue
            pending_agents.append((index, agent))

        if pending_agents:
            concurrency_source = "argument"
            if concurrency is None:
                effective_concurrency = max_agents if max_agents is not None else len(pending_agents)
                concurrency_source = "max_agents" if max_agents is not None else "pending_agents"
            else:
                effective_concurrency = concurrency
            if effective_concurrency < 1:
                raise ValueError("concurrency must be >= 1 or None for unlimited agent-level concurrency.")
            effective_concurrency = min(effective_concurrency, len(pending_agents))
            LOGGER.info(
                "Running %d pending agents with agent-level concurrency=%d (source=%s).",
                len(pending_agents),
                effective_concurrency,
                concurrency_source,
            )
        else:
            effective_concurrency = 1
            LOGGER.info("No pending agents to run for event=%s.", event_id)

        semaphore = asyncio.Semaphore(effective_concurrency)
        participation_rng = random.Random(self.seed)
        participation_gate_by_agent = {
            agent.agent_id: decide_participation(agent.profile, participation_rng)
            for _index, agent in pending_agents
        }

        async def run_agent_with_limit(index: int, agent: AgentRecord) -> dict[str, Any]:
            scheduled_at = time.perf_counter()
            async with semaphore:
                queue_elapsed = time.perf_counter() - scheduled_at
                LOGGER.info(
                    "Running agent %d/%d: %s (queue_wait=%.2fs)",
                    index,
                    len(agents),
                    agent.agent_id,
                    queue_elapsed,
                )
                return await self._run_one_agent(event, agent, run_id, participation_gate_by_agent[agent.agent_id])

        with reactions_path.open("a", encoding="utf-8", newline="\n") as file:
            tasks = [asyncio.create_task(run_agent_with_limit(index, agent)) for index, agent in pending_agents]
            done_count = 0
            for task in asyncio.as_completed(tasks):
                completed_at = time.perf_counter()
                row = await task
                collect_elapsed = time.perf_counter() - completed_at
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                done_count += 1
                LOGGER.info(
                    "Completed pending agent %d/%d: %s (completion_wait=%.2fs)",
                    done_count,
                    len(pending_agents),
                    row.get("agent_id", "unknown"),
                    collect_elapsed,
                )

        analyze_results(reactions_path, summary_report_path)
        LOGGER.info("Simulation output: %s", reactions_path)
        LOGGER.info("Summary report: %s", summary_report_path)
        return reactions_path

    def _log_dry_run(self, event: dict[str, Any], agents: list[AgentRecord]) -> None:
        LOGGER.info("DRY RUN: no model calls will be made.")
        LOGGER.info("Event: %s | %s", event.get("event_id"), event.get("topic"))
        participation_rng = random.Random(self.seed)
        for agent in agents:
            selected = select_memories(event, agent.memories)
            participation_tendency, participation_score = estimate_participation_tendency(agent.profile)
            gate_decision = decide_participation(agent.profile, participation_rng)
            LOGGER.info("Agent: %s user_id=%s level=%s", agent.agent_id, agent.user_id, agent.memory_user_level)
            LOGGER.info("System prompt preview: %s", agent.sys_prompt[:500].replace("\n", "\\n"))
            LOGGER.info(
                "Estimated participation tendency: %s (score=%.3f)",
                PARTICIPATION_TENDENCY_LABELS[participation_tendency],
                participation_score,
            )
            LOGGER.info(
                "Program participation gate: passed=%s probability=%.3f draw=%.3f",
                gate_decision.passed,
                gate_decision.probability,
                gate_decision.draw,
            )
            LOGGER.info("Selected memory count: %d", len(selected))
            for memory in selected:
                LOGGER.info(
                    "Memory [%s]: %s",
                    memory.get("mark", "general_memory"),
                    _safe_text(memory.get("content"))[:220].replace("\n", " "),
                )
            if gate_decision.passed:
                LOGGER.info("Event message:\n%s", build_event_message(event, selected, participation_tendency))
            else:
                LOGGER.info("Program participation gate skipped this agent; no LLM message would be sent.")

    async def _run_one_agent(
        self,
        event: dict[str, Any],
        agent_record: AgentRecord,
        run_id: str,
        participation_gate: ParticipationGateDecision,
    ) -> dict[str, Any]:
        agent_started_at = time.perf_counter()
        selected_memories = select_memories(event, agent_record.memories)
        participation_tendency, _participation_score = estimate_participation_tendency(agent_record.profile)
        base_row = self._build_base_result_row(event, agent_record, run_id, participation_gate)

        if not participation_gate.passed:
            agent_elapsed = time.perf_counter() - agent_started_at
            LOGGER.info(
                "Agent %s skipped by program participation gate in %.2fs (probability=%.3f draw=%.3f).",
                agent_record.agent_id,
                agent_elapsed,
                participation_gate.probability,
                participation_gate.draw,
            )
            return {
                **base_row,
                "participate": False,
                "action_type": "ignore",
                "emotion_label": "mixed",
                "emotion_intensity": 0,
                "stance_label": "neutral",
                "stance_intensity": 0,
                "reaction_text": "",
                "reason": "程序概率门控判定该用户不参与本事件讨论，未调用 LLM 生成内容。",
                "raw_output": json.dumps(
                    {
                        "program_participation_gate": {
                            "probability": round(participation_gate.probability, 4),
                            "draw": round(participation_gate.draw, 4),
                            "passed": participation_gate.passed,
                        },
                    },
                    ensure_ascii=False,
                ),
                "parse_status": "success",
                "error_message": "",
            }

        try:
            event_message = build_event_message(event, selected_memories, participation_tendency)
            raw_output, reaction = await self._call_agentscope(agent_record, event_message, selected_memories)
            agent_elapsed = time.perf_counter() - agent_started_at
            row = {**base_row, **reaction.model_dump()}
            row.update(
                {
                    "raw_output": raw_output,
                    "parse_status": "success",
                    "error_message": "",
                },
            )
            LOGGER.info("Agent %s finished successfully in %.2fs.", agent_record.agent_id, agent_elapsed)
            return row
        except ReactionParseError as exc:
            agent_elapsed = time.perf_counter() - agent_started_at
            LOGGER.exception("Agent %s returned unparseable JSON: %s", agent_record.agent_id, exc)
            LOGGER.info("Agent %s finished with parse_failed in %.2fs.", agent_record.agent_id, agent_elapsed)
            return {
                **base_row,
                "participate": None,
                "action_type": None,
                "emotion_label": None,
                "emotion_intensity": None,
                "stance_label": None,
                "stance_intensity": None,
                "reaction_text": "",
                "reason": "",
                "raw_output": exc.raw_output,
                "parse_status": "parse_failed",
                "error_message": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - failed agents must be persisted
            agent_elapsed = time.perf_counter() - agent_started_at
            LOGGER.exception("Agent %s failed: %s", agent_record.agent_id, exc)
            LOGGER.info("Agent %s finished with failed status in %.2fs.", agent_record.agent_id, agent_elapsed)
            return {
                **base_row,
                "participate": None,
                "action_type": None,
                "emotion_label": None,
                "emotion_intensity": None,
                "stance_label": None,
                "stance_intensity": None,
                "reaction_text": "",
                "reason": "",
                "raw_output": "",
                "parse_status": "failed",
                "error_message": str(exc),
            }

    async def _call_agentscope(
        self,
        agent_record: AgentRecord,
        event_message: str,
        selected_memories: list[dict[str, Any]],
    ) -> tuple[str, ReactionSchema]:
        try:
            from agentscope.agent import ReActAgent
            from agentscope.formatter import OpenAIChatFormatter
            from agentscope.memory import InMemoryMemory
            from agentscope.message import Msg
            from agentscope.model import OpenAIChatModel
        except ImportError as exc:
            raise RuntimeError(
                "AgentScope is not installed in the .gp environment. "
                "Recommended install: conda run -p D:\\GraduationProject\\.gp pip install agentscope"
            ) from exc

        if not self.api_key:
            raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY before running live simulation.")

        memory = InMemoryMemory()
        for memory_item in selected_memories:
            mark = _safe_text(memory_item.get("mark"), "general_memory")
            content = _safe_text(memory_item.get("content"))
            if content:
                await memory.add(Msg("user", f"历史记忆片段：{content}", "user"), marks=mark)

        generate_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            generate_kwargs["seed"] = self.seed

        model = OpenAIChatModel(
            model_name=self.model_name,
            api_key=self.api_key,
            stream=True,
            client_kwargs={"base_url": self.base_url},
            generate_kwargs=generate_kwargs,
        )
        agent = ReActAgent(
            name=agent_record.agent_id,
            sys_prompt=agent_record.sys_prompt,
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=memory,
            max_iters=3,
        )

        msg = Msg("user", event_message, "user")
        first_call_started_at = time.perf_counter()
        response = await agent(msg, structured_model=ReactionSchema)
        first_call_elapsed = time.perf_counter() - first_call_started_at
        structured = getattr(response, "metadata", {}).get("structured_output") if getattr(response, "metadata", None) else None
        raw_output = json.dumps(getattr(response, "metadata", {}), ensure_ascii=False)
        if structured:
            reaction = normalize_structured_output(structured)
            if not _is_participating_reaction(reaction):
                parse_status = "participation_gate_mismatch"
                error_message = "Program gate passed, but model returned a non-participating reaction."
            else:
                LOGGER.info(
                    "Agent %s first model call returned structured output in %.2fs.",
                    agent_record.agent_id,
                    first_call_elapsed,
                )
                return raw_output, reaction
        else:
            raw_text = _safe_text(getattr(response, "content", "")) or raw_output
            reaction, parse_status, error_message = parse_reaction_json(raw_text)
            if reaction is not None and _is_participating_reaction(reaction):
                LOGGER.info(
                    "Agent %s first model call returned parseable participating text in %.2fs.",
                    agent_record.agent_id,
                    first_call_elapsed,
                )
                return raw_text, reaction
            if reaction is not None:
                parse_status = "participation_gate_mismatch"
                error_message = "Program gate passed, but model returned a non-participating reaction."

        if structured:
            raw_text = raw_output
        LOGGER.warning(
            "Agent %s first model call did not produce a valid participating reaction after %.2fs; retrying. status=%s error=%s",
            agent_record.agent_id,
            first_call_elapsed,
            parse_status,
            error_message,
        )
        retry_msg = Msg(
            "user",
            "上一次输出未能满足要求。程序已判定该用户会参与本事件讨论；请只输出一个严格 JSON 对象，"
            "participate 必须为 true，action_type 必须为 comment、repost 或 repost_with_comment，reaction_text 不能为空。"
            "字符串内部如需引用短语，请使用中文引号“”，不要使用未转义的英文双引号。",
            "user",
        )
        retry_call_started_at = time.perf_counter()
        retry_response = await agent(retry_msg, structured_model=ReactionSchema)
        retry_call_elapsed = time.perf_counter() - retry_call_started_at
        retry_structured = (
            getattr(retry_response, "metadata", {}).get("structured_output")
            if getattr(retry_response, "metadata", None)
            else None
        )
        retry_raw = json.dumps(getattr(retry_response, "metadata", {}), ensure_ascii=False)
        if retry_structured:
            retry_reaction = normalize_structured_output(retry_structured)
            if _is_participating_reaction(retry_reaction):
                LOGGER.info(
                    "Agent %s retry model call returned structured participating output in %.2fs.",
                    agent_record.agent_id,
                    retry_call_elapsed,
                )
                return retry_raw, retry_reaction
            raise ReactionParseError(
                "Program gate passed, but retry returned a non-participating structured reaction.",
                raw_output=retry_raw,
            )
        retry_text = _safe_text(getattr(retry_response, "content", "")) or retry_raw
        retry_reaction, retry_status, retry_error = parse_reaction_json(retry_text)
        if retry_reaction is None or not _is_participating_reaction(retry_reaction):
            LOGGER.warning(
                "Agent %s retry model call failed participation validation after %.2fs. status=%s error=%s",
                agent_record.agent_id,
                retry_call_elapsed,
                retry_status,
                retry_error,
            )
            raise ReactionParseError(
                f"Participating JSON parse failed: {retry_error or error_message}; status={retry_status or parse_status}",
                raw_output=retry_text,
            )
        LOGGER.info(
            "Agent %s retry model call returned parseable participating text in %.2fs.",
            agent_record.agent_id,
            retry_call_elapsed,
        )
        return retry_text, retry_reaction

    def _build_base_result_row(
        self,
        event: dict[str, Any],
        agent: AgentRecord,
        run_id: str,
        participation_gate: ParticipationGateDecision,
    ) -> dict[str, Any]:
        profile = agent.profile
        participation_tendency, participation_score = estimate_participation_tendency(profile)
        return {
            "run_id": run_id,
            "event_id": _safe_text(event.get("event_id")),
            "weibo_id": _safe_text(event.get("weibo_id")),
            "topic": _safe_text(event.get("topic")),
            "agent_id": agent.agent_id,
            "user_id": agent.user_id,
            "memory_user_level": agent.memory_user_level,
            "verified_type_name": _safe_text(_nested_get(profile, "base_identity", "verified_type_name")),
            "influence_level": _safe_text(_nested_get(profile, "base_identity", "influence_level")),
            # "propagation_role": _safe_text(_nested_get(profile, "behavior_parameters", "propagation_role"))
            "propagation_role": _safe_text(_nested_get(profile, "base_identity", "propagation_role"))
            or _safe_text(_nested_get(profile, "behavior_parameters", "propagation_role"))
            or _safe_text(_nested_get(profile, "prompt_profile", "propagation_role")),
            "participation_tendency": participation_tendency,
            # "participation_tendency_label": PARTICIPATION_TENDENCY_LABELS[participation_tendency],
            "participation_tendency_score": round(participation_score, 4),
            "participation_gate_probability": round(participation_gate.probability, 4),
            "participation_gate_draw": round(participation_gate.draw, 4),
            "participation_gate_passed": participation_gate.passed,
            "model_name": self.model_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _load_completed_keys(path: Path) -> set[tuple[str, str]]:
        completed: set[tuple[str, str]] = set()
        if not path.exists():
            return completed
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if _safe_text(record.get("parse_status")) != "success":
                    continue
                event_id = _safe_text(record.get("event_id"))
                agent_id = _safe_text(record.get("agent_id"))
                if event_id and agent_id:
                    completed.add((event_id, agent_id))
        return completed


def run_event_sync(simulator: SingleEventSimulator, **kwargs: Any) -> Path | None:
    return asyncio.run(simulator.run_event(**kwargs))
