from __future__ import annotations

import asyncio
import json
import logging
import os
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


def build_event_message(event: dict[str, Any], memories: list[dict[str, Any]]) -> str:
    """Build the user message sent to a Weibo user Agent."""

    emotion_distribution = _summarize_distribution(event.get("emotion_distribution"))
    stance_distribution = _summarize_distribution(event.get("stance_distribution"))
    memory_snippets = _format_memory_snippets(memories)

    return (
        "请根据你的长期特征和历史表达习惯，判断你是否会参与下面这个微博热点事件的讨论，并只输出严格 JSON。\n\n"
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
        "【可参考的历史记忆片段】\n"
        f"{memory_snippets}\n\n"
        "【输出要求】\n"
        "不要出现“根据画像”“作为 Agent”“模型认为”等元话语。"
        "如果你不参与，reaction_text 必须为空字符串。"
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
        concurrency: int = 1,
        run_id: str | None = None,
    ) -> Path | None:
        event = get_event_by_id(event_id, self.paths.events_path)
        agents = load_agent_records(
            profiles_path=self.paths.profiles_path,
            memories_path=self.paths.memories_path,
            sys_prompts_path=self.paths.sys_prompts_path,
            memory_user_level=memory_user_level,
            max_agents=max_agents,
        )
        if not agents:
            raise ValueError("No agent records matched the requested filters.")

        if dry_run:
            self._log_dry_run(event, agents[: min(3, len(agents))])
            return None

        if concurrency != 1:
            LOGGER.warning("concurrency=%s was requested, but MVP execution is serial; using concurrency=1.", concurrency)

        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = output_dir or self.paths.output_dir
        run_output_dir = base_output_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        reactions_path = run_output_dir / "agent_reactions.jsonl"
        summary_report_path = run_output_dir / "summary_report.csv"

        completed = self._load_completed_keys(reactions_path) if resume and not overwrite else set()
        if overwrite and reactions_path.exists():
            reactions_path.unlink()

        with reactions_path.open("a", encoding="utf-8", newline="\n") as file:
            for index, agent in enumerate(agents, start=1):
                key = (event_id, agent.agent_id)
                if key in completed:
                    LOGGER.info("Skipping completed result for event=%s agent=%s", event_id, agent.agent_id)
                    continue

                LOGGER.info("Running agent %d/%d: %s", index, len(agents), agent.agent_id)
                row = await self._run_one_agent(event, agent, run_id)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()

        analyze_results(reactions_path, summary_report_path)
        LOGGER.info("Simulation output: %s", reactions_path)
        LOGGER.info("Summary report: %s", summary_report_path)
        return reactions_path

    def _log_dry_run(self, event: dict[str, Any], agents: list[AgentRecord]) -> None:
        LOGGER.info("DRY RUN: no model calls will be made.")
        LOGGER.info("Event: %s | %s", event.get("event_id"), event.get("topic"))
        for agent in agents:
            selected = select_memories(event, agent.memories)
            LOGGER.info("Agent: %s user_id=%s level=%s", agent.agent_id, agent.user_id, agent.memory_user_level)
            LOGGER.info("System prompt preview: %s", agent.sys_prompt[:500].replace("\n", "\\n"))
            LOGGER.info("Selected memory count: %d", len(selected))
            for memory in selected:
                LOGGER.info(
                    "Memory [%s]: %s",
                    memory.get("mark", "general_memory"),
                    _safe_text(memory.get("content"))[:220].replace("\n", " "),
                )
            LOGGER.info("Event message:\n%s", build_event_message(event, selected))

    async def _run_one_agent(self, event: dict[str, Any], agent_record: AgentRecord, run_id: str) -> dict[str, Any]:
        selected_memories = select_memories(event, agent_record.memories)
        event_message = build_event_message(event, selected_memories)
        base_row = self._build_base_result_row(event, agent_record, run_id)

        try:
            raw_output, reaction = await self._call_agentscope(agent_record, event_message, selected_memories)
            row = {**base_row, **reaction.model_dump()}
            row.update(
                {
                    "raw_output": raw_output,
                    "parse_status": "success",
                    "error_message": "",
                },
            )
            return row
        except ReactionParseError as exc:
            LOGGER.exception("Agent %s returned unparseable JSON: %s", agent_record.agent_id, exc)
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
            LOGGER.exception("Agent %s failed: %s", agent_record.agent_id, exc)
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
        response = await agent(msg, structured_model=ReactionSchema)
        structured = getattr(response, "metadata", {}).get("structured_output") if getattr(response, "metadata", None) else None
        raw_output = json.dumps(getattr(response, "metadata", {}), ensure_ascii=False)
        if structured:
            return raw_output, normalize_structured_output(structured)

        raw_text = _safe_text(getattr(response, "content", "")) or raw_output
        reaction, parse_status, error_message = parse_reaction_json(raw_text)
        if reaction is not None:
            return raw_text, reaction

        retry_msg = Msg(
            "user",
            "上一次输出无法解析。请只输出一个严格 JSON 对象，字段和值必须满足指定 schema，不要添加任何额外文字。",
            "user",
        )
        retry_response = await agent(retry_msg, structured_model=ReactionSchema)
        retry_structured = (
            getattr(retry_response, "metadata", {}).get("structured_output")
            if getattr(retry_response, "metadata", None)
            else None
        )
        retry_raw = json.dumps(getattr(retry_response, "metadata", {}), ensure_ascii=False)
        if retry_structured:
            return retry_raw, normalize_structured_output(retry_structured)
        retry_text = _safe_text(getattr(retry_response, "content", "")) or retry_raw
        retry_reaction, retry_status, retry_error = parse_reaction_json(retry_text)
        if retry_reaction is None:
            raise ReactionParseError(
                f"JSON parse failed: {retry_error or error_message}; status={retry_status or parse_status}",
                raw_output=retry_text,
            )
        return retry_text, retry_reaction

    def _build_base_result_row(self, event: dict[str, Any], agent: AgentRecord, run_id: str) -> dict[str, Any]:
        profile = agent.profile
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
            "propagation_role": _safe_text(_nested_get(profile, "behavior_parameters", "propagation_role"))
            or _safe_text(_nested_get(profile, "prompt_profile", "propagation_role")),
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
                event_id = _safe_text(record.get("event_id"))
                agent_id = _safe_text(record.get("agent_id"))
                if event_id and agent_id:
                    completed.add((event_id, agent_id))
        return completed


def run_event_sync(simulator: SingleEventSimulator, **kwargs: Any) -> Path | None:
    return asyncio.run(simulator.run_event(**kwargs))
