from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "scope" / "data" / "inputs"
DEFAULT_AGENT_PROFILES_PATH = DEFAULT_DATA_DIR / "agent_profiles.jsonl"
DEFAULT_AGENT_MEMORIES_PATH = DEFAULT_DATA_DIR / "agent_memories.jsonl"
DEFAULT_AGENT_SYS_PROMPTS_PATH = DEFAULT_DATA_DIR / "agent_sys_prompts.jsonl"


@dataclass(frozen=True)
class AgentRecord:
    """Merged simulation input for one Weibo user agent."""

    agent_id: str
    user_id: str
    profile: dict[str, Any]
    memories: list[dict[str, Any]]
    sys_prompt: str
    memory_user_level: str
    has_fallback_prompt: bool = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL records."""

    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(record)

    LOGGER.info("Loaded %d JSONL records from %s", len(records), path)
    return records


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _get_memory_user_level(profile: dict[str, Any], memory_record: dict[str, Any] | None) -> str:
    if memory_record:
        level = _as_text(memory_record.get("memory_user_level"))
        if level:
            return level
    base_identity = profile.get("base_identity") if isinstance(profile.get("base_identity"), dict) else {}
    level = _as_text(base_identity.get("memory_user_level"))
    return level or "unknown"


def build_fallback_sys_prompt(profile: dict[str, Any]) -> str:
    """Render a conservative system prompt when agent_sys_prompts is missing."""

    prompt_profile = profile.get("prompt_profile") if isinstance(profile.get("prompt_profile"), dict) else {}
    identity_summary = _as_text(prompt_profile.get("identity_summary"), "该用户画像信息较少。")
    emotion_summary = _as_text(prompt_profile.get("emotion_summary"), "暂无明确长期情绪倾向。")
    topic_summary = _as_text(prompt_profile.get("topic_summary"), "暂无明确主题兴趣。")
    propagation_summary = _as_text(prompt_profile.get("propagation_summary"), "暂无明确传播习惯。")

    return (
        "你正在扮演一个微博用户 Agent。你需要根据该用户的长期画像、主题兴趣、传播习惯和代表性记忆，"
        "对给定热点事件或微博语境做出符合该用户特征的反应。\n\n"
        f"【基础身份】\n{identity_summary}\n\n"
        f"【长期情绪倾向】\n{emotion_summary}\n\n"
        f"【主题兴趣】\n{topic_summary}\n\n"
        f"【传播行为】\n{propagation_summary}\n\n"
        "【行为要求】\n"
        "1. 你应根据该用户画像判断是否参与讨论，而不是默认每次都发言。\n"
        "2. 如果参与讨论，表达应符合该用户的情绪倾向、主题偏好和传播习惯。\n"
        "3. 不要机械复述画像字段，不要提到“画像”“数据集”“模型”等元信息。\n"
        "4. 输出必须是严格 JSON，不要输出额外解释。"
    )


def load_agent_records(
    profiles_path: Path = DEFAULT_AGENT_PROFILES_PATH,
    memories_path: Path = DEFAULT_AGENT_MEMORIES_PATH,
    sys_prompts_path: Path = DEFAULT_AGENT_SYS_PROMPTS_PATH,
    memory_user_level: str | None = None,
    max_agents: int | None = None,
    use_fallback_prompt: bool = True,
) -> list[AgentRecord]:
    """Load and merge profile, memory and system prompt records by agent_id."""

    profile_records = load_jsonl(profiles_path)
    memory_records = load_jsonl(memories_path)
    sys_prompt_records = load_jsonl(sys_prompts_path)

    profiles = {_as_text(record.get("agent_id")): record for record in profile_records if _as_text(record.get("agent_id"))}
    memories_by_agent = {
        _as_text(record.get("agent_id")): record
        for record in memory_records
        if _as_text(record.get("agent_id"))
    }
    prompts_by_agent = {
        _as_text(record.get("agent_id")): _as_text(record.get("sys_prompt"))
        for record in sys_prompt_records
        if _as_text(record.get("agent_id"))
    }

    target_level = memory_user_level.strip().lower() if memory_user_level else None
    merged: list[AgentRecord] = []

    for agent_id, profile in profiles.items():
        memory_record = memories_by_agent.get(agent_id)
        prompt = prompts_by_agent.get(agent_id, "")
        level = _get_memory_user_level(profile, memory_record)
        if target_level and level.lower() != target_level:
            continue

        if memory_record is None:
            LOGGER.warning("Agent %s has no memories; continuing with an empty memory list.", agent_id)
            memories: list[dict[str, Any]] = []
        else:
            raw_memories = memory_record.get("memories", [])
            memories = raw_memories if isinstance(raw_memories, list) else []

        has_fallback_prompt = False
        if not prompt:
            if not use_fallback_prompt:
                LOGGER.warning("Agent %s has no sys_prompt and will be skipped.", agent_id)
                continue
            prompt = build_fallback_sys_prompt(profile)
            has_fallback_prompt = True
            LOGGER.warning("Agent %s has no sys_prompt; using a fallback prompt.", agent_id)

        user_id = _as_text(profile.get("user_id")) or _as_text(memory_record.get("user_id") if memory_record else "")
        merged.append(
            AgentRecord(
                agent_id=agent_id,
                user_id=user_id,
                profile=profile,
                memories=memories,
                sys_prompt=prompt,
                memory_user_level=level,
                has_fallback_prompt=has_fallback_prompt,
            ),
        )

        if max_agents is not None and len(merged) >= max_agents:
            break

    LOGGER.info(
        "Prepared %d agent records from %d profiles; memory_user_level=%s max_agents=%s",
        len(merged),
        len(profiles),
        memory_user_level,
        max_agents,
    )
    return merged
