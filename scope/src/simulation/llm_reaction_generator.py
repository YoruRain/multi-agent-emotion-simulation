from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from .agent_loader import AgentRecord
from .reaction_schema import ReactionSchema, is_valid_repost_text, normalize_structured_output, parse_reaction_json

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class ReactionParseError(ValueError):
    """Raised when model output cannot be parsed into ReactionSchema."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def is_participating_reaction(reaction: ReactionSchema) -> bool:
    if not reaction.participate or reaction.action_type == "ignore":
        return False
    reaction_text = safe_text(reaction.reaction_text)
    if reaction.action_type == "repost":
        return is_valid_repost_text(reaction_text)
    return bool(reaction_text)


class LLMReactionGenerator:
    """Generate one participating Weibo-style reaction through AgentScope."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        temperature: float = 0.2,
    ) -> None:
        load_dotenv()
        self.model_name = model_name or os.environ.get("MODEL_NAME") or DEFAULT_MODEL_NAME
        self.base_url = base_url or os.environ.get("BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.seed = seed
        self.temperature = temperature

    async def generate(
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
            mark = safe_text(memory_item.get("mark"), "general_memory")
            content = safe_text(memory_item.get("content"))
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
            if not is_participating_reaction(reaction):
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
            raw_text = safe_text(getattr(response, "content", "")) or raw_output
            reaction, parse_status, error_message = parse_reaction_json(raw_text)
            if reaction is not None and is_participating_reaction(reaction):
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
            "participate 必须为 true，action_type 必须为 comment、repost 或 repost_with_comment。"
            "若 action_type 为 repost，reaction_text 只能为空字符串或极短转发占位；若有实质评论，必须使用 repost_with_comment。"
            "若 action_type 为 comment 或 repost_with_comment，reaction_text 不能为空。"
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
            if is_participating_reaction(retry_reaction):
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
        retry_text = safe_text(getattr(retry_response, "content", "")) or retry_raw
        retry_reaction, retry_status, retry_error = parse_reaction_json(retry_text)
        if retry_reaction is None or not is_participating_reaction(retry_reaction):
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
