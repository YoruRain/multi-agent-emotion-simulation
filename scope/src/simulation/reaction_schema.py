from __future__ import annotations

import json
import re
import ast
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


ActionType = Literal["ignore", "comment", "repost", "repost_with_comment"]
EmotionLabel = Literal["anger", "sadness", "fear", "joy", "disgust", "disappointment", "surprise", "sympathy", "confusion", "admiration", "mixed"]
StanceLabel = Literal["favor", "against", "neutral", "mixed", "unclear"]
STRING_FIELD_ORDER = [
    "action_type",
    "emotion_label",
    "stance_label",
    "reaction_text",
    "reason",
]


class ReactionSchema(BaseModel):
    """Validated model output for one Weibo user reaction."""

    participate: bool = Field(description="Whether the user participates in the event discussion")
    action_type: ActionType = Field(description="The user's visible action")
    emotion_label: EmotionLabel = Field(description="Main emotion label")
    emotion_intensity: int = Field(ge=0, le=2, description="Emotion intensity: 0, 1, or 2")
    stance_label: StanceLabel = Field(description="Main stance label")
    stance_intensity: int = Field(ge=0, le=2, description="Stance intensity: 0, 1, or 2")
    reaction_text: str = Field(description="Natural Weibo-style reaction text")
    reason: str = Field(description="Short experiment-analysis reason")

    @model_validator(mode="after")
    def validate_ignore_consistency(self) -> "ReactionSchema":
        if not self.participate:
            if self.action_type != "ignore":
                raise ValueError("action_type must be 'ignore' when participate is false")
            if self.reaction_text != "":
                raise ValueError("reaction_text must be empty when participate is false")
            if self.emotion_intensity != 0 or self.stance_intensity != 0:
                raise ValueError("emotion_intensity and stance_intensity must be 0 when participate is false")
        elif self.action_type == "ignore":
            raise ValueError("action_type cannot be 'ignore' when participate is true")
        return self


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from text."""

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_text_from_content_blocks(raw_output: str) -> list[str]:
    """Extract text fields from AgentScope/OpenAI-style content blocks.

    AgentScope message content can be a Python list representation like
    ``[{'type': 'text', 'text': '{...}'}]`` after ``str(content)``. That wrapper
    is not JSON, but the inner ``text`` value can contain valid JSON.
    """

    stripped = raw_output.strip()
    if not stripped.startswith("["):
        return []

    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    texts: list[str] = []
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            texts.append(item["text"])
    return texts


def _escape_unescaped_quotes(value: str) -> str:
    escaped_chars: list[str] = []
    for index, char in enumerate(value):
        if char != '"':
            escaped_chars.append(char)
            continue

        backslash_count = 0
        cursor = index - 1
        while cursor >= 0 and value[cursor] == "\\":
            backslash_count += 1
            cursor -= 1
        escaped_chars.append('\\"' if backslash_count % 2 == 0 else char)
    return "".join(escaped_chars)


def _repair_unescaped_string_quotes(candidate: str) -> str | None:
    """Escape bare quotes inside known string fields from otherwise JSON-like output."""

    repaired = candidate
    changed = False
    for index, field in enumerate(STRING_FIELD_ORDER):
        next_field = STRING_FIELD_ORDER[index + 1] if index + 1 < len(STRING_FIELD_ORDER) else None
        if next_field:
            pattern = rf'("{field}"\s*:\s*")(.*?)("\s*,\s*"{next_field}"\s*:)'
        else:
            pattern = rf'("{field}"\s*:\s*")(.*?)("\s*\}})'

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            value = match.group(2)
            escaped_value = _escape_unescaped_quotes(value)
            if escaped_value != value:
                changed = True
            return f"{match.group(1)}{escaped_value}{match.group(3)}"

        repaired = re.sub(pattern, replace, repaired, count=1, flags=re.DOTALL)

    return repaired if changed else None


def parse_reaction_json(raw_output: str) -> tuple[ReactionSchema | None, str, str | None]:
    """Parse and validate model output.

    Returns:
        (reaction, parse_status, error_message)
    """

    text = _strip_markdown_fence(raw_output)
    candidates = [text]
    for block_text in _extract_text_from_content_blocks(text):
        stripped_block_text = _strip_markdown_fence(block_text)
        candidates.append(stripped_block_text)

    for candidate_text in list(candidates):
        extracted = _extract_first_json_object(candidate_text)
        if extracted and extracted != candidate_text:
            candidates.append(extracted)
            repaired_extracted = _repair_unescaped_string_quotes(extracted)
            if repaired_extracted and repaired_extracted != extracted:
                candidates.append(repaired_extracted)
        repaired = _repair_unescaped_string_quotes(candidate_text)
        if repaired and repaired != candidate_text:
            candidates.append(repaired)

    last_error: str | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            return ReactionSchema.model_validate(payload), "success", None
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = str(exc)

    return None, "parse_failed", last_error or "Unable to parse model output as ReactionSchema JSON"


def normalize_structured_output(value: Any) -> ReactionSchema:
    """Normalize AgentScope structured output metadata into ReactionSchema."""

    if isinstance(value, ReactionSchema):
        return value
    return ReactionSchema.model_validate(value)
