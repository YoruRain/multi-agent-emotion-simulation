"""Async DeepSeek API calls, response parsing, and schema validation."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

import httpx
from pydantic import ValidationError

from comment_analysis.config import AnalysisConfig
from comment_analysis.data_io import CommentSample, as_clean_str, make_jsonl_record, now_iso
from comment_analysis.schema import (
    SCHEMA_INSTRUCTION,
    SYSTEM_PROMPT,
    USER_PROMPT,
    CommentAnalysis,
    get_comment_analysis_schema_json,
)


def endpoint_from_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def build_input_payload(sample: CommentSample) -> dict[str, str]:
    return {
        "comment_id": sample.comment_id,
        "topic": sample.topic,
        "source_weibo_summary": sample.analysis_context,
        "parent_comment_summary": sample.parent_comment_text,
        "parent_comment_text": sample.parent_comment_text,
        "comment_text": sample.content,
    }


def build_request_body(config: AnalysisConfig, sample: CommentSample) -> dict[str, Any]:
    input_json = json.dumps(build_input_payload(sample), ensure_ascii=False, indent=2)
    user_prompt = "\n\n".join(
        [
            USER_PROMPT.format(input_json=input_json),
            SCHEMA_INSTRUCTION.format(schema_json=get_comment_analysis_schema_json()),
        ]
    )
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "stream": False,
    }
    if config.response_format_json:
        body["response_format"] = {"type": "json_object"}
    return body


def extract_message_content(payload: dict[str, Any]) -> str:
    if "error" in payload:
        raise RuntimeError(f"DeepSeek API error: {payload['error']}")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected DeepSeek response shape: {payload}") from exc
    content = as_clean_str(content)
    if not content:
        raise RuntimeError("DeepSeek returned an empty message content.")
    return content


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def find_balanced_json_object(text: str) -> str | None:
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


def parse_json_response(content: str) -> dict[str, Any]:
    candidates = [content, strip_code_fence(content)]
    balanced = find_balanced_json_object(content)
    if balanced:
        candidates.append(balanced)

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append(f"JSON root is {type(parsed).__name__}, expected object")

    raise ValueError(f"Could not parse a valid JSON object. Errors: {' | '.join(errors)}")


def normalize_emotion_label(raw: dict[str, Any]) -> dict[str, Any]:
    task = raw.get("task_output_layer") or {}
    appraisal = raw.get("appraisal_layer") or {}

    label = task.get("emotion_label")
    if not isinstance(label, str):
        return raw

    label = label.strip().lower()
    intensity = task.get("emotion_intensity", 0)
    norm_violation = appraisal.get("norm_violation")

    if label == "disapproval":
        # disapproval is disfavor/opposition and is mainly represented by stance_label=against.
        # Avoid inventing an emotion when the emotional intensity is low.
        if isinstance(intensity, int) and intensity <= 1:
            task["emotion_label"] = "none"
        elif norm_violation == "high":
            task["emotion_label"] = "disgust"
        else:
            task["emotion_label"] = "disappointment"

    elif label == "sarcasm":
        # sarcasm is an expression style, not an emotion.
        task["emotion_label"] = "mixed"
        if task.get("argument_type") in {None, "", "unclear"}:
            task["argument_type"] = "sarcasm"

    elif label == "amusement":
        # If mocking/amusement becomes analytically important, consider adding it formally later.
        task["emotion_label"] = "joy"

    elif label == "agreement":
        # agreement is an argument_type, not an emotion_label.
        task["emotion_label"] = "none"
        if task.get("argument_type") in {None, "", "unclear"}:
            task["argument_type"] = "agreement"
        if task.get("stance_label") in {None, "", "unclear"}:
            task["stance_label"] = "favor"

    elif label == "frustration":
        # Frustration/impatience/irritation is usually folded into disappointment or anger.
        if isinstance(intensity, int) and intensity >= 3:
            task["emotion_label"] = "anger"
        else:
            task["emotion_label"] = "disappointment"

    return raw


def validate_analysis(raw_data: dict[str, Any], expected_comment_id: str) -> CommentAnalysis:
    try:
        result = CommentAnalysis.model_validate(raw_data)
    except AttributeError:
        result = CommentAnalysis.parse_obj(raw_data)

    if result.comment_id != expected_comment_id:
        raise ValueError(
            f"Model returned comment_id={result.comment_id!r}, "
            f"expected {expected_comment_id!r}"
        )
    return result


async def call_deepseek(
    client: httpx.AsyncClient,
    config: AnalysisConfig,
    sample: CommentSample,
) -> str:
    endpoint = endpoint_from_base(config.api_base)
    body = build_request_body(config, sample)
    response = await client.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=body,
    )

    if (
        response.status_code == 400
        and config.response_format_json
        and "response_format" in response.text.lower()
    ):
        logging.warning(
            "API rejected response_format for comment_id=%s; retrying this request without it",
            sample.comment_id,
        )
        body.pop("response_format", None)
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=body,
        )

    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned non-JSON response: {response.text[:1000]}") from exc
    return extract_message_content(payload)


async def analyze_with_retries(
    client: httpx.AsyncClient,
    config: AnalysisConfig,
    sample: CommentSample,
) -> dict[str, Any]:
    last_error: Exception | None = None
    total_attempts = config.max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            content = await call_deepseek(client, config, sample)
            parsed = parse_json_response(content)
            parsed = normalize_emotion_label(parsed)
            analysis = validate_analysis(parsed, sample.comment_id)
            analyzed_at = now_iso()
            return make_jsonl_record(sample, analysis, config, analyzed_at)
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            RuntimeError,
            ValueError,
            ValidationError,
        ) as exc:
            last_error = exc
            if attempt >= total_attempts:
                break
            sleep_seconds = min(
                config.retry_max_delay,
                config.retry_base_delay * (2 ** (attempt - 1)),
            )
            sleep_seconds *= 1 + random.uniform(0, 0.25)
            logging.warning(
                "Retry %s/%s for comment_id=%s after %s: %s",
                attempt,
                config.max_retries,
                sample.comment_id,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error
