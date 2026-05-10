from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, configure_logging, read_table, safe_str, write_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "scope" / "agent_profiles.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "scope" / "agent_sys_prompts.jsonl"
DEFAULT_SUMMARY_TEXT = "暂无相关画像信息。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render system prompts from Weibo Agent profiles.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_prompt_profile(record: dict[str, Any]) -> dict[str, Any]:
    prompt_profile = record.get("prompt_profile")
    return prompt_profile if isinstance(prompt_profile, dict) else {}


def render_sys_prompt(prompt_profile: dict[str, Any]) -> str:
    identity_summary = safe_str(prompt_profile.get("identity_summary"), DEFAULT_SUMMARY_TEXT)
    emotion_summary = safe_str(prompt_profile.get("emotion_summary"), DEFAULT_SUMMARY_TEXT)
    topic_summary = safe_str(prompt_profile.get("topic_summary"), DEFAULT_SUMMARY_TEXT)
    propagation_summary = safe_str(prompt_profile.get("propagation_summary"), DEFAULT_SUMMARY_TEXT)

    return f"""你正在扮演一个微博用户 Agent。你需要根据该用户的长期画像、主题兴趣、传播习惯和代表性记忆，对给定热点事件或微博语境做出符合该用户特征的反应。

【基础身份】
{identity_summary}

【长期情绪倾向】
{emotion_summary}

【主题兴趣】
{topic_summary}

【传播行为】
{propagation_summary}

【行为要求】
1. 你应根据该用户画像判断是否参与讨论，而不是默认每次都发言。
2. 如果参与讨论，表达应符合该用户的情绪倾向、主题偏好和传播习惯。
3. 如果该用户对当前事件兴趣较低，或者画像可靠性不足，可以选择低强度反应或不参与。
4. 不要机械复述画像字段，不要提到“画像”“数据集”“模型”等元信息。
5. 输出必须是严格 JSON，不要输出额外解释。

【输出格式】
{{
  "participate": true,
  "action_type": "comment",
  "emotion_label": "neutral",
  "emotion_intensity": 1,
  "stance_label": "neutral",
  "stance_intensity": 1,
  "reaction_text": "这里写该用户可能发表的微博式反应",
  "reason": "用一句话说明为什么该用户会产生这种反应"
}}

字段约束：
- participate: true 或 false
- action_type: "ignore", "comment", "repost", "repost_with_comment"
- 如果 participate 为 false，则 action_type 必须为 "ignore"，reaction_text 必须为空字符串，emotion_intensity 和 stance_intensity 应为 0
- emotion_label: "positive", "neutral", "anger", "sadness", "disgust", "worry", "surprise"
- emotion_intensity: 0, 1, 2
- stance_label: "support", "against", "neutral", "unclear"
- stance_intensity: 0, 1, 2
- reaction_text: 如果 participate 为 false，可以为空字符串
- reason: 简短说明即可"""


def render_record(record: dict[str, Any]) -> dict[str, str]:
    prompt_profile = get_prompt_profile(record)
    user_id = safe_str(record.get("user_id"), "")
    agent_id = safe_str(record.get("agent_id"), f"weibo_user_{user_id}")
    return {
        "agent_id": agent_id,
        "user_id": user_id,
        "sys_prompt": render_sys_prompt(prompt_profile),
    }


def render_prompts(input_path: Path) -> list[dict[str, str]]:
    df = read_table(input_path)
    records = df.to_dict(orient="records")
    LOGGER.info("读取 Agent Profile 数量=%d", len(records))

    rendered = [render_record(record) for record in records]
    LOGGER.info("成功渲染系统 Prompt 数量=%d", len(rendered))
    for sample in rendered[:2]:
        LOGGER.info("样例: %s", sample)
    return rendered


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, "agent_prompt_renderer.log")
    records = render_prompts(args.input_path)
    write_jsonl(records, args.output_path)
    LOGGER.info("输出文件路径: %s", args.output_path)


if __name__ == "__main__":
    main()
