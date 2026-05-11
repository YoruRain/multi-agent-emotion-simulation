from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROJECT_ROOT,
    configure_logging,
    normalize_id,
    parse_bool,
    parse_list_like,
    read_table,
    safe_float,
    safe_get,
    safe_str,
    stringify_list,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "profile" / "weibos" / "memory_sample" / "user_weibo_memory_sample.parquet"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "scope" / "data" / "inputs" / "agent_memories.jsonl"
PUBLIC_ISSUE_CATEGORIES = {"社会公共事件", "政策民生", "时事政治"}
LEVEL_LIMITS = {"core": 6, "normal": 3, "background": 1}

REQUIRED_COLUMNS = [
    "user_id",
    "weibo_id",
    "memory_user_level",
    "memory_type",
    "content_for_agent",
    "topic_tags_for_agent",
    "mentions_for_agent",
    "source_context_for_agent",
    "is_repost",
    "has_repost_comment",
    "sentiment_label",
    "final_topic_categories",
    "final_topic_labels",
    "topic_signal_source",
    "source_author_type",
    "engagement_score",
    "selection_reason",
    "memory_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Weibo Agent memory JSONL.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def prepare_memory_samples(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "user_id" not in prepared.columns:
        raise ValueError("记忆样本缺少 user_id 字段，无法按 Agent 分组")

    for column in REQUIRED_COLUMNS:
        if column not in prepared.columns:
            LOGGER.warning("memory_sample 缺少字段 %s，将使用默认值", column)
            prepared[column] = pd.NA

    prepared["user_id"] = prepared["user_id"].map(normalize_id)
    prepared["weibo_id"] = prepared["weibo_id"].map(normalize_id)
    prepared = prepared[prepared["user_id"] != ""].copy()
    prepared["memory_score"] = prepared["memory_score"].map(lambda value: safe_float(value, 0.0))
    prepared["is_repost"] = prepared["is_repost"].map(parse_bool)
    prepared["has_repost_comment"] = prepared["has_repost_comment"].map(parse_bool)
    LOGGER.info("读取记忆样本数=%d", len(prepared))
    return prepared


def normalize_level(value: Any) -> str:
    level = safe_str(value, "background").lower()
    return level if level in LEVEL_LIMITS else "background"


def build_memory_mark(row: pd.Series) -> str:
    memory_type = safe_str(safe_get(row, "memory_type"), "").lower()
    categories = {safe_str(item, "") for item in parse_list_like(safe_get(row, "final_topic_categories"))}
    if "style" in memory_type or "表达风格" in memory_type:
        return "style_memory"
    if categories & PUBLIC_ISSUE_CATEGORIES:
        return "public_issue_memory"
    if parse_bool(safe_get(row, "is_repost")):
        return "propagation_memory"
    return "general_memory"


def build_memory_content(row: pd.Series) -> str:
    categories = parse_list_like(safe_get(row, "final_topic_categories"))
    topic_label = safe_str(categories[0], "未知主题") if categories else "未知主题"
    sentiment = safe_str(safe_get(row, "sentiment_label"), "中性")
    content = safe_str(safe_get(row, "content_for_agent"), "暂无可用历史内容")

    is_repost = parse_bool(safe_get(row, "is_repost"))
    has_comment = parse_bool(safe_get(row, "has_repost_comment"))
    if not is_repost:
        action = "原创"
    elif has_comment:
        action = "转发附评"
    else:
        action = "纯转发"

    memory_text = f"历史微博样本：[{action}｜{topic_label}｜{sentiment}] {content}"
    source_context = safe_str(safe_get(row, "source_context_for_agent"), "")
    if source_context:
        memory_text = f"{memory_text}\n源微博背景：{source_context}"
    return memory_text


def build_memory_item(row: pd.Series) -> dict[str, Any]:
    user_id = normalize_id(safe_get(row, "user_id"))
    weibo_id = normalize_id(safe_get(row, "weibo_id"))
    categories = [safe_str(item, "") for item in parse_list_like(safe_get(row, "final_topic_categories"))]
    labels = [safe_str(item, "") for item in parse_list_like(safe_get(row, "final_topic_labels"))]
    return {
        "memory_id": f"weibo_user_{user_id}_memory_{weibo_id}",
        "weibo_id": weibo_id,
        "mark": build_memory_mark(row),
        "content": build_memory_content(row),
        "topics": stringify_list(safe_get(row, "topic_tags_for_agent")),
        "mentions": stringify_list(safe_get(row, "mentions_for_agent")),
        "metadata": {
            "memory_type": safe_str(safe_get(row, "memory_type"), "general"),
            "is_repost": parse_bool(safe_get(row, "is_repost")),
            "has_repost_comment": parse_bool(safe_get(row, "has_repost_comment")),
            "sentiment_label": safe_str(safe_get(row, "sentiment_label"), "中性"),
            "final_topic_categories": [item for item in categories if item],
            "final_topic_labels": [item for item in labels if item],
            "topic_signal_source": safe_str(safe_get(row, "topic_signal_source"), "未知"),
            "source_author_type": safe_str(safe_get(row, "source_author_type"), "未知"),
            "selection_reason": safe_str(safe_get(row, "selection_reason"), "暂无筛选说明"),
        },
    }


def build_agent_memory_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    level_counter: Counter[str] = Counter()
    mark_counter: Counter[str] = Counter()

    for user_id, group in df.groupby("user_id", sort=True):
        level = normalize_level(safe_get(group.iloc[0], "memory_user_level", "background"))
        level_counter[level] += 1
        limit = LEVEL_LIMITS[level]
        selected = group.sort_values("memory_score", ascending=False).head(limit)
        memories = [build_memory_item(row) for _, row in selected.iterrows()]
        mark_counter.update(memory["mark"] for memory in memories)
        records.append(
            {
                "agent_id": f"weibo_user_{user_id}",
                "user_id": user_id,
                "memory_user_level": level,
                "memories": memories,
            }
        )

    LOGGER.info("用户数=%d", len(records))
    LOGGER.info("core/normal/background 用户数=%s", dict(level_counter))
    LOGGER.info("每类 mark 数量=%s", dict(mark_counter))
    return records


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, "agent_memory_builder.log")
    samples = prepare_memory_samples(read_table(args.input_path))
    records = build_agent_memory_records(samples)
    write_jsonl(records, args.output_path)
    LOGGER.info("输出文件路径: %s", args.output_path)


if __name__ == "__main__":
    main()
