from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROJECT_ROOT,
    configure_logging,
    normalize_id,
    parse_list_like,
    parse_mapping_like,
    read_table,
    safe_float,
    safe_get,
    safe_int,
    safe_str,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_INFO_PATH = PROJECT_ROOT / "data" / "high_quality" / "user_info.parquet"
DEFAULT_EMOTION_PATH = PROJECT_ROOT / "data" / "profile" / "weibos" / "emotion_profile" / "user_emotion_profile.parquet"
DEFAULT_TOPIC_PATH = PROJECT_ROOT / "data" / "profile" / "weibos" / "subject_profile" / "user_topic_profile_final.parquet"
DEFAULT_PROPAGATION_PATH = (
    PROJECT_ROOT / "data" / "profile" / "weibos" / "propagation_profile" / "user_propagation_profile.parquet"
)
DEFAULT_MEMORY_SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "profile" / "weibos" / "memory_sample" / "user_memory_summary.parquet"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "scope" / "agent_profiles.jsonl"

BASE_COLUMNS = [
    "user_id",
    "screen_name",
    "gender",
    "verified_type_name",
    "description",
    "user_value_label",
]
EMOTION_COLUMNS = [
    "user_id",
    "pos_ratio",
    "neu_ratio",
    "neg_ratio",
    "avg_intensity_score",
    "strong_emotion_ratio",
    "emotion_profile_summary",
]
TOPIC_COLUMNS = [
    "user_id",
    "final_public_issue_topic_ratio",
    "final_entertainment_topic_ratio",
    "final_daily_life_topic_ratio",
    "topic_summary",
]
PROPAGATION_COLUMNS = [
    "user_id",
    "follower_level",
    "propagation_activity_level",
    "repost_ratio",
    "repost_with_comment_ratio",
    "media_dependency_score",
    "kol_sensitivity_score",
    "influence_score",
    "influence_level",
    "propagation_role",
    "propagation_summary",
]
MEMORY_COLUMNS = [
    "user_id",
    "memory_user_level",
    "selected_memory_count",
    "memory_type_counts",
    "selected_weibo_ids",
    "memory_summary_for_agent",
]
NUMERIC_FIELDS = [
    "pos_ratio",
    "neg_ratio",
    "neu_ratio",
    "avg_intensity_score",
    "strong_emotion_ratio",
    "final_public_issue_topic_ratio",
    "final_entertainment_topic_ratio",
    "final_daily_life_topic_ratio",
    "repost_ratio",
    "repost_with_comment_ratio",
    "media_dependency_score",
    "kol_sensitivity_score",
    "influence_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized Weibo Agent profiles.")
    parser.add_argument("--user-info-path", type=Path, default=DEFAULT_USER_INFO_PATH)
    parser.add_argument("--emotion-path", type=Path, default=DEFAULT_EMOTION_PATH)
    parser.add_argument("--topic-path", type=Path, default=DEFAULT_TOPIC_PATH)
    parser.add_argument("--propagation-path", type=Path, default=DEFAULT_PROPAGATION_PATH)
    parser.add_argument("--memory-summary-path", type=Path, default=DEFAULT_MEMORY_SUMMARY_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def prepare_frame(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    prepared = df.copy()
    if "user_id" not in prepared.columns:
        LOGGER.warning("%s 缺少 user_id 字段，将返回空表", name)
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in prepared.columns:
            LOGGER.warning("%s 缺少字段 %s，将使用默认值", name, column)
            prepared[column] = pd.NA
    prepared = prepared[columns].copy()
    prepared["user_id"] = prepared["user_id"].map(normalize_id)
    prepared = prepared[prepared["user_id"] != ""].drop_duplicates(subset=["user_id"], keep="first")
    LOGGER.info("%s 准备完成: 行数=%d", name, len(prepared))
    return prepared


def build_identity_summary(row: pd.Series) -> str:
    verified = safe_str(safe_get(row, "verified_type_name", "普通微博用户"), "普通微博用户")
    follower_level = safe_str(safe_get(row, "follower_level", "未知"), "未知")
    activity_level = safe_str(safe_get(row, "propagation_activity_level", "未知"), "未知")
    role = safe_str(safe_get(row, "propagation_role", "普通参与者"), "普通参与者")
    return f"该用户是{verified}，粉丝规模{follower_level}，活跃程度{activity_level}，主要作为{role}出现。"


def build_emotion_summary(row: pd.Series) -> str:
    return safe_str(safe_get(row, "emotion_profile_summary"), "暂无相关情绪画像信息。")


def build_topic_summary(row: pd.Series) -> str:
    return safe_str(safe_get(row, "topic_summary"), "暂无相关主题画像信息。")


def build_propagation_summary(row: pd.Series) -> str:
    return safe_str(safe_get(row, "propagation_summary"), "暂无相关传播画像信息。")


def merge_profiles(
    user_info: pd.DataFrame,
    emotion: pd.DataFrame,
    topic: pd.DataFrame,
    propagation: pd.DataFrame,
    memory: pd.DataFrame,
) -> pd.DataFrame:
    merged = user_info.copy()
    for name, frame in [
        ("emotion", emotion),
        ("topic", topic),
        ("propagation", propagation),
        ("memory", memory),
    ]:
        before = len(merged)
        merged = merged.merge(frame, on="user_id", how="left", indicator=f"{name}_merge")
        missing_count = int((merged[f"{name}_merge"] == "left_only").sum())
        LOGGER.info("%s 画像缺失数量=%d / %d", name, missing_count, before)
        merged = merged.drop(columns=[f"{name}_merge"])
    LOGGER.info("合并后的用户数=%d", len(merged))
    return merged


def build_profile_record(row: pd.Series) -> dict[str, Any]:
    user_id = normalize_id(safe_get(row, "user_id"))
    return {
        "agent_id": f"weibo_user_{user_id}",
        "user_id": user_id,
        "base_identity": {
            "screen_name": safe_str(safe_get(row, "screen_name"), "未知"),
            "gender": safe_str(safe_get(row, "gender"), "未知"),
            "verified_type_name": safe_str(safe_get(row, "verified_type_name"), "未知"),
            "memory_user_level": safe_str(safe_get(row, "memory_user_level"), "background"),
            "user_value_label": safe_str(safe_get(row, "user_value_label"), "未知"),
            "influence_level": safe_str(safe_get(row, "influence_level"), "未知"),
        },
        "prompt_profile": {
            "identity_summary": build_identity_summary(row),
            "emotion_summary": build_emotion_summary(row),
            "topic_summary": build_topic_summary(row),
            "propagation_summary": build_propagation_summary(row),
        },
        "behavior_parameters": {field: safe_float(safe_get(row, field), 0.0) for field in NUMERIC_FIELDS},
        "metadata": {
            "selected_memory_count": safe_int(safe_get(row, "selected_memory_count"), 0),
            "memory_type_counts": parse_mapping_like(safe_get(row, "memory_type_counts")),
            "selected_weibo_ids": [normalize_id(item) for item in parse_list_like(safe_get(row, "selected_weibo_ids"))],
        },
    }


def build_agent_profiles(args: argparse.Namespace) -> list[dict[str, Any]]:
    user_info = prepare_frame(read_table(args.user_info_path), BASE_COLUMNS, "user_info")
    emotion = prepare_frame(read_table(args.emotion_path), EMOTION_COLUMNS, "emotion_profile")
    topic = prepare_frame(read_table(args.topic_path), TOPIC_COLUMNS, "topic_profile")
    propagation = prepare_frame(read_table(args.propagation_path), PROPAGATION_COLUMNS, "propagation_profile")
    memory = prepare_frame(read_table(args.memory_summary_path), MEMORY_COLUMNS, "memory_summary")

    merged = merge_profiles(user_info, emotion, topic, propagation, memory)
    records = [build_profile_record(row) for _, row in merged.iterrows()]
    LOGGER.info("Agent Profile 构建完成: %d 条", len(records))
    return records


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, "agent_profile_builder.log")
    records = build_agent_profiles(args)
    write_jsonl(records, args.output_path)
    LOGGER.info("输出文件路径: %s", args.output_path)


if __name__ == "__main__":
    main()
