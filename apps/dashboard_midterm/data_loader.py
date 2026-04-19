from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils import coerce_datetime_columns, ensure_columns, normalize_topic, safe_divide, to_path


DEFAULT_DATA_DIR = Path(r"D:\GraduationProject\data\cleaned")

DATASET_FILE_CANDIDATES: dict[str, list[str]] = {
    "topic_weibo": ["topic_weibo.parquet", "df_topic_weibo.parquet", "topic_weibo.csv", "df_topic_weibo.csv"],
    "topic_comment": [
        "topic_comment.parquet",
        "df_topic_comment.parquet",
        "topic_comment.csv",
        "df_topic_comment.csv",
    ],
    "user_info": ["user_info.parquet", "df_user_info.parquet", "user_info.csv", "df_user_info.csv"],
    "user_weibo": ["user_weibo.parquet", "df_user_weibo.parquet", "user_weibo.csv", "df_user_weibo.csv"],
}

TIME_COLUMNS = ["create_time", "trending_date", "registration_time"]
PRIMARY_KEYS = {
    "topic_weibo": ["weibo_id"],
    "topic_comment": ["comment_id"],
    "user_info": ["user_id"],
    "user_weibo": ["weibo_id", "user_id", "create_time"],
}


def resolve_dataset_paths(data_dir: str | Path) -> dict[str, Path]:
    base_dir = to_path(data_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{base_dir}")

    dataset_paths: dict[str, Path] = {}
    for dataset_name, candidates in DATASET_FILE_CANDIDATES.items():
        matched = next((base_dir / file_name for file_name in candidates if (base_dir / file_name).exists()), None)
        if matched is None:
            candidate_text = "、".join(candidates)
            raise FileNotFoundError(f"未找到 {dataset_name} 对应文件，请检查：{candidate_text}")
        dataset_paths[dataset_name] = matched

    return dataset_paths


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"暂不支持的数据格式：{path.name}")


def normalize_dataframe(dataset_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    normalized = coerce_datetime_columns(frame, TIME_COLUMNS)
    primary_key = [column for column in PRIMARY_KEYS.get(dataset_name, []) if column in normalized.columns]
    if primary_key:
        normalized = normalized.drop_duplicates(subset=primary_key, keep="first")

    if dataset_name == "topic_weibo":
        normalized = ensure_columns(
            normalized,
            [
                "weibo_id",
                "topic",
                "screen_name",
                "content",
                "comment_count",
                "comment_crawled_count",
                "comment_hq_count",
                "comment_hq_ratio",
                "engagement",
                "topic_value_label",
                "trending_type",
                "trending_date",
                "trending_click",
            ],
        )
        normalized["topic_display"] = normalized["topic"].apply(normalize_topic)
        normalized["trending_type_display"] = normalized["trending_type"].fillna("未标注")
        normalized["topic_value_label_display"] = normalized["topic_value_label"].fillna("未标注")

    if dataset_name == "topic_comment":
        normalized = ensure_columns(
            normalized,
            [
                "comment_id",
                "weibo_id",
                "parent_id",
                "user_id",
                "screen_name",
                "content",
                "create_time",
                "like_count",
                "sub_comment_count",
                "engagement",
                "text_quality_label",
                "ip_location",
            ],
        )
        normalized["text_quality_label_display"] = normalized["text_quality_label"].fillna("未标注")

    if dataset_name == "user_info":
        normalized = ensure_columns(
            normalized,
            [
                "user_id",
                "screen_name",
                "gender",
                "ip_location",
                "verified",
                "verified_type_name",
                "follower_count",
                "following_count",
                "follower_following_ratio",
                "user_rank",
                "user_value_label",
                "active_days",
                "description",
            ],
        )
        normalized["user_value_label_display"] = normalized["user_value_label"].fillna("未标注")

    return normalized


def build_comment_stats(topic_comment: pd.DataFrame) -> pd.DataFrame:
    stat_rows: list[dict[str, Any]] = []
    needed_columns = ensure_columns(
        topic_comment,
        ["weibo_id", "comment_id", "parent_id", "like_count", "sub_comment_count", "text_quality_label"],
    )

    for weibo_id, group in needed_columns.groupby("weibo_id", sort=False):
        comment_ids = set(group["comment_id"].tolist())
        valid_parent_mask = group["parent_id"].isin(comment_ids)
        top_level_mask = ~valid_parent_mask
        replied_parent_ids = set(group.loc[valid_parent_mask, "parent_id"].tolist())
        stat_rows.append(
            {
                "weibo_id": weibo_id,
                "sampled_comment_count": int(len(group)),
                "reply_edge_count": int(valid_parent_mask.sum()),
                "top_level_comment_count": int(top_level_mask.sum()),
                "thread_root_count": int(group.loc[top_level_mask, "comment_id"].nunique()),
                "active_thread_count": int(len(replied_parent_ids)),
                "sampled_like_sum": float(pd.to_numeric(group["like_count"], errors="coerce").fillna(0).sum()),
                "sampled_high_quality_count": int(group["text_quality_label"].fillna("").eq("可分析").sum()),
            }
        )

    return pd.DataFrame(stat_rows)


def build_weibo_profile(topic_weibo: pd.DataFrame, topic_comment: pd.DataFrame) -> pd.DataFrame:
    stats = build_comment_stats(topic_comment)
    profile = topic_weibo.merge(stats, on="weibo_id", how="left")

    numeric_fill_columns = [
        "sampled_comment_count",
        "reply_edge_count",
        "top_level_comment_count",
        "thread_root_count",
        "active_thread_count",
        "sampled_like_sum",
        "sampled_high_quality_count",
        "comment_crawled_count",
        "comment_hq_count",
        "engagement",
        "comment_count",
        "like_count",
        "repost_count",
    ]
    for column in numeric_fill_columns:
        if column in profile.columns:
            profile[column] = pd.to_numeric(profile[column], errors="coerce").fillna(0)

    profile["graph_score"] = (
        profile.get("reply_edge_count", 0) * 4
        + profile.get("sampled_comment_count", 0) * 2
        + profile.get("comment_hq_count", 0) * 3
        + profile.get("engagement", 0).clip(upper=20_000) / 200
    )
    profile["hq_comment_ratio_display"] = profile.apply(
        lambda row: safe_divide(row.get("comment_hq_count"), row.get("comment_crawled_count")),
        axis=1,
    )

    return profile.sort_values(
        by=["graph_score", "comment_crawled_count", "engagement", "comment_count"],
        ascending=False,
    ).reset_index(drop=True)


def build_topic_summary(weibo_profile: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        weibo_profile.groupby("topic_display", dropna=False)
        .agg(
            topic_weibo_count=("weibo_id", "count"),
            comment_crawled_total=("comment_crawled_count", "sum"),
            reply_edge_total=("reply_edge_count", "sum"),
            avg_engagement=("engagement", "mean"),
        )
        .sort_values(["comment_crawled_total", "reply_edge_total", "avg_engagement"], ascending=False)
    )
    return grouped.reset_index()


@st.cache_data(show_spinner=False)
def load_processed_data(data_dir: str) -> dict[str, Any]:
    dataset_paths = resolve_dataset_paths(data_dir)
    data_frames = {
        dataset_name: normalize_dataframe(dataset_name, read_dataframe(path))
        for dataset_name, path in dataset_paths.items()
    }
    weibo_profile = build_weibo_profile(data_frames["topic_weibo"], data_frames["topic_comment"])
    topic_summary = build_topic_summary(weibo_profile)

    return {
        **data_frames,
        "dataset_paths": {name: str(path) for name, path in dataset_paths.items()},
        "weibo_profile": weibo_profile,
        "topic_summary": topic_summary,
    }


def get_comments_for_weibo(topic_comment: pd.DataFrame, weibo_id: int) -> pd.DataFrame:
    comments = topic_comment.loc[topic_comment["weibo_id"] == weibo_id].copy()
    if "create_time" in comments.columns:
        comments = comments.sort_values(["create_time", "like_count"], ascending=[True, False], na_position="last")
    return comments.reset_index(drop=True)


def merge_comment_with_user(comment_frame: pd.DataFrame, user_info: pd.DataFrame) -> pd.DataFrame:
    renamed_comments = comment_frame.rename(
        columns={
            "screen_name": "comment_screen_name",
            "gender": "comment_gender",
            "ip_location": "comment_ip_location",
        }
    )
    selected_user_columns = [
        column
        for column in [
            "user_id",
            "screen_name",
            "gender",
            "ip_location",
            "verified",
            "verified_type_name",
            "follower_count",
            "following_count",
            "follower_following_ratio",
            "user_rank",
            "user_value_label",
            "active_days",
            "description",
        ]
        if column in user_info.columns
    ]
    renamed_users = user_info[selected_user_columns].rename(
        columns={
            "screen_name": "user_screen_name",
            "gender": "user_gender",
            "ip_location": "user_ip_location",
            "verified": "user_verified",
            "verified_type_name": "user_verified_type_name",
            "follower_count": "user_follower_count",
            "following_count": "user_following_count",
            "follower_following_ratio": "user_follower_following_ratio",
            "user_rank": "user_rank_display",
            "user_value_label": "user_value_label_display",
            "active_days": "user_active_days",
            "description": "user_description",
        }
    )
    return renamed_comments.merge(renamed_users, on="user_id", how="left")


def get_user_recent_weibo(user_weibo: pd.DataFrame, user_id: int, limit: int = 8) -> pd.DataFrame:
    history = user_weibo.loc[user_weibo["user_id"] == user_id].copy()
    if history.empty:
        return history
    if "create_time" in history.columns:
        history = history.sort_values("create_time", ascending=False, na_position="last")
    selected_columns = [
        column
        for column in [
            "create_time",
            "content",
            "text_quality_label",
            "like_count",
            "comment_count",
            "repost_count",
            "engagement",
            "topics",
            "is_repost",
        ]
        if column in history.columns
    ]
    return history[selected_columns].head(limit).reset_index(drop=True)
