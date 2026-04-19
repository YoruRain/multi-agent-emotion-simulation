from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DISPLAY_FONT_FAMILY = "Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif"


def shorten_text(value: Any, max_length: int = 48) -> str:
    """将长文本裁剪为适合界面展示的长度。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "暂无内容"
    text = str(value).strip()
    if not text:
        return "暂无内容"
    return text if len(text) <= max_length else f"{text[: max_length - 1]}…"


def safe_text(value: Any, default: str = "暂无") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    return text if text else default


def format_count(value: Any) -> str:
    """将数值格式化为适合中文界面的简洁写法。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(number) >= 100_000_000:
        return f"{number / 100_000_000:.2f}亿"
    if abs(number) >= 10_000:
        return f"{number / 10_000:.2f}万"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}"


def format_ratio(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if 0 <= number <= 1:
        return f"{number:.1%}"
    return f"{number:.2f}"


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """补齐缺失列，避免后续展示逻辑报错。"""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result


def coerce_datetime_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def normalize_topic(value: Any) -> str:
    topic = safe_text(value, default="未标注话题")
    return topic if topic != "0" else "未标注话题"


def to_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def build_weibo_option(row: pd.Series) -> str:
    topic = shorten_text(row.get("topic_display"), 20)
    author = safe_text(row.get("screen_name"))
    content = shorten_text(row.get("content"), 34)
    crawled_comments = format_count(row.get("comment_crawled_count", 0))
    return f"{topic} | @{author} | 已采样评论 {crawled_comments} | {content}"


def build_comment_option(row: pd.Series) -> str:
    user_name = safe_text(row.get("screen_name"))
    like_count = format_count(row.get("like_count", 0))
    content = shorten_text(row.get("content"), 32)
    return f"{row.get('comment_id')} | @{user_name} | 赞 {like_count} | {content}"


def safe_divide(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_value == 0:
        return None
    return numerator_value / denominator_value
