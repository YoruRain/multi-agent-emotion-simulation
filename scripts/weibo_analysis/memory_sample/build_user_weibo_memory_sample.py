from __future__ import annotations

import argparse
import ast
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TOPIC_FUSION_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "user_weibo_topic_fusion.parquet"
)
DEFAULT_USER_WEIBO_PATH = PROJECT_ROOT / "data" / "high_quality" / "user_weibo.parquet"
DEFAULT_CLEANED_USER_WEIBO_PATH = PROJECT_ROOT / "data" / "cleaned" / "user_weibo.parquet"
DEFAULT_WEIBO_EMOTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "emotion_profile"
    / "user_weibo_emotion_analysis.parquet"
)
DEFAULT_USER_EMOTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "emotion_profile"
    / "user_emotion_profile.parquet"
)
DEFAULT_USER_TOPIC_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "user_topic_profile_final.parquet"
)
DEFAULT_USER_PROPAGATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "propagation_profile"
    / "user_propagation_profile.parquet"
)
DEFAULT_USER_INFO_PATH = PROJECT_ROOT / "data" / "high_quality" / "user_info.parquet"
DEFAULT_SOURCE_CREATOR_PATH = PROJECT_ROOT / "data" / "high_quality" / "source_creator_info.parquet"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "memory_sample"
    / "user_weibo_memory_sample.parquet"
)
DEFAULT_USER_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "memory_sample"
    / "user_memory_summary.parquet"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "user_weibo_memory_sample.log"
DEFAULT_PREVIEW_ROWS = 500

TOPIC_FUSION_COLUMNS = [
    "weibo_id",
    "user_id",
    "content",
    "text_quality",
    "is_repost",
    "reposted_weibo_id",
    "source_content",
    "has_repost_comment",
    "user_topics",
    "source_topics",
    "final_topic_categories",
    "final_topic_labels",
    "topic_signal_source",
    "final_topic_confidence",
]
USER_WEIBO_COLUMNS = [
    "weibo_id",
    "user_id",
    "content",
    "cleaned_content",
    "text_length",
    "cleaned_text_length",
    "text_quality",
    "text_quality_label",
    "create_time",
    "year",
    "like_count",
    "comment_count",
    "repost_count",
    "engagement",
    "is_repost",
    "reposted_weibo_id",
    "topics",
    "at_users",
]
WEIBO_EMOTION_COLUMNS = [
    "weibo_id",
    "user_id",
    "sentiment_label_en",
    "sentiment_label",
    "model_confidence",
    "polarity_score",
    "emotion_intensity_score",
]
USER_EMOTION_COLUMNS = [
    "user_id",
    "profile_version",
    "analyzable_weibo_count",
    "pos_ratio",
    "neu_ratio",
    "neg_ratio",
    "dominant_emotion",
    "polarity_std",
    "strong_emotion_ratio",
    "profile_reliability",
]
USER_TOPIC_COLUMNS = [
    "user_id",
    "total_weibo_count",
    "top_implicit_topic_labels",
    "final_category_distribution",
    "final_public_issue_topic_ratio",
    "marketing_topic_ratio",
    "final_topic_coverage",
    "avg_final_topic_confidence",
    "final_topic_profile_reliability",
]
USER_PROPAGATION_COLUMNS = [
    "user_id",
    "weibo_hq_count",
    "active_days",
    "propagation_activity_level",
    "original_ratio",
    "repost_ratio",
    "repost_with_comment_ratio",
    "media_dependency_score",
    "kol_sensitivity_score",
    "high_engagement_weibo_ratio",
    "influence_score",
    "influence_level",
    "propagation_role",
]
USER_INFO_COLUMNS = ["user_id", "user_value_label"]
SOURCE_CREATOR_COLUMNS = ["user_id", "verified_type_name", "follower_count", "user_rank"]

OUTPUT_COLUMNS = [
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
    "polarity_score",
    "emotion_intensity_score",
    "model_confidence",
    "final_topic_categories",
    "final_topic_labels",
    "topic_signal_source",
    "final_topic_confidence",
    "source_author_type",
    "engagement_score",
    "quality_score",
    "emotion_score",
    "topic_score",
    "public_issue_score",
    "propagation_score",
    "engagement_score_norm",
    "style_score",
    "diversity_penalty",
    "memory_score",
    "selection_reason",
]
USER_SUMMARY_COLUMNS = [
    "user_id",
    "memory_user_level",
    "selected_memory_count",
    "memory_type_counts",
    "selected_weibo_ids",
    "memory_summary_for_agent",
]

PUBLIC_ISSUE_CATEGORIES = {"社会公共事件", "政策民生", "时事政治"}
MARKETING_CATEGORY = "广告营销"
LOW_INFORMATION_PATTERNS = [
    re.compile(
        r"^\s*("
        r"[转轉][发發]?(至?微博)?|"
        r"Repost|"
        r"[存码马](住|下|克)?|"
        r"转一个|必须转|"
        r"签到|"
        r"收藏(了)?|"
        r"打卡"
        r")\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^#[^#]+#(\s*#[^#]+#)*$"),
    re.compile(r"^(https?://|网页链接|全文：|查看图片|评论配图)\s*$", re.IGNORECASE),
    re.compile(r"^\d+$"),
    re.compile(r"^[^\w\u4e00-\u9fff\U00010000-\U0010FFFF]+$"),
    re.compile(r"^[a-zA-Z]+$"),
    re.compile(r"^(@[^\s@]+\s*)+$"),
    re.compile(r"^(/\s*/|//\s*)+$"),
]
TEMPLATE_PATTERNS = [
    re.compile(r"抽奖|助力|推荐.*角逐|快来和我一起|转发送出|参与活动|微博视界大会|超话签到"),
    re.compile(r"我正在参与|为.*打call|点击链接|网页链接"),
    re.compile(r"快来一起推荐|分享解锁|推荐次数|短剧节|短剧推荐|宝藏短剧|活动火热进行中"),
]
VIEWPOINT_MARKERS = [
    "我",
    "觉得",
    "认为",
    "希望",
    "真的",
    "喜欢",
    "讨厌",
    "支持",
    "反对",
    "因为",
    "所以",
    "但是",
    "不过",
    "感觉",
    "笑死",
    "烦",
    "开心",
    "难过",
    "离谱",
    "必须",
    "应该",
    "不是",
    "就是",
    "太",
    "好",
]
TONE_MARKERS = [
    "哈哈",
    "笑死",
    "真的",
    "啊",
    "吧",
    "呢",
    "呀",
    "啦",
    "救命",
    "离谱",
    "烦死",
    "太",
    "！",
    "？",
    "...",
    "……",
]


@dataclass(frozen=True)
class MemorySampleConfig:
    min_text_quality: float = 3.0
    min_agent_text_length: int = 8
    min_repost_comment_length: int = 8
    typical_repost_comment_min_length: int = 12
    max_agent_text_length: int = 300
    ideal_text_min_length: int = 15
    ideal_text_max_length: int = 100
    medium_text_max_length: int = 180
    weibo_hq_high_threshold: float = 40.0
    public_issue_high_ratio: float = 0.2834
    marketing_ratio_mid: float = 0.1818
    marketing_ratio_high: float = 0.2941
    marketing_ratio_very_high: float = 0.5424
    strong_user_emotion_ratio: float = 0.10
    topic_reliable_threshold: float = 0.55
    topic_high_threshold: float = 0.75
    topic_low_threshold: float = 0.35
    emotion_usable_confidence: float = 0.72
    emotion_high_confidence: float = 0.99
    emotion_low_confidence: float = 0.56
    strong_weibo_emotion_threshold: float = 0.65
    high_engagement_threshold: float = 84.0
    high_engagement_user_floor: float = 12.0
    media_dependency_high: float = 0.24
    kol_sensitivity_high: float = 0.4375
    category_ratio_threshold: float = 0.10
    label_ratio_threshold: float = 0.03
    similarity_skip_threshold: float = 0.85
    similarity_penalty_threshold: float = 0.65
    min_selected_score: float = 0.25
    min_background_score: float = 0.50
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "quality_score": 0.20,
            "topic_score": 0.20,
            "emotion_score": 0.20,
            "style_score": 0.15,
            "public_issue_score": 0.10,
            "propagation_score": 0.10,
            "engagement_score_norm": 0.05,
        }
    )
    core_quota: tuple[str, ...] = (
        "typical_style",
        "topic_representative",
        "topic_representative",
        "emotion_representative",
        "public_issue",
        "repost_behavior",
    )
    normal_quota: tuple[str, ...] = (
        "typical_style",
        "topic_representative",
        "emotion_representative",
    )
    normal_repost_quota: tuple[str, ...] = (
        "typical_style",
        "topic_representative",
        "repost_behavior",
    )


def configure_logging(verbose: bool = False, log_dir: Path = DEFAULT_LOG_DIR, log_file: Path | None = None) -> Path:
    level = logging.DEBUG if verbose else logging.INFO
    if log_file is None:
        log_file = log_dir / DEFAULT_LOG_FILE_NAME
    elif log_file.suffix == "":
        log_file = log_file / DEFAULT_LOG_FILE_NAME

    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)

    LOGGER.info("Logging to %s", log_file)
    return log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build user-level Weibo memory samples for Agent prompts.")
    parser.add_argument("--topic-fusion-path", type=Path, default=DEFAULT_TOPIC_FUSION_PATH)
    parser.add_argument("--user-weibo-path", type=Path, default=DEFAULT_USER_WEIBO_PATH)
    parser.add_argument("--cleaned-user-weibo-path", type=Path, default=DEFAULT_CLEANED_USER_WEIBO_PATH)
    parser.add_argument("--weibo-emotion-path", type=Path, default=DEFAULT_WEIBO_EMOTION_PATH)
    parser.add_argument("--user-emotion-path", type=Path, default=DEFAULT_USER_EMOTION_PATH)
    parser.add_argument("--user-topic-path", type=Path, default=DEFAULT_USER_TOPIC_PATH)
    parser.add_argument("--user-propagation-path", type=Path, default=DEFAULT_USER_PROPAGATION_PATH)
    parser.add_argument("--user-info-path", type=Path, default=DEFAULT_USER_INFO_PATH)
    parser.add_argument("--source-creator-path", type=Path, default=DEFAULT_SOURCE_CREATOR_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--user-summary-output-path", type=Path, default=DEFAULT_USER_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--limit-users", type=int, default=None, help="Optional deterministic user limit for smoke tests.")
    parser.add_argument("--preview-rows", type=int, default=DEFAULT_PREVIEW_ROWS)
    parser.add_argument("--no-user-summary", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_parquet(path: Path, columns: list[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"{name} must be a parquet file: {path}")

    try:
        df = pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise ImportError("Reading parquet requires pyarrow or fastparquet.") from exc
    except OSError as exc:
        raise OSError(
            f"Failed to read {name}: {path}. "
            "If this mentions parquet page metadata, activate the project environment first: "
            r"conda activate D:\GraduationProject\.gp"
        ) from exc

    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    LOGGER.info("Loaded %s rows=%d from %s", name, len(df), path)
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet output is supported: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    LOGGER.info("Saved %d rows to %s", len(df), path)


def save_preview(df: pd.DataFrame, output_path: Path, preview_rows: int) -> Path:
    preview_path = output_path.with_name(f"{output_path.stem}_preview.csv")
    df.head(max(preview_rows, 1)).to_csv(preview_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved preview rows=%d to %s", min(len(df), max(preview_rows, 1)), preview_path)
    return preview_path


def normalize_id_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_id_column(series: pd.Series) -> pd.Series:
    return series.map(normalize_id_value).astype("string")


def id_to_int(value: Any) -> int:
    normalized = normalize_id_value(value)
    if not normalized:
        raise ValueError("ID value is empty and cannot be converted to int64.")
    return int(normalized)


def convert_id_column_to_int64(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series.map(normalize_id_value), errors="coerce")
    if numeric.isna().any():
        bad_values = series[numeric.isna()].head(10).tolist()
        raise ValueError(f"{column_name} contains non-numeric ID values: {bad_values}")
    return numeric.astype("int64")


def parse_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f", ""}:
        return False
    return False


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_for_similarity(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    normalized = re.sub(r"https?://\S+|网页链接", "", normalized)
    return normalized.lower()


def parse_multi_value(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[,，]", text) if part.strip()]
    return list(dict.fromkeys(parts))


def parse_profile_items(value: Any) -> list[tuple[str, int, float]]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw_items = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    items: list[tuple[str, int, float]] = []
    for item in raw_items:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        name = normalize_text(item[0])
        if not name:
            continue
        count = pd.to_numeric(item[1], errors="coerce")
        ratio = pd.to_numeric(item[2], errors="coerce")
        items.append((name, int(count) if pd.notna(count) else 0, float(ratio) if pd.notna(ratio) else 0.0))
    return items


def profile_item_ratio_map(value: Any) -> dict[str, float]:
    return {name: ratio for name, _count, ratio in parse_profile_items(value)}


def profile_item_top_set(value: Any, top_n: int = 3) -> set[str]:
    return {name for name, _count, _ratio in parse_profile_items(value)[:top_n]}


def clip01(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0.0
    return float(min(max(float(numeric), 0.0), 1.0))


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def is_low_information_text(text: str) -> bool:
    stripped = normalize_text(text)
    if not stripped:
        return True
    return any(pattern.fullmatch(stripped) for pattern in LOW_INFORMATION_PATTERNS)


def is_template_text(text: str) -> bool:
    stripped = normalize_text(text)
    if not stripped:
        return True
    return any(pattern.search(stripped) for pattern in TEMPLATE_PATTERNS)


def extract_repost_comment(text: Any) -> str:
    raw_text = normalize_text(text)
    if not raw_text:
        return ""
    comment = raw_text.split("//", 1)[0].strip()
    if is_low_information_text(comment):
        return ""
    return comment


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    meaningful = re.findall(r"[\w\u4e00-\u9fff]", text)
    return 1.0 - len(meaningful) / max(len(text), 1)


def has_repeated_noise(text: str) -> bool:
    if re.search(r"(.)\1{5,}", text):
        return True
    return text.count("//") >= 2


def length_score(length: float, config: MemorySampleConfig) -> float:
    if config.ideal_text_min_length <= length <= config.ideal_text_max_length:
        return 1.0
    if config.min_agent_text_length <= length < config.ideal_text_min_length:
        return 0.75
    if config.ideal_text_max_length < length <= config.medium_text_max_length:
        return 0.75
    if config.medium_text_max_length < length <= config.max_agent_text_length:
        return 0.45
    return 0.0


def completeness_score(text: str) -> float:
    if count_chinese_chars(text) < 8:
        return 0.0
    marker_hit = any(marker in text for marker in VIEWPOINT_MARKERS)
    punctuation_hit = bool(re.search(r"[。！？!?；;，,、…]", text))
    if marker_hit and punctuation_hit:
        return 1.0
    if marker_hit or punctuation_hit or len(text) >= 20:
        return 0.75
    return 0.35


def tone_score(text: str) -> float:
    if not text:
        return 0.0
    score = 0.55
    if any(marker in text for marker in TONE_MARKERS):
        score += 0.25
    if re.search(r"[！？!?…]", text):
        score += 0.15
    if has_repeated_noise(text):
        score -= 0.20
    return clip01(score)


def low_noise_score(text: str) -> float:
    if is_template_text(text) or is_low_information_text(text):
        return 0.0
    ratio = symbol_ratio(text)
    if ratio <= 0.20:
        return 1.0
    if ratio <= 0.35:
        return 0.75
    if ratio <= 0.50:
        return 0.35
    return 0.0


def user_voice_score(is_repost: bool, has_repost_comment: bool, text_length: float, config: MemorySampleConfig) -> float:
    if not is_repost:
        return 1.0
    if has_repost_comment and text_length >= config.typical_repost_comment_min_length:
        return 0.70
    if has_repost_comment and text_length >= config.min_repost_comment_length:
        return 0.45
    return 0.0


def compute_style_score(row: pd.Series, config: MemorySampleConfig) -> float:
    text = normalize_text(row["content_for_agent"])
    length = float(row["agent_text_length"])
    score = (
        0.35 * length_score(length, config)
        + 0.25 * completeness_score(text)
        + 0.15 * tone_score(text)
        + 0.15
        * user_voice_score(
            bool(row["is_repost"]),
            bool(row["has_repost_comment"]),
            length,
            config,
        )
        + 0.10 * low_noise_score(text)
    )
    return round(clip01(score), 4)


def normalize_verified_type(verified_type_name: Any) -> str:
    value = normalize_text(verified_type_name)
    if value == "媒体":
        return "媒体"
    if value == "政府":
        return "政府"
    if value in {"企业", "校园", "网站", "应用", "团体/机构"}:
        return "机构"
    if value == "个人认证":
        return "个人认证"
    if value == "普通用户":
        return "普通用户"
    return "未知"


def build_content_for_agent(row: pd.Series, config: MemorySampleConfig) -> str:
    is_repost = bool(row["is_repost"])
    has_comment = bool(row["has_repost_comment"])
    raw_content = normalize_text(row.get("cleaned_content")) or normalize_text(row.get("content"))

    if is_repost:
        if not has_comment:
            return ""
        comment = extract_repost_comment(raw_content)
        if len(comment) < config.min_repost_comment_length:
            return ""
        return comment[: config.max_agent_text_length].strip()

    return raw_content[: config.max_agent_text_length].strip()


def build_source_context(row: pd.Series) -> str:
    if not bool(row.get("is_repost")):
        return ""
    parts: list[str] = []
    source_author_type = normalize_text(row.get("source_author_type"))
    if source_author_type:
        parts.append(f"源作者类型：{source_author_type}")
    source_categories = normalize_text(row.get("final_topic_categories"))
    if source_categories:
        parts.append(f"相关主题：{source_categories}")
    source_topics = normalize_text(row.get("source_topics"))
    if source_topics:
        parts.append(f"源微博话题：{source_topics}")
    return "；".join(parts)


def prepare_topic_fusion(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = normalize_id_column(prepared["user_id"])
    prepared["weibo_id"] = normalize_id_column(prepared["weibo_id"])
    prepared["reposted_weibo_id"] = normalize_id_column(prepared["reposted_weibo_id"])
    prepared["is_repost"] = prepared["is_repost"].map(parse_bool)
    prepared["has_repost_comment"] = prepared["has_repost_comment"].map(parse_bool)
    prepared["text_quality"] = safe_numeric(prepared["text_quality"])
    prepared["final_topic_confidence"] = safe_numeric(prepared["final_topic_confidence"]).clip(0, 1)
    return prepared


def prepare_user_weibo(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = normalize_id_column(prepared["user_id"])
    prepared["weibo_id"] = normalize_id_column(prepared["weibo_id"])
    prepared["reposted_weibo_id"] = normalize_id_column(prepared["reposted_weibo_id"])
    prepared["is_repost"] = prepared["is_repost"].map(parse_bool)
    for column in [
        "text_quality",
        "text_length",
        "cleaned_text_length",
        "like_count",
        "comment_count",
        "repost_count",
        "engagement",
    ]:
        prepared[column] = safe_numeric(prepared[column])
    return prepared


def prepare_weibo_emotion(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = normalize_id_column(prepared["user_id"])
    prepared["weibo_id"] = normalize_id_column(prepared["weibo_id"])
    for column in ["model_confidence", "polarity_score", "emotion_intensity_score"]:
        prepared[column] = safe_numeric(prepared[column])
    return prepared


def prepare_user_emotion(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = normalize_id_column(prepared["user_id"])
    prepared = prepared[prepared["profile_version"].astype(str).eq("all")].copy()
    for column in [
        "analyzable_weibo_count",
        "pos_ratio",
        "neu_ratio",
        "neg_ratio",
        "polarity_std",
        "strong_emotion_ratio",
    ]:
        prepared[column] = safe_numeric(prepared[column])
    return prepared.drop(columns=["profile_version"])


def prepare_user_topic(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = normalize_id_column(prepared["user_id"])
    for column in [
        "total_weibo_count",
        "final_public_issue_topic_ratio",
        "marketing_topic_ratio",
        "final_topic_coverage",
        "avg_final_topic_confidence",
    ]:
        prepared[column] = safe_numeric(prepared[column])
    prepared["top_category_ratio_map"] = prepared["final_category_distribution"].map(profile_item_ratio_map)
    prepared["top_category_set"] = prepared["final_category_distribution"].map(profile_item_top_set)
    prepared["top_label_ratio_map"] = prepared["top_implicit_topic_labels"].map(profile_item_ratio_map)
    prepared["top_label_set"] = prepared["top_implicit_topic_labels"].map(profile_item_top_set)
    return prepared


def prepare_user_propagation(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = normalize_id_column(prepared["user_id"])
    for column in [
        "weibo_hq_count",
        "active_days",
        "original_ratio",
        "repost_ratio",
        "repost_with_comment_ratio",
        "media_dependency_score",
        "kol_sensitivity_score",
        "high_engagement_weibo_ratio",
        "influence_score",
    ]:
        prepared[column] = safe_numeric(prepared[column])
    return prepared


def prepare_user_info(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = normalize_id_column(prepared["user_id"])
    return prepared


def build_source_author_lookup(
    cleaned_user_weibo: pd.DataFrame,
    source_creator: pd.DataFrame,
) -> pd.DataFrame:
    source_weibo = cleaned_user_weibo[["weibo_id", "user_id"]].copy()
    source_weibo["weibo_id"] = normalize_id_column(source_weibo["weibo_id"])
    source_weibo["source_user_id"] = normalize_id_column(source_weibo["user_id"])
    source_weibo = (
        source_weibo.drop(columns=["user_id"])
        .dropna(subset=["weibo_id"])
        .drop_duplicates(subset=["weibo_id"])
        .rename(columns={"weibo_id": "reposted_weibo_id"})
    )

    creator = source_creator.copy()
    creator["source_user_id"] = normalize_id_column(creator["user_id"])
    creator["source_author_type"] = creator["verified_type_name"].map(normalize_verified_type)
    creator = creator[["source_user_id", "source_author_type"]].drop_duplicates(subset=["source_user_id"])
    lookup = source_weibo.merge(creator, on="source_user_id", how="left")
    lookup["source_author_type"] = lookup["source_author_type"].fillna("未知")
    return lookup[["reposted_weibo_id", "source_author_type"]]


def build_user_profiles(
    user_info: pd.DataFrame,
    user_emotion: pd.DataFrame,
    user_topic: pd.DataFrame,
    user_propagation: pd.DataFrame,
    config: MemorySampleConfig,
) -> pd.DataFrame:
    profiles = (
        user_info.merge(user_emotion, on="user_id", how="left")
        .merge(user_topic, on="user_id", how="left")
        .merge(user_propagation, on="user_id", how="left")
    )
    score = pd.Series(0, index=profiles.index, dtype="int64")
    score += (
        profiles["profile_reliability"].eq("high")
        & profiles["final_topic_profile_reliability"].eq("高可靠")
    ).astype(int) * 2
    score += profiles["user_value_label"].eq("核心建模候选用户").astype(int) * 2
    score += profiles["propagation_activity_level"].eq("高").astype(int)
    score += profiles["influence_level"].eq("高").astype(int)
    score += safe_numeric(profiles["weibo_hq_count"]).ge(config.weibo_hq_high_threshold).astype(int)
    score += safe_numeric(profiles["final_public_issue_topic_ratio"]).ge(config.public_issue_high_ratio).astype(int)
    score += (
        safe_numeric(profiles["repost_ratio"]).ge(0.5)
        & safe_numeric(profiles["repost_with_comment_ratio"]).ge(0.5)
    ).astype(int)
    score += safe_numeric(profiles["strong_emotion_ratio"]).ge(config.strong_user_emotion_ratio).astype(int)
    profiles["core_score_before_marketing_penalty"] = score
    marketing_ratio = safe_numeric(profiles["marketing_topic_ratio"])
    marketing_penalty = pd.Series(0, index=profiles.index, dtype="int64")
    marketing_penalty += marketing_ratio.ge(config.marketing_ratio_mid).astype(int)
    marketing_penalty += marketing_ratio.ge(config.marketing_ratio_high).astype(int)
    marketing_penalty += marketing_ratio.ge(config.marketing_ratio_very_high).astype(int)
    profiles["marketing_score_penalty"] = marketing_penalty
    score = (score - marketing_penalty).clip(lower=0)
    profiles["core_score"] = score

    background_mask = safe_numeric(profiles["weibo_hq_count"]).lt(3) | (
        profiles["profile_reliability"].fillna("insufficient").eq("insufficient")
        & profiles["final_topic_profile_reliability"].fillna("低可靠").eq("低可靠")
        & safe_numeric(profiles["final_topic_coverage"]).lt(0.3)
    )
    profiles["memory_user_level"] = "normal"
    profiles.loc[background_mask, "memory_user_level"] = "background"
    profiles.loc[profiles["core_score"].ge(3), "memory_user_level"] = "core"

    return profiles


def build_base_table(
    topic_fusion: pd.DataFrame,
    user_weibo: pd.DataFrame,
    weibo_emotion: pd.DataFrame,
    user_profiles: pd.DataFrame,
    source_author_lookup: pd.DataFrame,
    limit_users: int | None,
) -> pd.DataFrame:
    if limit_users is not None:
        user_ids = sorted(user_profiles["user_id"].dropna().unique().tolist())[:limit_users]
        user_profiles = user_profiles[user_profiles["user_id"].isin(user_ids)].copy()
        topic_fusion = topic_fusion[topic_fusion["user_id"].isin(user_ids)].copy()

    weibo_meta_columns = [
        "weibo_id",
        "user_id",
        "cleaned_content",
        "text_length",
        "cleaned_text_length",
        "text_quality_label",
        "create_time",
        "year",
        "like_count",
        "comment_count",
        "repost_count",
        "engagement",
        "topics",
        "at_users",
    ]
    emotion_columns = [
        "weibo_id",
        "user_id",
        "sentiment_label_en",
        "sentiment_label",
        "model_confidence",
        "polarity_score",
        "emotion_intensity_score",
    ]

    base = topic_fusion.merge(user_weibo[weibo_meta_columns], on=["user_id", "weibo_id"], how="left")
    base = base.merge(weibo_emotion[emotion_columns], on=["user_id", "weibo_id"], how="left")
    base = base.merge(user_profiles, on="user_id", how="left")
    base = base.merge(source_author_lookup, on="reposted_weibo_id", how="left")
    base["source_author_type"] = base["source_author_type"].fillna("")
    return base


def add_agent_text(base: pd.DataFrame, config: MemorySampleConfig) -> pd.DataFrame:
    result = base.copy()
    result["cleaned_content"] = result["cleaned_content"].where(
        result["cleaned_content"].notna(),
        result["content"],
    )
    result["content_for_agent"] = result.apply(build_content_for_agent, axis=1, config=config)
    result["topic_tags_for_agent"] = result["topics"]
    result["mentions_for_agent"] = result["at_users"]
    result["agent_text_length"] = result["content_for_agent"].map(len)
    result["normalized_agent_text"] = result["content_for_agent"].map(normalize_for_similarity)
    result["source_context_for_agent"] = result.apply(build_source_context, axis=1)
    return result


def filter_candidates(df: pd.DataFrame, config: MemorySampleConfig) -> pd.DataFrame:
    working = df.copy()
    working["model_confidence"] = safe_numeric(working["model_confidence"])
    working["final_topic_confidence"] = safe_numeric(working["final_topic_confidence"]).clip(0, 1)
    working["text_quality"] = safe_numeric(working["text_quality"])
    working["cleaned_text_length"] = safe_numeric(working["cleaned_text_length"])

    quality_mask = working["text_quality"].ge(config.min_text_quality)
    length_mask = working["agent_text_length"].between(config.min_agent_text_length, config.max_agent_text_length)
    content_mask = working["content_for_agent"].map(lambda text: bool(text) and not is_low_information_text(text))
    template_mask = ~working["content_for_agent"].map(is_template_text)
    marketing_mask = ~working.apply(is_marketing_topic, axis=1)
    reliable_mask = ~(
        working["final_topic_confidence"].lt(config.topic_low_threshold)
        & working["model_confidence"].lt(config.emotion_low_confidence)
    )

    candidates = working[quality_mask & length_mask & content_mask & template_mask & marketing_mask & reliable_mask].copy()
    LOGGER.info(
        "Candidate filter kept %d/%d rows (quality=%d length=%d content=%d template=%d non_marketing=%d reliable=%d)",
        len(candidates),
        len(working),
        int(quality_mask.sum()),
        int(length_mask.sum()),
        int(content_mask.sum()),
        int(template_mask.sum()),
        int(marketing_mask.sum()),
        int(reliable_mask.sum()),
    )
    return candidates


def score_quality(df: pd.DataFrame, config: MemorySampleConfig) -> pd.Series:
    text_quality_component = (safe_numeric(df["text_quality"]) / 3.0).clip(0, 1)
    length_component = df["agent_text_length"].map(lambda value: length_score(float(value), config))
    non_empty_component = df["content_for_agent"].map(lambda text: 0.0 if is_low_information_text(text) else 1.0)
    return (0.45 * text_quality_component + 0.45 * length_component + 0.10 * non_empty_component).clip(0, 1)


def confidence_score(series: pd.Series, low: float, high: float) -> pd.Series:
    numeric = safe_numeric(series)
    if high <= low:
        return numeric.clip(0, 1)
    return ((numeric - low) / (high - low)).clip(0, 1)


def sentiment_matches_profile(row: pd.Series) -> float:
    label = normalize_text(row.get("sentiment_label_en"))
    dominant = normalize_text(row.get("dominant_emotion"))
    polarity_std = float(row.get("polarity_std") or 0.0)
    if not label:
        return 0.0
    if dominant == "Mixed" or polarity_std >= 0.56:
        return 0.70
    if label == dominant:
        return 1.0
    if dominant == "Neutral" and label == "Neutral":
        return 1.0
    return 0.25


def score_emotion(df: pd.DataFrame, config: MemorySampleConfig) -> pd.Series:
    confidence_component = confidence_score(
        df["model_confidence"],
        config.emotion_low_confidence,
        config.emotion_high_confidence,
    )
    intensity_component = safe_numeric(df["emotion_intensity_score"]).clip(0, 1)
    match_component = df.apply(sentiment_matches_profile, axis=1)
    return (0.45 * confidence_component + 0.35 * intensity_component + 0.20 * match_component).clip(0, 1)


def topic_match_score(row: pd.Series, config: MemorySampleConfig) -> float:
    categories = parse_multi_value(row.get("final_topic_categories"))
    labels = parse_multi_value(row.get("final_topic_labels"))
    category_map = row.get("top_category_ratio_map")
    label_map = row.get("top_label_ratio_map")
    category_set = row.get("top_category_set")
    label_set = row.get("top_label_set")
    if not isinstance(category_map, dict):
        category_map = {}
    if not isinstance(label_map, dict):
        label_map = {}
    if not isinstance(category_set, set):
        category_set = set()
    if not isinstance(label_set, set):
        label_set = set()

    score = 0.0
    for category in categories:
        ratio = float(category_map.get(category, 0.0))
        if category in category_set or ratio >= config.category_ratio_threshold:
            score = max(score, min(1.0, 0.65 + ratio))
    for label in labels:
        ratio = float(label_map.get(label, 0.0))
        if label in label_set or ratio >= config.label_ratio_threshold:
            score = max(score, min(1.0, 0.60 + ratio * 2))
    return score


def score_topic(df: pd.DataFrame, config: MemorySampleConfig) -> pd.Series:
    confidence_component = safe_numeric(df["final_topic_confidence"]).clip(0, 1)
    match_component = df.apply(topic_match_score, axis=1, config=config)
    source_component = df["topic_signal_source"].fillna("").map(
        lambda value: 1.0 if value == "显式主题" else (0.75 if value == "隐式主题" else 0.0)
    )
    return (0.55 * confidence_component + 0.35 * match_component + 0.10 * source_component).clip(0, 1)


def is_public_issue(row: pd.Series) -> bool:
    return any(category in PUBLIC_ISSUE_CATEGORIES for category in parse_multi_value(row.get("final_topic_categories")))


def is_marketing_topic(row: pd.Series) -> bool:
    return MARKETING_CATEGORY in parse_multi_value(row.get("final_topic_categories"))


def score_public_issue(df: pd.DataFrame, config: MemorySampleConfig) -> pd.Series:
    category_component = df.apply(lambda row: 1.0 if is_public_issue(row) else 0.0, axis=1)
    user_component = (safe_numeric(df["final_public_issue_topic_ratio"]) / config.public_issue_high_ratio).clip(0, 1)
    confidence_component = safe_numeric(df["final_topic_confidence"]).clip(0, 1)
    return (0.65 * category_component + 0.20 * user_component + 0.15 * confidence_component).clip(0, 1)


def score_propagation(df: pd.DataFrame, config: MemorySampleConfig) -> pd.Series:
    repost_comment_component = (
        df["is_repost"].astype(bool) & df["has_repost_comment"].astype(bool)
    ).astype(float)
    media_component = (safe_numeric(df["media_dependency_score"]) / config.media_dependency_high).clip(0, 1)
    kol_component = (safe_numeric(df["kol_sensitivity_score"]) / config.kol_sensitivity_high).clip(0, 1)
    repost_user_component = safe_numeric(df["repost_with_comment_ratio"]).clip(0, 1)
    return (
        0.45 * repost_comment_component
        + 0.20 * media_component
        + 0.20 * kol_component
        + 0.15 * repost_user_component
    ).clip(0, 1)


def score_engagement(df: pd.DataFrame, config: MemorySampleConfig) -> pd.Series:
    engagement = safe_numeric(df["engagement"])
    q99 = float(engagement.quantile(0.99)) if len(engagement) else config.high_engagement_threshold
    cap = max(q99, config.high_engagement_threshold)
    if cap <= 0:
        return pd.Series(0.0, index=df.index)
    return (np.log1p(engagement.clip(lower=0)) / math.log1p(cap)).clip(0, 1)


def add_scores(df: pd.DataFrame, config: MemorySampleConfig) -> pd.DataFrame:
    result = df.copy()
    result["quality_score"] = score_quality(result, config).round(4)
    result["emotion_score"] = score_emotion(result, config).round(4)
    result["topic_score"] = score_topic(result, config).round(4)
    result["public_issue_score"] = score_public_issue(result, config).round(4)
    result["propagation_score"] = score_propagation(result, config).round(4)
    result["engagement_score_norm"] = score_engagement(result, config).round(4)
    result["style_score"] = result.apply(compute_style_score, axis=1, config=config).round(4)
    result["diversity_penalty"] = 0.0
    result["engagement_score"] = safe_numeric(result["engagement"]).round(4)

    weighted = pd.Series(0.0, index=result.index)
    for column, weight in config.score_weights.items():
        weighted += weight * safe_numeric(result[column])
    result["base_memory_score"] = weighted.clip(0, 1).round(4)
    result["memory_score"] = result["base_memory_score"]

    user_q80 = result.groupby("user_id")["emotion_intensity_score"].transform(lambda s: safe_numeric(s).quantile(0.80))
    user_q90_engagement = result.groupby("user_id")["engagement"].transform(lambda s: safe_numeric(s).quantile(0.90))
    result["is_strong_emotion_weibo"] = safe_numeric(result["emotion_intensity_score"]).ge(
        config.strong_weibo_emotion_threshold
    ) | safe_numeric(result["emotion_intensity_score"]).ge(user_q80)
    result["is_high_engagement_weibo"] = safe_numeric(result["engagement"]).ge(config.high_engagement_threshold) | (
        result["influence_level"].eq("高")
        & safe_numeric(result["engagement"]).ge(np.maximum(config.high_engagement_user_floor, user_q90_engagement))
    )
    return result


def eligible_for_type(row: pd.Series, memory_type: str, config: MemorySampleConfig) -> bool:
    length = float(row.get("agent_text_length") or 0.0)
    if memory_type == "typical_style":
        strong_user = float(row.get("strong_emotion_ratio") or 0.0) >= config.strong_user_emotion_ratio
        if bool(row.get("is_strong_emotion_weibo")) and not strong_user:
            return False
        return (
            float(row.get("style_score") or 0.0) >= 0.55
            and float(row.get("quality_score") or 0.0) >= 0.55
            and (not bool(row.get("is_repost")) or length >= config.typical_repost_comment_min_length)
        )
    if memory_type == "topic_representative":
        return (
            float(row.get("topic_score") or 0.0) >= 0.50
            and float(row.get("final_topic_confidence") or 0.0) >= config.topic_reliable_threshold
        )
    if memory_type == "emotion_representative":
        return (
            float(row.get("emotion_score") or 0.0) >= 0.50
            and float(row.get("model_confidence") or 0.0) >= config.emotion_usable_confidence
            and float(row.get("style_score") or 0.0) >= 0.40
            and length >= 12
        )
    if memory_type == "public_issue":
        return float(row.get("public_issue_score") or 0.0) >= 0.60 and is_public_issue(row)
    if memory_type == "repost_behavior":
        return (
            bool(row.get("is_repost"))
            and bool(row.get("has_repost_comment"))
            and float(row.get("propagation_score") or 0.0) >= 0.50
        )
    if memory_type == "high_engagement":
        return bool(row.get("is_high_engagement_weibo"))
    return False


def infer_best_memory_type(row: pd.Series, config: MemorySampleConfig) -> str:
    if eligible_for_type(row, "public_issue", config):
        return "public_issue"
    if eligible_for_type(row, "repost_behavior", config):
        return "repost_behavior"
    if eligible_for_type(row, "topic_representative", config):
        return "topic_representative"
    if eligible_for_type(row, "emotion_representative", config):
        return "emotion_representative"
    if eligible_for_type(row, "high_engagement", config):
        return "high_engagement"
    return "typical_style"


def primary_category(row: pd.Series) -> str:
    categories = parse_multi_value(row.get("final_topic_categories"))
    return categories[0] if categories else ""


def selection_reason(row: pd.Series, memory_type: str, fallback_from: str | None = None) -> str:
    prefix = ""
    if fallback_from:
        prefix = f"{fallback_from} 配额候选不足，改用综合得分更高的样本。"
    if memory_type == "typical_style":
        return prefix + "该微博为用户自己的完整表达，文本长度和语气较适合体现日常语言风格。"
    if memory_type == "emotion_representative":
        return prefix + "该微博情绪识别可信度较高，情绪强度和表达完整度适合作为情绪代表样本。"
    if memory_type == "topic_representative":
        return prefix + "该微博命中用户高频或高可信主题，适合作为长期关注主题的代表样本。"
    if memory_type == "public_issue":
        return prefix + "该微博属于社会公共事件、政策民生或时事政治相关主题，适合作为公共议题参与样本。"
    if memory_type == "repost_behavior":
        return prefix + "该微博为带有效评语的转发，能够体现用户的转发扩散行为和信息源偏好。"
    if memory_type == "high_engagement":
        return prefix + "该微博互动量明显较高，可辅助体现用户潜在影响力或被关注程度。"
    return prefix + "该微博综合质量较高，适合作为用户记忆样本。"


def get_quota_for_user(group: pd.DataFrame, config: MemorySampleConfig) -> list[str]:
    level = normalize_text(group["memory_user_level"].iloc[0])
    if level == "core":
        return list(config.core_quota)
    if level == "normal":
        first_row = group.iloc[0]
        if (
            float(first_row.get("repost_ratio") or 0.0) >= 0.5
            and float(first_row.get("repost_with_comment_ratio") or 0.0) >= 0.5
        ):
            return list(config.normal_repost_quota)
        return list(config.normal_quota)
    return []


def diversity_penalty(row: pd.Series, selected_rows: list[pd.Series], selected_types: list[str], config: MemorySampleConfig) -> float:
    penalty = 0.0
    candidate_text = normalize_text(row.get("normalized_agent_text"))
    candidate_category = primary_category(row)
    candidate_sentiment = normalize_text(row.get("sentiment_label_en"))

    for selected in selected_rows:
        selected_text = normalize_text(selected.get("normalized_agent_text"))
        if candidate_text and selected_text:
            if candidate_text == selected_text:
                return float("inf")
            similarity = SequenceMatcher(None, candidate_text, selected_text).ratio()
            if similarity >= config.similarity_skip_threshold:
                return float("inf")
            if similarity >= config.similarity_penalty_threshold:
                penalty += 0.20

    category_count = sum(1 for selected in selected_rows if primary_category(selected) == candidate_category and candidate_category)
    if category_count >= 2:
        penalty += 0.15 * (category_count - 1)

    sentiment_count = sum(
        1
        for selected in selected_rows
        if normalize_text(selected.get("sentiment_label_en")) == candidate_sentiment and candidate_sentiment
    )
    if sentiment_count >= 2:
        penalty += 0.15 * (sentiment_count - 1)

    inferred_type = normalize_text(row.get("_target_memory_type"))
    if inferred_type and selected_types.count(inferred_type) >= 2:
        penalty += 0.10
    return round(penalty, 4)


def subtype_score(row: pd.Series, memory_type: str) -> float:
    if memory_type == "typical_style":
        return float(row.get("style_score") or 0.0)
    if memory_type == "topic_representative":
        return float(row.get("topic_score") or 0.0)
    if memory_type == "emotion_representative":
        return float(row.get("emotion_score") or 0.0)
    if memory_type == "public_issue":
        return float(row.get("public_issue_score") or 0.0)
    if memory_type == "repost_behavior":
        return float(row.get("propagation_score") or 0.0)
    if memory_type == "high_engagement":
        return float(row.get("engagement_score_norm") or 0.0)
    return 0.0


def choose_candidate(
    available: pd.DataFrame,
    selected_rows: list[pd.Series],
    selected_types: list[str],
    target_type: str | None,
    config: MemorySampleConfig,
) -> tuple[pd.Series | None, float, str | None]:
    best_row: pd.Series | None = None
    best_penalty = 0.0
    best_score = -1.0
    fallback_from: str | None = None

    if target_type is not None:
        pool = available[available.apply(eligible_for_type, axis=1, memory_type=target_type, config=config)].copy()
        if pool.empty:
            pool = available.copy()
            fallback_from = target_type
    else:
        pool = available.copy()

    for _, row in pool.iterrows():
        row = row.copy()
        row["_target_memory_type"] = target_type or infer_best_memory_type(row, config)
        penalty = diversity_penalty(row, selected_rows, selected_types, config)
        if math.isinf(penalty):
            continue
        row_type = target_type if fallback_from is None and target_type is not None else infer_best_memory_type(row, config)
        rank_score = float(row["base_memory_score"]) - penalty + 0.25 * subtype_score(row, row_type)
        if rank_score > best_score:
            best_score = rank_score
            best_row = row
            best_penalty = penalty

    return best_row, best_penalty, fallback_from


def select_user_samples(group: pd.DataFrame, config: MemorySampleConfig) -> list[dict[str, Any]]:
    group = group.sort_values(["base_memory_score", "weibo_id"], ascending=[False, True]).copy()
    available = group.copy()
    selected_rows: list[pd.Series] = []
    selected_types: list[str] = []
    rows: list[dict[str, Any]] = []
    level = normalize_text(group["memory_user_level"].iloc[0])

    quota = get_quota_for_user(group, config)
    if level == "background":
        candidate, penalty, _fallback = choose_candidate(available, selected_rows, selected_types, None, config)
        if candidate is None or float(candidate["base_memory_score"]) - penalty < config.min_background_score:
            return []
        quota = [infer_best_memory_type(candidate, config)]

    for target_type in quota:
        candidate, penalty, fallback_from = choose_candidate(
            available,
            selected_rows,
            selected_types,
            target_type,
            config,
        )
        if candidate is None:
            continue
        memory_type = target_type if fallback_from is None else infer_best_memory_type(candidate, config)
        candidate = candidate.copy()
        candidate["memory_type"] = memory_type
        candidate["diversity_penalty"] = penalty
        candidate["memory_score"] = round(max(float(candidate["base_memory_score"]) - penalty, 0.0), 4)
        if float(candidate["memory_score"]) < config.min_selected_score:
            continue
        candidate["selection_reason"] = selection_reason(candidate, memory_type, fallback_from=fallback_from)

        rows.append(candidate.to_dict())
        selected_rows.append(candidate)
        selected_types.append(memory_type)
        available = available[available["weibo_id"] != candidate["weibo_id"]].copy()

    return rows


def select_memory_samples(candidates: pd.DataFrame, config: MemorySampleConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _user_id, group in candidates.groupby("user_id", sort=True):
        try:
            rows.extend(select_user_samples(group, config))
        except Exception:
            LOGGER.exception("Failed to select samples for user_id=%s", _user_id)
            continue
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    selected = pd.DataFrame(rows)
    selected["user_id"] = convert_id_column_to_int64(selected["user_id"], "user_id")
    selected["weibo_id"] = convert_id_column_to_int64(selected["weibo_id"], "weibo_id")
    selected["sentiment_label"] = selected["sentiment_label"].fillna(selected["sentiment_label_en"])
    for column in [
        "polarity_score",
        "emotion_intensity_score",
        "model_confidence",
        "final_topic_confidence",
        "engagement_score",
        "quality_score",
        "emotion_score",
        "topic_score",
        "public_issue_score",
        "propagation_score",
        "engagement_score_norm",
        "style_score",
        "diversity_penalty",
        "memory_score",
    ]:
        selected[column] = safe_numeric(selected[column]).round(4)
    selected["has_repost_comment"] = selected["has_repost_comment"].astype(bool)
    selected["is_repost"] = selected["is_repost"].astype(bool)
    return selected[OUTPUT_COLUMNS].sort_values(["user_id", "memory_score"], ascending=[True, False]).reset_index(drop=True)


def build_user_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for user_id, group in selected.groupby("user_id", sort=True):
        type_counts = Counter(group["memory_type"].tolist())
        snippets: list[str] = []
        for row in group.sort_values("memory_score", ascending=False).head(3).itertuples(index=False):
            snippets.append(f"{row.memory_type}：{row.content_for_agent}")
        rows.append(
            {
                "user_id": user_id,
                "memory_user_level": group["memory_user_level"].iloc[0],
                "selected_memory_count": int(len(group)),
                "memory_type_counts": repr(dict(type_counts)),
                "selected_weibo_ids": [id_to_int(weibo_id) for weibo_id in group["weibo_id"].tolist()],
                "memory_summary_for_agent": "；".join(snippets),
            }
        )
    summary = pd.DataFrame(rows, columns=USER_SUMMARY_COLUMNS)
    if not summary.empty:
        summary["user_id"] = convert_id_column_to_int64(summary["user_id"], "user_id")
    return summary


def validate_output(selected: pd.DataFrame, config: MemorySampleConfig) -> None:
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in selected.columns]
    if missing_columns:
        raise AssertionError(f"Output missing columns: {missing_columns}")
    if selected.empty:
        LOGGER.warning("Selected memory sample output is empty.")
        return
    if selected["content_for_agent"].astype(str).str.strip().eq("").any():
        raise AssertionError("content_for_agent contains empty values")
    if selected.duplicated(subset=["user_id", "weibo_id"]).any():
        raise AssertionError("Duplicate user_id/weibo_id rows found")
    valid_types = {
        "typical_style",
        "emotion_representative",
        "topic_representative",
        "public_issue",
        "repost_behavior",
        "high_engagement",
    }
    invalid_types = set(selected["memory_type"].dropna()) - valid_types
    if invalid_types:
        raise AssertionError(f"Invalid memory_type values: {sorted(invalid_types)}")
    marketing_mask = selected["final_topic_categories"].fillna("").str.contains(MARKETING_CATEGORY, regex=False)
    if marketing_mask.any():
        bad_count = int(marketing_mask.sum())
        raise AssertionError(f"Selected output contains {bad_count} marketing topic samples")
    quota_limits = {"core": len(config.core_quota), "normal": len(config.normal_quota), "background": 1}
    counts = selected.groupby(["user_id", "memory_user_level"]).size().reset_index(name="count")
    too_many = counts[counts.apply(lambda row: row["count"] > quota_limits.get(row["memory_user_level"], 1), axis=1)]
    if not too_many.empty:
        raise AssertionError(f"Users exceed quota: {too_many.head(10).to_dict(orient='records')}")
    score_columns = [
        "quality_score",
        "emotion_score",
        "topic_score",
        "public_issue_score",
        "propagation_score",
        "engagement_score_norm",
        "style_score",
        "memory_score",
    ]
    for column in score_columns:
        if not selected[column].between(0.0, 1.0, inclusive="both").all():
            raise AssertionError(f"{column} has values outside [0, 1]")


def log_summary(user_profiles: pd.DataFrame, candidates: pd.DataFrame, selected: pd.DataFrame, output_path: Path) -> None:
    LOGGER.info("用户分层数量:\n%s", user_profiles["memory_user_level"].value_counts(dropna=False).to_string())
    if "marketing_topic_ratio" in user_profiles.columns:
        LOGGER.info(
            "marketing_topic_ratio describe:\n%s",
            safe_numeric(user_profiles["marketing_topic_ratio"]).describe(
                percentiles=[0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]
            ).to_string(),
        )
    if "marketing_score_penalty" in user_profiles.columns:
        LOGGER.info(
            "marketing_score_penalty 分布:\n%s",
            user_profiles["marketing_score_penalty"].value_counts(dropna=False).sort_index().to_string(),
        )
    LOGGER.info("候选微博数量: %d", len(candidates))
    LOGGER.info("成功选出记忆样本的用户数: %d", selected["user_id"].nunique() if not selected.empty else 0)
    avg_count = selected.groupby("user_id").size().mean() if not selected.empty else 0.0
    LOGGER.info("平均每用户样本数: %.4f", float(avg_count))
    if selected.empty:
        LOGGER.warning("No memory samples selected.")
        return
    LOGGER.info("memory_type 样本数量:\n%s", selected["memory_type"].value_counts(dropna=False).to_string())
    LOGGER.info("原创 / 转发样本数量:\n%s", selected["is_repost"].value_counts(dropna=False).to_string())
    LOGGER.info("主题覆盖 top20:\n%s", selected["final_topic_categories"].fillna("").value_counts().head(20).to_string())
    LOGGER.info("情绪标签分布:\n%s", selected["sentiment_label"].fillna("").value_counts(dropna=False).to_string())
    LOGGER.info("输出文件路径: %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, args.log_dir, args.log_file)
    config = MemorySampleConfig()

    topic_fusion = prepare_topic_fusion(load_parquet(args.topic_fusion_path, TOPIC_FUSION_COLUMNS, "topic fusion"))
    user_weibo = prepare_user_weibo(load_parquet(args.user_weibo_path, USER_WEIBO_COLUMNS, "user weibo"))
    weibo_emotion = prepare_weibo_emotion(
        load_parquet(args.weibo_emotion_path, WEIBO_EMOTION_COLUMNS, "weibo emotion")
    )
    user_emotion = prepare_user_emotion(load_parquet(args.user_emotion_path, USER_EMOTION_COLUMNS, "user emotion"))
    user_topic = prepare_user_topic(load_parquet(args.user_topic_path, USER_TOPIC_COLUMNS, "user topic"))
    user_propagation = prepare_user_propagation(
        load_parquet(args.user_propagation_path, USER_PROPAGATION_COLUMNS, "user propagation")
    )
    user_info = prepare_user_info(load_parquet(args.user_info_path, USER_INFO_COLUMNS, "user info"))

    cleaned_source_weibo = load_parquet(
        args.cleaned_user_weibo_path,
        ["weibo_id", "user_id"],
        "cleaned user weibo source lookup",
    )
    source_creator = load_parquet(args.source_creator_path, SOURCE_CREATOR_COLUMNS, "source creator")

    source_author_lookup = build_source_author_lookup(cleaned_source_weibo, source_creator)
    user_profiles = build_user_profiles(user_info, user_emotion, user_topic, user_propagation, config)
    base = build_base_table(
        topic_fusion,
        user_weibo,
        weibo_emotion,
        user_profiles,
        source_author_lookup,
        limit_users=args.limit_users,
    )
    base = add_agent_text(base, config)
    candidates = filter_candidates(base, config)
    candidates = add_scores(candidates, config)
    selected = select_memory_samples(candidates, config)
    validate_output(selected, config)

    save_dataframe(selected, args.output_path)
    # save_preview(selected, args.output_path, args.preview_rows)

    if not args.no_user_summary:
        user_summary = build_user_summary(selected)
        save_dataframe(user_summary, args.user_summary_output_path)
        # save_preview(user_summary, args.user_summary_output_path, args.preview_rows)

    log_summary(user_profiles, candidates, selected, args.output_path)


if __name__ == "__main__":
    main()
