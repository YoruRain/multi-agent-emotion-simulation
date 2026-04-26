from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

MODEL_TEXT_COLUMN = "cleaned_content"

INTENSIFIER_WORDS = (
    "真的", "太", "特别", "极其", "完全", "超级",
    "巨", "爆", "狠狠", "太顶了", "太强了", "太牛了",
    "太绝了", "绝了", "离谱", "爆炸", "救命", "真的会谢",
    "狠狠爱了", "绝绝子",
)

POSITIVE_EMOTION_WORDS = (
    "开心", "感动", "高兴", "快乐", "喜欢", "期待",
    "好看", "好听", "好帅", "好美", "幸福", "舒服",
    "爽", "惊喜", "爱了", "绝美", "封神", "治愈",
    "满足", "开心死", "太会了", "不错", "真好", "可爱",
    "萌", "好萌",
)

NEGATIVE_EMOTION_WORDS = (
    "难受", "崩溃", "恶心", "震惊", "无语", "生气", 
    "失望", "破防", "笑死","绷不住", "烦", "累", 
    "痛苦", "难过", "遗憾", "绝望", "气死", "讨厌",
    "恨", "无力", "命苦", "受不了", "放过我", "求放过", 
    "愁死", "心疼","裂开", "想哭", "哭死", "疯了", 
    "有病", "离谱",
)

EMOTION_WORDS = POSITIVE_EMOTION_WORDS + NEGATIVE_EMOTION_WORDS

EMOTION_MARKERS = (
    "😂", "🤣", "😭", "😢", "😡", "😠", "😱", "🥲", "😅",
    "[泪]", "[怒]", "[哈哈]", "[笑哭]", "[允悲]", "[裂开]", "[苦涩]",
    "[悲伤]", "[抓狂]", "[可怜]", "[抱抱]", "[赞]", "[good]", "[心]", "[爱你]",
)

LOW_PROFILE_VALUE_WORDS = (
    "打卡", "day", "记录", "存图", "随拍", "侵删", 
    "上线", "更新", "预告","发布", "汇总", "整理", 
    "名单", "时间表", "攻略", "教程", "经验", "案例",
    "案例反馈", "想了解", "礼貌提问", "网页链接", "详情", 
    "福利", "活动","推荐", "安利",
)

REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
PUNCTUATION_RE = re.compile(r"[!！?？]")
CONTINUOUS_PUNCTUATION_RE = re.compile(r"([!！?？])\1+")


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Prepare model input text from cleaned_content."""
    if MODEL_TEXT_COLUMN not in df.columns:
        raise ValueError(f"Input dataframe must contain {MODEL_TEXT_COLUMN} for model inference.")

    return df[MODEL_TEXT_COLUMN].fillna("").astype(str).str.strip().tolist()


def clean_text_for_model(text: Any) -> str:
    """Compatibility wrapper: upstream cleaning is already stored in cleaned_content."""
    return _safe_text(text)


def _safe_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    return str(text).strip()


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(lowered.count(keyword.lower()) for keyword in keywords)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def has_obvious_emotion_signal(text: Any) -> bool:
    """Detect short but explicit emotional expressions for weight exceptions."""
    value = _safe_text(text)
    if not value:
        return False

    return any(
        (
            bool(CONTINUOUS_PUNCTUATION_RE.search(value)),
            bool(REPEATED_CHAR_RE.search(value)),
            _count_keyword_hits(value, INTENSIFIER_WORDS) > 0,
            _count_keyword_hits(value, EMOTION_WORDS) > 0,
            _count_keyword_hits(value, EMOTION_MARKERS) > 0,
        )
    )


def compute_rule_intensity(text: Any) -> float:
    """Compute the intensity of emotions in the text based on predefined rules.

    Args:
        text (Any): The input text to analyze.

    Returns:
        float: The computed intensity score between 0 and 1.
    """
    value = _safe_text(text)
    if not value:
        return 0.0

    punctuation_count = len(PUNCTUATION_RE.findall(value))
    continuous_punctuation_count = len(CONTINUOUS_PUNCTUATION_RE.findall(value))
    repeated_char_count = len(REPEATED_CHAR_RE.findall(value))
    intensifier_count = _count_keyword_hits(value, INTENSIFIER_WORDS)
    positive_count = _count_keyword_hits(value, POSITIVE_EMOTION_WORDS)
    negative_count = _count_keyword_hits(value, NEGATIVE_EMOTION_WORDS)
    marker_count = _count_keyword_hits(value, EMOTION_MARKERS)

    score = 0.0
    score += min(punctuation_count, 8) * 0.035
    score += min(continuous_punctuation_count, 4) * 0.11
    score += min(repeated_char_count, 4) * 0.09
    score += min(intensifier_count, 5) * 0.09
    score += min(positive_count + negative_count, 6) * 0.10
    score += min(marker_count, 5) * 0.10

    if punctuation_count == 1 and not has_obvious_emotion_signal(value):
        score *= 0.5

    return _clip(score, 0.0, 1.0)


def _to_bool(value: Any) -> bool:
    """Convert a value to a boolean.

    Args:
        value (Any): The value to convert.

    Returns:
        bool: The converted boolean value.
    """
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "转发", "repost"}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_confidence_weight(model_confidence: Any = None) -> float:
    confidence = _safe_float(model_confidence)
    if confidence is None:
        return 1.0
    if confidence >= 0.75:
        return 1.0
    if confidence >= 0.6:
        return 0.8
    return 0.5


def compute_profile_value_factor(text: Any) -> float:
    """Downweight texts that carry little stable long-term profile signal."""
    value = _safe_text(text)
    if not value:
        return 0.4

    has_signal = has_obvious_emotion_signal(value)
    if len(value) < 6 and not has_signal:
        return 0.4

    low_value_hits = _count_keyword_hits(value, LOW_PROFILE_VALUE_WORDS)
    if low_value_hits == 0:
        return 1.0
    if has_signal:
        return 0.8
    return 0.6


def compute_text_weight(text: Any, is_repost: Any = False, model_confidence: Any = None) -> float:
    value = _safe_text(text)
    text_length = len(value)

    if text_length < 6:
        length_weight = 0.6 if has_obvious_emotion_signal(value) else 0.3
    elif text_length < 15:
        length_weight = 0.6
    else:
        length_weight = 1.0

    source_weight = 0.7 if _to_bool(is_repost) else 1.0
    confidence_weight = _compute_confidence_weight(model_confidence)
    profile_value_factor = compute_profile_value_factor(value)
    weight = length_weight * source_weight * confidence_weight * profile_value_factor
    return _clip(weight, 0.1, 1.0)
