from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROJECT_ROOT,
    configure_logging,
    normalize_id,
    parse_bool,
    read_table,
    safe_float,
    safe_get,
    safe_int,
    safe_str,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TOPIC_WEIBO_PATH = PROJECT_ROOT / "data" / "high_quality" / "topic_weibo.parquet"
DEFAULT_COMMENT_ANALYSIS_PATH = PROJECT_ROOT / "data" / "profile" / "comments" / "comment_analysis_result.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "scope" / "data" / "inputs" / "events.jsonl"

EVENT_TYPE = "public_issue"
UNKNOWN_STANCE_FOCUS = "评论区主要围绕该事件本身展开讨论，但具体争议焦点不够明确。"
NO_EMOTION_ANALYSIS_SUMMARY = "评论区暂无足够的情绪分析结果。"
NO_STANCE_ANALYSIS_SUMMARY = "评论区暂无足够的立场分析结果。"

NEGATIVE_EMOTIONS = {"anger", "sadness", "fear", "disgust", "disappointment", "confusion"}
POSITIVE_EMOTIONS = {"joy", "sympathy", "admiration"}
NEUTRAL_EMOTIONS = {"none", "surprise"}
MIXED_EMOTIONS = {"mixed"}
UNCLEAR_EMOTIONS = {"unclear"}
EMOTION_GROUPS = ["negative", "positive", "neutral", "mixed", "unclear"]
VALID_EMOTION_LABELS = NEGATIVE_EMOTIONS | POSITIVE_EMOTIONS | NEUTRAL_EMOTIONS | MIXED_EMOTIONS | UNCLEAR_EMOTIONS

STANCE_LABELS = ["favor", "against", "neutral", "mixed", "unclear"]
VALID_STANCE_LABELS = set(STANCE_LABELS)
VALID_TARGET_TYPES = {"person", "institution", "policy", "group", "media", "behavior", "event", "other"}

METADATA_FIELDS = {
    "comment_crawled_count": 0,
    "comment_hq_count": 0,
    "comment_hq_ratio": 0.0,
    "topic_value": 0,
    "topic_value_label": "",
    "trending_type": "",
    "trending_click": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event JSONL inputs for Weibo Agent simulations.")
    parser.add_argument("--topic-weibo-path", type=Path, default=DEFAULT_TOPIC_WEIBO_PATH)
    parser.add_argument("--comment-analysis-path", type=Path, default=DEFAULT_COMMENT_ANALYSIS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normalize_label(value: Any, valid_labels: set[str], default: str = "unclear") -> str:
    label = safe_str(value, default).strip().lower()
    return label if label in valid_labels else default


def infer_emotion_group(label: str) -> str:
    normalized = normalize_label(label, VALID_EMOTION_LABELS)
    if normalized in NEGATIVE_EMOTIONS:
        return "negative"
    if normalized in POSITIVE_EMOTIONS:
        return "positive"
    if normalized in NEUTRAL_EMOTIONS:
        return "neutral"
    if normalized in MIXED_EMOTIONS:
        return "mixed"
    return "unclear"


def rounded_ratio(value: float) -> float:
    return round(float(value), 4)


def weighted_distribution(labels: list[str], weights: list[float], ordered_labels: list[str]) -> dict[str, float]:
    totals = {label: 0.0 for label in ordered_labels}
    total_weight = sum(weights)
    if total_weight <= 0:
        return {label: 0.0 for label in ordered_labels}

    for label, weight in zip(labels, weights):
        normalized = label if label in totals else "unclear"
        totals[normalized] += weight
    return {label: rounded_ratio(totals[label] / total_weight) for label in ordered_labels}


def dominant_from_weights(labels: list[str], weights: list[float], default: str = "unclear") -> str:
    totals: defaultdict[str, float] = defaultdict(float)
    for label, weight in zip(labels, weights):
        totals[label] += weight
    if not totals:
        return default
    return max(totals.items(), key=lambda item: (item[1], item[0]))[0]


def build_weights(group: pd.DataFrame, intensity_column: str) -> list[float]:
    weights: list[float] = []
    for _, row in group.iterrows():
        confidence = safe_float(safe_get(row, "confidence"), 0.0)
        intensity = max(safe_float(safe_get(row, intensity_column), 0.0), 0.0)
        weight = confidence * (1 + 0.3 * intensity)
        if parse_bool(safe_get(row, "needs_more_context"), False):
            weight *= 0.5
        weights.append(max(weight, 0.0))

    if sum(weights) <= 0 and len(weights) > 0:
        return [1.0] * len(weights)
    return weights


def aggregate_emotion_features(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return empty_emotion_features()

    weights = build_weights(group, "emotion_intensity")
    labels = [normalize_label(safe_get(row, "emotion_label"), VALID_EMOTION_LABELS) for _, row in group.iterrows()]
    groups = [infer_emotion_group(label) for label in labels]
    intensities = [max(safe_float(safe_get(row, "emotion_intensity"), 0.0), 0.0) for _, row in group.iterrows()]
    total_weight = sum(weights)

    group_distribution = weighted_distribution(groups, weights, EMOTION_GROUPS)
    avg_intensity = (
        sum(weight * intensity for weight, intensity in zip(weights, intensities)) / total_weight if total_weight > 0 else 0.0
    )
    strong_ratio = (
        sum(weight for weight, intensity in zip(weights, intensities) if intensity >= 2) / total_weight
        if total_weight > 0
        else 0.0
    )

    return {
        "negative_emotion_ratio": group_distribution["negative"],
        "positive_emotion_ratio": group_distribution["positive"],
        "neutral_emotion_ratio": group_distribution["neutral"],
        "mixed_emotion_ratio": group_distribution["mixed"],
        "unclear_emotion_ratio": group_distribution["unclear"],
        "dominant_emotion_label": dominant_from_weights(labels, weights),
        "avg_emotion_intensity": rounded_ratio(avg_intensity),
        "strong_emotion_ratio": rounded_ratio(strong_ratio),
        "emotion_distribution": group_distribution,
    }


def empty_emotion_features() -> dict[str, Any]:
    return {
        "negative_emotion_ratio": 0.0,
        "positive_emotion_ratio": 0.0,
        "neutral_emotion_ratio": 0.0,
        "mixed_emotion_ratio": 0.0,
        "unclear_emotion_ratio": 0.0,
        "dominant_emotion_label": "unclear",
        "avg_emotion_intensity": 0.0,
        "strong_emotion_ratio": 0.0,
        "emotion_distribution": {group: 0.0 for group in EMOTION_GROUPS},
    }


def infer_event_emotion_tendency(features: dict[str, Any]) -> str:
    negative_ratio = safe_float(features.get("negative_emotion_ratio"), 0.0)
    positive_ratio = safe_float(features.get("positive_emotion_ratio"), 0.0)
    neutral_ratio = safe_float(features.get("neutral_emotion_ratio"), 0.0)
    mixed_ratio = safe_float(features.get("mixed_emotion_ratio"), 0.0)

    if negative_ratio >= 0.45 and negative_ratio - positive_ratio >= 0.15:
        return "negative"
    if positive_ratio >= 0.45 and positive_ratio - negative_ratio >= 0.15:
        return "positive"
    if negative_ratio >= 0.25 and positive_ratio >= 0.25:
        return "mixed"
    if mixed_ratio >= 0.25:
        return "mixed"
    if neutral_ratio >= 0.50:
        return "neutral"
    return "unclear"


def choose_preferred_emotion_label(
    labels: list[str],
    weights: list[float],
    preferred_labels: set[str],
    fallback: str,
) -> str:
    preferred = [(label, weight) for label, weight in zip(labels, weights) if label in preferred_labels]
    if not preferred:
        return fallback
    return dominant_from_weights([label for label, _ in preferred], [weight for _, weight in preferred], fallback)


def refine_dominant_emotion_label(comment_group: pd.DataFrame, tendency: str, fallback: str) -> str:
    if comment_group.empty:
        return fallback

    weights = build_weights(comment_group, "emotion_intensity")
    labels = [
        normalize_label(safe_get(row, "emotion_label"), VALID_EMOTION_LABELS)
        for _, row in comment_group.iterrows()
    ]
    if tendency == "negative":
        return choose_preferred_emotion_label(labels, weights, NEGATIVE_EMOTIONS, fallback)
    if tendency == "positive":
        return choose_preferred_emotion_label(labels, weights, {"joy", "sympathy"}, fallback)
    return fallback


def describe_intensity(features: dict[str, Any]) -> str:
    avg_intensity = safe_float(features.get("avg_emotion_intensity"), 0.0)
    strong_ratio = safe_float(features.get("strong_emotion_ratio"), 0.0)
    if avg_intensity >= 2 or strong_ratio >= 0.45:
        return "较高"
    if avg_intensity >= 1 or strong_ratio >= 0.2:
        return "中等"
    return "较低"


def build_event_emotion_summary(features: dict[str, Any]) -> str:
    tendency = infer_event_emotion_tendency(features)
    dominant_label = safe_str(features.get("dominant_emotion_label"), "unclear")
    intensity_text = describe_intensity(features)

    if tendency == "negative":
        return f"评论区整体以负向情绪为主，主导情绪接近 {dominant_label}，整体情绪强度{intensity_text}。"
    if tendency == "positive":
        return f"评论区整体以正向情绪为主，主导情绪接近 {dominant_label}，整体情绪强度{intensity_text}。"
    if tendency == "mixed":
        return "评论区情绪较为分化，正负向反应或混合情绪同时存在。"
    if tendency == "neutral":
        return "评论区整体情绪较为中性，强烈情绪表达相对较少。"
    return "评论区整体情绪倾向不够明确。"


def normalize_target_type(value: Any) -> str:
    target_type = safe_str(value, "unclear").strip().lower()
    if target_type in VALID_TARGET_TYPES:
        return target_type
    return "unclear"


def normalize_text(value: Any) -> str:
    return safe_str(value, "").strip()


def aggregate_by_labels(rows: list[tuple[pd.Series, float]], field: str, valid_labels: set[str] | None = None) -> str:
    labels: list[str] = []
    weights: list[float] = []
    for row, weight in rows:
        if valid_labels is None:
            label = safe_str(safe_get(row, field), "unclear").strip().lower()
        else:
            label = normalize_label(safe_get(row, field), valid_labels)
        labels.append(label or "unclear")
        weights.append(weight)
    return dominant_from_weights(labels, weights)


def choose_dominant_target(rows: list[tuple[pd.Series, float]]) -> tuple[str, str, list[tuple[pd.Series, float]]]:
    if not rows:
        return "unclear", "", []

    type_weights: defaultdict[str, float] = defaultdict(float)
    for row, weight in rows:
        type_weights[normalize_target_type(safe_get(row, "stance_target_type"))] += weight
    target_type = max(type_weights.items(), key=lambda item: (item[1], item[0]))[0]
    same_type_rows = [
        (row, weight)
        for row, weight in rows
        if normalize_target_type(safe_get(row, "stance_target_type")) == target_type
    ]

    text_weights: defaultdict[str, float] = defaultdict(float)
    for row, weight in same_type_rows:
        text_weights[normalize_text(safe_get(row, "stance_target_text"))] += weight
    target_text, text_weight = max(text_weights.items(), key=lambda item: (item[1], item[0]))
    type_weight = sum(weight for _, weight in same_type_rows)

    if not target_text or (type_weight > 0 and text_weight / type_weight < 0.4):
        return target_type, "", same_type_rows

    matched_rows = [
        (row, weight)
        for row, weight in same_type_rows
        if normalize_text(safe_get(row, "stance_target_text")) == target_text
    ]
    return target_type, target_text, matched_rows


def aggregate_stance_features(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return empty_stance_features()

    weights = build_weights(group, "stance_intensity")
    rows = [(row, weight) for (_, row), weight in zip(group.iterrows(), weights)]
    labels = [normalize_label(safe_get(row, "stance_label"), VALID_STANCE_LABELS) for row, _ in rows]
    stance_distribution = weighted_distribution(labels, weights, STANCE_LABELS)

    candidate_rows = [
        (row, weight)
        for row, weight in rows
        if normalize_target_type(safe_get(row, "stance_target_type")) != "unclear"
        and normalize_label(safe_get(row, "stance_label"), VALID_STANCE_LABELS) != "unclear"
    ]
    target_type, target_text, target_rows = choose_dominant_target(candidate_rows)

    if not target_rows:
        return {
            **empty_stance_features(),
            "stance_distribution": stance_distribution,
            "dominant_stance_label": dominant_from_weights(labels, weights),
        }

    target_labels = [normalize_label(safe_get(row, "stance_label"), VALID_STANCE_LABELS) for row, _ in target_rows]
    target_weights = [weight for _, weight in target_rows]
    dominant_stance_label = dominant_from_weights(target_labels, target_weights)
    dominant_responsibility = aggregate_by_labels(target_rows, "responsibility")
    dominant_norm_violation = aggregate_by_labels(target_rows, "norm_violation")

    return {
        "dominant_stance_label": dominant_stance_label,
        "dominant_stance_target_type": target_type,
        "dominant_stance_target_text": target_text,
        "dominant_responsibility": dominant_responsibility,
        "dominant_norm_violation": dominant_norm_violation,
        "event_stance_focus": build_stance_focus_description(
            target_type,
            target_text,
            dominant_responsibility,
            dominant_norm_violation,
        ),
        "stance_distribution": stance_distribution,
    }


def empty_stance_features() -> dict[str, Any]:
    return {
        "dominant_stance_label": "unclear",
        "dominant_stance_target_type": "unclear",
        "dominant_stance_target_text": "",
        "dominant_responsibility": "unclear",
        "dominant_norm_violation": "unclear",
        "event_stance_focus": UNKNOWN_STANCE_FOCUS,
        "stance_distribution": {label: 0.0 for label in STANCE_LABELS},
    }


def typed_target_text(target_type: str, target_text: str) -> str:
    if target_text:
        return f"“{target_text}”"
    defaults = {
        "person": "相关人物",
        "institution": "相关机构",
        "policy": "相关政策",
        "group": "相关群体",
        "media": "相关媒体",
        "behavior": "相关行为",
        "event": "该事件本身",
    }
    return defaults.get(target_type, "相关对象")


def clean_focus_text(text: str, limit: int = 80) -> str:
    cleaned = re.sub(r"\s+", "", text)
    cleaned = cleaned.strip(" ，,。；;：:“”\"'（）()[]【】")
    return cleaned[:limit]


def extract_focus_from_context(event_context: str) -> str:
    context = normalize_text(event_context)
    if not context:
        return ""

    patterns = [
        r"(?:争议点|争议焦点)(?:主要)?在于[:：]?\s*([^。；;\n]+)",
        r"引发对([^。；;\n]{2,80}?)(?:的)?争议",
        r"引发了?关于([^。；;\n]{2,80}?)(?:的)?争议",
    ]
    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            focus = clean_focus_text(match.group(1))
            if focus:
                return focus
    return ""


def infer_stance_target_text(row: pd.Series, event_context: str) -> str:
    extracted = extract_focus_from_context(event_context)
    if extracted:
        return extracted
    return clean_focus_text(safe_str(safe_get(row, "topic"), ""))


def build_stance_focus_description(
    target_type: str,
    target_text: str,
    responsibility: str,
    norm_violation: str,
) -> str:
    target_type = normalize_target_type(target_type)
    target = typed_target_text(target_type, normalize_text(target_text))

    if target_type == "unclear":
        return UNKNOWN_STANCE_FOCUS
    if target_type == "person":
        description = f"评论区主要围绕人物{target}的言行是否合理展开评价。"
    elif target_type == "institution":
        description = f"评论区主要围绕机构{target}的处理方式和责任归属展开评价。"
    elif target_type == "policy":
        description = f"评论区主要围绕相关政策{target}是否合理、是否应被支持展开讨论。"
    elif target_type == "group":
        description = f"评论区主要围绕群体{target}的行为、处境或责任展开评价。"
    elif target_type == "media":
        description = f"评论区主要围绕媒体{target}的报道方式或舆论引导展开评价。"
    elif target_type == "behavior":
        suffix = "这一行为" if normalize_text(target_text) else ""
        description = f"评论区主要围绕{target}{suffix}是否合理、是否违反公共规范展开评价。"
    elif target_type == "event":
        suffix = "这一事件本身" if normalize_text(target_text) else ""
        description = f"评论区主要围绕{target}{suffix}的性质、影响和处理方式展开评价。"
    else:
        description = "评论区主要围绕该事件本身的性质、责任归属和处理方式展开讨论。"

    norm = safe_str(norm_violation, "unclear").strip().lower()
    if norm in {"high", "medium"}:
        description += "其中不少评论带有规范违背或道德谴责判断。"

    responsibility_text = safe_str(responsibility, "unclear").strip().lower()
    if responsibility_text == "institution":
        description += "评论中较多将责任归于相关机构或管理方。"
    elif responsibility_text == "other_individual":
        description += "评论中较多将责任归于具体个人。"
    elif responsibility_text == "media":
        description += "评论中较多关注媒体报道或舆论传播责任。"
    elif responsibility_text == "society":
        description += "评论中较多将问题归因于社会层面的结构性因素。"

    return description


def infer_event_type(_: pd.Series) -> str:
    return EVENT_TYPE


def build_event_context(event_row: pd.Series) -> str:
    context = safe_str(safe_get(event_row, "analysis_context"), "")
    if context:
        return context

    summary = safe_str(safe_get(event_row, "summary_text"), "")
    content = safe_str(safe_get(event_row, "content"), "")
    topic = safe_str(safe_get(event_row, "topic"), "")
    parts = []
    if topic:
        parts.append(f"话题：{topic}")
    if summary:
        parts.append(f"摘要：{summary}")
    if content:
        parts.append(f"原微博：{content}")
    return "\n".join(parts) if parts else "暂无可用事件上下文。"


def build_metadata(row: pd.Series) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field, default in METADATA_FIELDS.items():
        value = safe_get(row, field, default)
        if isinstance(default, int):
            metadata[field] = safe_int(value, default)
        elif isinstance(default, float):
            metadata[field] = safe_float(value, default)
        else:
            metadata[field] = safe_str(value, default)
    return metadata


def build_empty_analysis_record(row: pd.Series) -> dict[str, Any]:
    weibo_id = normalize_id(safe_get(row, "weibo_id"))
    return {
        "event_id": f"event_{weibo_id}",
        "weibo_id": weibo_id,
        "topic": safe_str(safe_get(row, "topic"), ""),
        "event_context": build_event_context(row),
        "event_type": infer_event_type(row),
        "event_emotion_tendency": "unclear",
        "event_emotion_summary": NO_EMOTION_ANALYSIS_SUMMARY,
        "event_stance_focus": NO_STANCE_ANALYSIS_SUMMARY,
        "dominant_emotion_label": "unclear",
        "dominant_stance_label": "unclear",
        "dominant_stance_target_type": "unclear",
        "dominant_stance_target_text": "",
        "dominant_responsibility": "unclear",
        "dominant_norm_violation": "unclear",
        "comment_count_used": 0,
        "emotion_distribution": {group: 0.0 for group in EMOTION_GROUPS},
        "stance_distribution": {label: 0.0 for label in STANCE_LABELS},
        "metadata": build_metadata(row),
    }


def build_event_record(row: pd.Series, comment_group: pd.DataFrame) -> dict[str, Any]:
    if comment_group.empty:
        return build_empty_analysis_record(row)

    weibo_id = normalize_id(safe_get(row, "weibo_id"))
    emotion_features = aggregate_emotion_features(comment_group)
    stance_features = aggregate_stance_features(comment_group)
    event_emotion_tendency = infer_event_emotion_tendency(emotion_features)
    dominant_emotion_label = refine_dominant_emotion_label(
        comment_group,
        event_emotion_tendency,
        safe_str(emotion_features.get("dominant_emotion_label"), "unclear"),
    )
    emotion_features["dominant_emotion_label"] = dominant_emotion_label
    event_context = build_event_context(row)

    stance_target_type = safe_str(stance_features.get("dominant_stance_target_type"), "unclear")
    stance_target_text = safe_str(stance_features.get("dominant_stance_target_text"), "")
    if not stance_target_text:
        stance_target_text = infer_stance_target_text(row, event_context)
    event_stance_focus = build_stance_focus_description(
        stance_target_type,
        stance_target_text,
        safe_str(stance_features.get("dominant_responsibility"), "unclear"),
        safe_str(stance_features.get("dominant_norm_violation"), "unclear"),
    )

    return {
        "event_id": f"event_{weibo_id}",
        "weibo_id": weibo_id,
        "topic": safe_str(safe_get(row, "topic"), ""),
        "event_context": event_context,
        "event_type": infer_event_type(row),
        "event_emotion_tendency": event_emotion_tendency,
        "event_emotion_summary": build_event_emotion_summary(emotion_features),
        "event_stance_focus": event_stance_focus,
        "dominant_emotion_label": dominant_emotion_label,
        "dominant_stance_label": safe_str(stance_features.get("dominant_stance_label"), "unclear"),
        "dominant_stance_target_type": stance_target_type,
        "dominant_stance_target_text": stance_target_text,
        "dominant_responsibility": safe_str(stance_features.get("dominant_responsibility"), "unclear"),
        "dominant_norm_violation": safe_str(stance_features.get("dominant_norm_violation"), "unclear"),
        "comment_count_used": int(len(comment_group)),
        "emotion_distribution": emotion_features["emotion_distribution"],
        "stance_distribution": stance_features["stance_distribution"],
        "metadata": build_metadata(row),
    }


def prepare_comment_groups(df_comment_analysis: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "weibo_id" not in df_comment_analysis.columns:
        LOGGER.warning("评论分析结果缺少 weibo_id 字段，将无法匹配任何事件")
        return {}

    prepared = df_comment_analysis.copy()
    prepared["_normalized_weibo_id"] = prepared["weibo_id"].map(normalize_id)
    prepared = prepared[prepared["_normalized_weibo_id"] != ""].copy()
    return {weibo_id: group.drop(columns=["_normalized_weibo_id"]) for weibo_id, group in prepared.groupby("_normalized_weibo_id")}


def build_event_inputs(
    df_topic_weibo: pd.DataFrame,
    df_comment_analysis: pd.DataFrame,
) -> list[dict[str, Any]]:
    if "weibo_id" not in df_topic_weibo.columns:
        raise ValueError("话题微博数据缺少 weibo_id 字段，无法构建事件输入")

    comment_groups = prepare_comment_groups(df_comment_analysis)
    records: list[dict[str, Any]] = []
    missing_comment_analysis = 0

    for _, row in df_topic_weibo.iterrows():
        weibo_id = normalize_id(safe_get(row, "weibo_id"))
        comment_group = comment_groups.get(weibo_id, pd.DataFrame())
        if comment_group.empty:
            missing_comment_analysis += 1
        records.append(build_event_record(row, comment_group))

    LOGGER.info("没有评论分析结果的事件数: %d", missing_comment_analysis)
    return records


def truncate_text(value: Any, limit: int = 180) -> str:
    text = safe_str(value, "")
    return text if len(text) <= limit else text[:limit] + "..."


def log_record_distributions(records: list[dict[str, Any]]) -> None:
    LOGGER.info("成功生成的事件数: %d", len(records))
    for field in ["event_emotion_tendency", "dominant_emotion_label", "dominant_stance_label", "event_type"]:
        LOGGER.info("%s 分布: %s", field, dict(Counter(record.get(field, "unclear") for record in records)))

    empty_target_text_count = sum(1 for record in records if not safe_str(record.get("dominant_stance_target_text"), ""))
    LOGGER.info("dominant_stance_target_text 为空的数量: %d", empty_target_text_count)
    for index, record in enumerate(records[:5], start=1):
        LOGGER.info("event_stance_focus 样例 %d: %s", index, truncate_text(record.get("event_stance_focus"), 220))

    for index, record in enumerate(records[:3], start=1):
        sample = dict(record)
        sample["event_context"] = truncate_text(sample.get("event_context"), 180)
        LOGGER.info("事件输入样例 %d: %s", index, json.dumps(sample, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, "event_builder.log")

    LOGGER.info("话题微博输入文件: %s", args.topic_weibo_path)
    LOGGER.info("评论分析输入文件: %s", args.comment_analysis_path)
    df_topic_weibo = read_table(args.topic_weibo_path)
    df_comment_analysis = read_table(args.comment_analysis_path)
    LOGGER.info("df_topic_weibo 行数: %d", len(df_topic_weibo))
    LOGGER.info("df_comment_analysis 行数: %d", len(df_comment_analysis))

    records = build_event_inputs(df_topic_weibo, df_comment_analysis)
    log_record_distributions(records)
    write_jsonl(records, args.output_path)
    LOGGER.info("输出文件路径: %s", args.output_path)


if __name__ == "__main__":
    main()
