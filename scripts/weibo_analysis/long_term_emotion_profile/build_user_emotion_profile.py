from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "emotion_profile"
    / "user_weibo_emotion_analysis.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "emotion_profile"
    / "user_emotion_profile.parquet"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "user_weibo_emotion.log"

REQUIRED_COLUMNS = {
    "user_id",
    "year",
    "sentiment_label_en",
    "polarity_score",
    "emotion_intensity_score",
    "text_weight",
}

PROFILE_COLUMNS = [
    "user_id",
    "profile_version",
    "analyzable_weibo_count",
    "weighted_weibo_count",
    "pos_ratio",
    "neu_ratio",
    "neg_ratio",
    "avg_polarity_score",
    "median_polarity_score",
    "polarity_std",
    "avg_intensity_score",
    "strong_emotion_ratio",
    "strong_positive_ratio",
    "strong_negative_ratio",
    "dominant_emotion",
    "emotion_profile_type",
    "profile_reliability",
    "emotion_profile_summary",
]


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


def load_weibo_emotion_results(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Weibo emotion result file not found: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported, got: {input_path}")

    try:
        df = pd.read_parquet(input_path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow or fastparquet in the active Python environment."
        ) from exc

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    LOGGER.info("Loaded %d weibo emotion rows from %s", len(df), input_path)
    return df


def prepare_weibo_emotion_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["user_id"] = pd.to_numeric(prepared["user_id"], errors="coerce")
    prepared["year"] = pd.to_numeric(prepared["year"], errors="coerce")
    prepared["polarity_score"] = pd.to_numeric(prepared["polarity_score"], errors="coerce").fillna(0.0)
    prepared["emotion_intensity_score"] = pd.to_numeric(
        prepared["emotion_intensity_score"],
        errors="coerce",
    ).fillna(0.0)
    prepared["text_weight"] = pd.to_numeric(prepared["text_weight"], errors="coerce").fillna(0.0)
    prepared["text_weight"] = prepared["text_weight"].clip(lower=0.0)
    prepared["sentiment_label_en"] = prepared["sentiment_label_en"].fillna("Neutral").astype(str)
    prepared = prepared[prepared["user_id"].notna()].copy()
    prepared["user_id"] = prepared["user_id"].astype("int64")
    LOGGER.info("Prepared %d rows for user-level aggregation", len(prepared))
    return prepared


def _weighted_sum(mask: pd.Series, weights: pd.Series) -> float:
    return float(weights[mask].sum())


def _safe_weight_total(weights: pd.Series) -> float:
    total = float(weights.sum())
    return total if total > 0 else float(len(weights))


def _weighted_ratio(mask: pd.Series, weights: pd.Series) -> float:
    total = _safe_weight_total(weights)
    if total <= 0:
        return 0.0
    if float(weights.sum()) > 0:
        return float(weights[mask].sum() / total)
    return float(mask.mean()) if len(mask) else 0.0


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    total = _safe_weight_total(weights)
    if total <= 0:
        return 0.0
    if float(weights.sum()) > 0:
        return float((values * weights).sum() / total)
    return float(values.mean()) if len(values) else 0.0


def determine_dominant_emotion(pos_ratio: float, neu_ratio: float, neg_ratio: float) -> str:
    ratios = {"Positive": pos_ratio, "Neutral": neu_ratio, "Negative": neg_ratio}
    sorted_ratios = sorted(ratios.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_ratios) >= 2 and abs(sorted_ratios[0][1] - sorted_ratios[1][1]) < 0.05:
        return "Mixed"
    return sorted_ratios[0][0]


def classify_emotion_profile(row: dict[str, Any]) -> str:
    if row["analyzable_weibo_count"] < 5:
        return "样本不足型"
    if row["neu_ratio"] >= 0.6 and row["avg_intensity_score"] < 0.4:
        return "稳定中性型"
    if row["neg_ratio"] >= 0.4 and row["strong_negative_ratio"] >= 0.2:
        return "强消极表达型"
    if row["pos_ratio"] >= 0.4 and row["strong_positive_ratio"] >= 0.2:
        return "强积极表达型"
    if row["polarity_std"] >= 0.5:
        return "高波动型"
    if row["avg_polarity_score"] > 0.15:
        return "轻度积极型"
    if row["avg_polarity_score"] < -0.15:
        return "轻度消极型"
    return "混合表达型"


def determine_profile_reliability(analyzable_weibo_count: int) -> str:
    if analyzable_weibo_count >= 20:
        return "high"
    if analyzable_weibo_count >= 10:
        return "medium"
    if analyzable_weibo_count >= 5:
        return "low"
    return "insufficient"


def build_emotion_profile_summary(row: dict[str, Any]) -> str:
    count = row["analyzable_weibo_count"]
    neu_ratio = row["neu_ratio"]
    pos_ratio = row["pos_ratio"]
    neg_ratio = row["neg_ratio"]
    avg_polarity_score = row["avg_polarity_score"]
    avg_intensity_score = row["avg_intensity_score"]
    polarity_std = row["polarity_std"]
    strong_emotion_ratio = row["strong_emotion_ratio"]
    strong_positive_ratio = row["strong_positive_ratio"]
    strong_negative_ratio = row["strong_negative_ratio"]
    profile_reliability = row["profile_reliability"]

    if count < 5:
        tendency_phrase = "该用户历史原创微博样本较少，暂难稳定判断整体情绪倾向"
    elif neu_ratio >= 0.6 and abs(avg_polarity_score) < 0.15:
        tendency_phrase = "该用户历史原创微博整体偏中性"
    elif avg_polarity_score >= 0.25 or pos_ratio >= 0.45:
        tendency_phrase = "该用户历史原创微博整体偏积极"
    elif avg_polarity_score <= -0.25 or neg_ratio >= 0.45:
        tendency_phrase = "该用户历史原创微博整体偏消极"
    elif avg_polarity_score >= 0.10:
        tendency_phrase = "该用户历史原创微博略偏积极"
    elif avg_polarity_score <= -0.10:
        tendency_phrase = "该用户历史原创微博略偏消极"
    else:
        tendency_phrase = "该用户历史原创微博积极、消极与中性表达较为混合"

    if avg_intensity_score >= 0.65 or strong_emotion_ratio >= 0.35:
        intensity_phrase = "情绪强度较高"
    elif avg_intensity_score >= 0.45 or strong_emotion_ratio >= 0.20:
        intensity_phrase = "情绪强度中等"
    else:
        intensity_phrase = "情绪强度较低"

    if polarity_std >= 0.65:
        volatility_phrase = "表达波动较大"
    elif polarity_std >= 0.40:
        volatility_phrase = "存在一定情绪波动"
    else:
        volatility_phrase = "表达相对稳定"

    extra_phrase = ""
    if strong_negative_ratio >= 0.25 and neg_ratio >= 0.35:
        extra_phrase = "较容易出现强消极表达"
    elif strong_positive_ratio >= 0.25 and pos_ratio >= 0.35:
        extra_phrase = "较容易出现强积极表达"
    elif pos_ratio >= 0.35 and neg_ratio >= 0.35:
        extra_phrase = "积极与消极表达均有一定比例"

    if profile_reliability == "insufficient":
        reliability_phrase = "样本不足，画像仅供参考"
    elif profile_reliability == "low":
        reliability_phrase = "可分析样本较少，画像可靠性有限"
    elif profile_reliability == "medium":
        reliability_phrase = "画像可靠性中等"
    else:
        reliability_phrase = ""

    phrases = [tendency_phrase, intensity_phrase, volatility_phrase]
    if extra_phrase:
        phrases.append(extra_phrase)
    if reliability_phrase:
        phrases.append(reliability_phrase)
    return "；".join(phrases) + "。"


def aggregate_user_group(group: pd.DataFrame, profile_version: str, strong_threshold: float) -> dict[str, Any]:
    weights = group["text_weight"]
    polarity = group["polarity_score"]
    intensity = group["emotion_intensity_score"]
    positive_mask = group["sentiment_label_en"].eq("Positive")
    neutral_mask = group["sentiment_label_en"].eq("Neutral")
    negative_mask = group["sentiment_label_en"].eq("Negative")
    strong_mask = intensity >= strong_threshold

    row: dict[str, Any] = {
        "user_id": int(group["user_id"].iloc[0]),
        "profile_version": profile_version,
        "analyzable_weibo_count": int(len(group)),
        "weighted_weibo_count": float(weights.sum()),
        "pos_ratio": _weighted_ratio(positive_mask, weights),
        "neu_ratio": _weighted_ratio(neutral_mask, weights),
        "neg_ratio": _weighted_ratio(negative_mask, weights),
        "avg_polarity_score": _weighted_average(polarity, weights),
        "median_polarity_score": float(polarity.median()) if len(polarity) else 0.0,
        "polarity_std": float(polarity.std(ddof=0)) if len(polarity) > 1 else 0.0,
        "avg_intensity_score": _weighted_average(intensity, weights),
        "strong_emotion_ratio": _weighted_ratio(strong_mask, weights),
        "strong_positive_ratio": _weighted_ratio(strong_mask & positive_mask, weights),
        "strong_negative_ratio": _weighted_ratio(strong_mask & negative_mask, weights),
    }
    row["dominant_emotion"] = determine_dominant_emotion(row["pos_ratio"], row["neu_ratio"], row["neg_ratio"])
    row["emotion_profile_type"] = classify_emotion_profile(row)
    row["profile_reliability"] = determine_profile_reliability(row["analyzable_weibo_count"])
    row["emotion_profile_summary"] = build_emotion_profile_summary(row)
    return row


def filter_by_profile_version(df: pd.DataFrame, profile_version: str) -> pd.DataFrame:
    if profile_version == "2025":
        return df[df["year"] == 2025].copy()
    if profile_version == "all":
        return df.copy()
    raise ValueError(f"Unsupported profile_version: {profile_version}")


def build_user_emotion_profiles(
    df_weibo_emotion: pd.DataFrame,
    strong_threshold: float = 0.7,
    profile_versions: tuple[str, ...] = ("2025", "all"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_version in profile_versions:
        version_df = filter_by_profile_version(df_weibo_emotion, profile_version)
        LOGGER.info("Aggregating profile_version=%s from %d rows", profile_version, len(version_df))
        for _, group in version_df.groupby("user_id", sort=True):
            rows.append(aggregate_user_group(group, profile_version, strong_threshold))

    profile_df = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    profile_df["user_id"] = profile_df["user_id"].astype("int64")
    LOGGER.info("Built %d user emotion profile rows", len(profile_df))
    return profile_df


def save_user_emotion_profile(df: pd.DataFrame, output_path: Path) -> None:
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet output is supported, got: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Saved %d user emotion profile rows to %s", len(df), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate weibo-level emotion results into user-level profiles.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH, help="Input weibo emotion parquet path.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output user profile parquet path.")
    parser.add_argument("--strong_threshold", type=float, default=0.7, help="Threshold for strong emotion weibos.")
    parser.add_argument(
        "--profile_versions",
        nargs="+",
        choices=["2025", "all"],
        default=["2025", "all"],
        help="Profile versions to generate.",
    )
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for log files.")
    parser.add_argument("--log_file", type=Path, default=None, help="Optional explicit log file path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, log_dir=args.log_dir, log_file=args.log_file)
    LOGGER.info("Using input_path=%s", args.input_path)
    LOGGER.info("Using output_path=%s", args.output_path)
    LOGGER.info("Using strong_threshold=%s profile_versions=%s", args.strong_threshold, args.profile_versions)

    df_weibo_emotion = load_weibo_emotion_results(args.input_path)
    prepared = prepare_weibo_emotion_dataframe(df_weibo_emotion)
    df_user_emotion_profile = build_user_emotion_profiles(
        prepared,
        strong_threshold=args.strong_threshold,
        profile_versions=tuple(args.profile_versions),
    )
    save_user_emotion_profile(df_user_emotion_profile, args.output_path)


if __name__ == "__main__":
    main()
