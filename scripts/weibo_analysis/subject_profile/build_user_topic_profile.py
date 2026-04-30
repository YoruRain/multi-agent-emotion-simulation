from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
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
    / "subject_profile"
    / "user_weibo_topic_fusion.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "user_topic_profile_final.parquet"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "user_topic_profile_final.log"

REQUIRED_COLUMNS = {
    "user_id",
    "is_repost",
    "user_topics",
    "source_topics",
    "explicit_topic_categories",
    "signal_confidence",
    "implicit_topic_label",
    "implicit_topic_category",
    "implicit_topic_valid",
    "implicit_topic_confidence_score",
    "final_topic_categories",
    "final_topic_labels",
    "final_topic_confidence",
}
OUTPUT_COLUMNS = [
    "user_id",
    "total_weibo_count",
    "repost_weibo_count",
    "original_weibo_count",
    "top_user_topics",
    "top_source_topics",
    "top_all_topics",
    "top_categories",
    "top_implicit_topic_labels",
    "top_final_topic_categories",
    "final_category_distribution",
    "public_issue_topic_ratio",
    "final_public_issue_topic_ratio",
    "entertainment_topic_ratio",
    "final_entertainment_topic_ratio",
    "daily_life_topic_ratio",
    "final_daily_life_topic_ratio",
    "repost_topic_dependency",
    "explicit_topic_coverage",
    "implicit_valid_topic_coverage",
    "final_topic_coverage",
    "topic_source_balance_label",
    "explicit_topic_profile_reliability",
    "final_topic_profile_reliability",
    "avg_signal_confidence",
    "avg_final_topic_confidence",
]

PUBLIC_ISSUE_CATEGORIES = {"社会公共事件", "政策民生", "时事政治"}
ENTERTAINMENT_CATEGORIES = {"娱乐文化", "游戏动漫", "体育竞技"}
DAILY_LIFE_CATEGORIES = {"日常生活", "数码科技"}


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


def load_input_dataframe(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported, got: {input_path}")

    try:
        df = pd.read_parquet(input_path, columns=sorted(REQUIRED_COLUMNS))
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow or fastparquet in the active Python environment."
        ) from exc
    except OSError as exc:
        created_by = get_parquet_created_by(input_path)
        raise OSError(
            "Failed to read the parquet input data pages. "
            f"Input path: {input_path}. "
            f"Parquet created_by: {created_by or 'unknown'}. "
            "If the file was written by a newer pyarrow/parquet-cpp version, upgrade the active pyarrow "
            "environment or regenerate the parquet file with the current environment."
        ) from exc

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    prepared = df.copy()
    prepared["user_id"] = prepared["user_id"].astype(str)
    prepared["is_repost"] = prepared["is_repost"].map(parse_bool).fillna(False).astype(bool)
    prepared["signal_confidence"] = pd.to_numeric(prepared["signal_confidence"], errors="coerce").fillna(0.0)
    prepared["final_topic_confidence"] = pd.to_numeric(
        prepared["final_topic_confidence"],
        errors="coerce",
    ).fillna(0.0)
    prepared["implicit_topic_valid"] = prepared["implicit_topic_valid"].map(parse_bool).fillna(False).astype(bool)
    return prepared


def get_parquet_created_by(input_path: Path) -> str | None:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(input_path).metadata.created_by
    except Exception:
        return None


def parse_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f", ""}:
        return False
    return False


def parse_multi_value(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return list(dict.fromkeys(parts))


def parse_categories(value: Any) -> list[str]:
    categories = []
    for item in parse_multi_value(value):
        categories.append(item)
    return list(dict.fromkeys(categories))


def dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def parse_single_value_as_list(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    item = str(value).strip()
    return [item] if item else []


def count_top_items(list_series: pd.Series, denominator: int, top_n: int | None = 10) -> str | None:
    counter: Counter[str] = Counter()
    for values in list_series:
        if not values:
            continue
        counter.update(dedupe_preserve_order(values))

    if not counter:
        return None

    top_items: list[tuple[str, int, float]] = []
    most_common_items = counter.most_common(top_n) if top_n is not None else counter.most_common()
    for item, count in most_common_items:
        ratio = round(count / denominator, 4) if denominator > 0 else 0.0
        top_items.append((item, int(count), ratio))
    return repr(top_items)


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def determine_topic_source_balance_label(repost_topic_dependency: float) -> str:
    if repost_topic_dependency >= 0.7:
        return "转发源主题依赖型"
    if repost_topic_dependency <= 0.3:
        return "自写主题主导型"
    return "混合主题来源型"


def determine_topic_profile_reliability(total_weibo_count: int, explicit_topic_coverage: float) -> str:
    # The user-provided thresholds leave a gap for mid-sized / mid-coverage users,
    # so the remaining cases default to low reliability.
    if total_weibo_count >= 20 and explicit_topic_coverage >= 0.5:
        return "高可靠"
    if total_weibo_count >= 10 and explicit_topic_coverage >= 0.3:
        return "中可靠"
    return "低可靠"


def determine_final_topic_profile_reliability(
    total_weibo_count: int,
    final_topic_coverage: float,
    avg_final_topic_confidence: float,
) -> str:
    if total_weibo_count >= 20 and final_topic_coverage >= 0.5 and avg_final_topic_confidence >= 0.5:
        return "高可靠"
    if total_weibo_count >= 10 and final_topic_coverage >= 0.3:
        return "中可靠"
    return "低可靠"


def build_user_topic_profile(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["is_repost"] = working["is_repost"].map(parse_bool).fillna(False).astype(bool)
    working["signal_confidence"] = pd.to_numeric(working["signal_confidence"], errors="coerce").fillna(0.0)
    working["implicit_topic_valid"] = working["implicit_topic_valid"].map(parse_bool).fillna(False).astype(bool)
    working["final_topic_confidence"] = pd.to_numeric(
        working["final_topic_confidence"],
        errors="coerce",
    ).fillna(0.0)
    working["user_topic_list"] = working["user_topics"].map(parse_multi_value)
    working["source_topic_list"] = working["source_topics"].map(parse_multi_value)
    working["all_topic_list"] = working.apply(
        lambda row: dedupe_preserve_order([*row["user_topic_list"], *row["source_topic_list"]]),
        axis=1,
    )
    working["category_list"] = working["explicit_topic_categories"].map(parse_categories)
    working["implicit_topic_label_list"] = working["implicit_topic_label"].map(parse_single_value_as_list)
    working.loc[~working["implicit_topic_valid"], "implicit_topic_label_list"] = working.loc[
        ~working["implicit_topic_valid"],
        "implicit_topic_label_list",
    ].map(lambda _: [])
    working["final_category_list"] = working["final_topic_categories"].map(parse_categories)

    rows: list[dict[str, Any]] = []
    for user_id, group in working.groupby("user_id", sort=True):
        total_weibo_count = int(len(group))
        repost_weibo_count = int(group["is_repost"].sum())
        original_weibo_count = total_weibo_count - repost_weibo_count
        has_explicit_category_count = int(group["category_list"].map(bool).sum())
        has_user_topic_count = int(group["user_topic_list"].map(bool).sum())
        has_source_topic_count = int(group["source_topic_list"].map(bool).sum())

        public_issue_count = int(
            group["category_list"].map(lambda items: any(item in PUBLIC_ISSUE_CATEGORIES for item in items)).sum()
        )
        final_public_issue_count = int(
            group["final_category_list"]
            .map(lambda items: any(item in PUBLIC_ISSUE_CATEGORIES for item in items))
            .sum()
        )
        entertainment_count = int(
            group["category_list"].map(lambda items: any(item in ENTERTAINMENT_CATEGORIES for item in items)).sum()
        )
        final_entertainment_count = int(
            group["final_category_list"]
            .map(lambda items: any(item in ENTERTAINMENT_CATEGORIES for item in items))
            .sum()
        )
        daily_life_count = int(
            group["category_list"].map(lambda items: any(item in DAILY_LIFE_CATEGORIES for item in items)).sum()
        )
        final_daily_life_count = int(
            group["final_category_list"].map(lambda items: any(item in DAILY_LIFE_CATEGORIES for item in items)).sum()
        )

        repost_topic_dependency = safe_ratio(
            has_source_topic_count,
            has_user_topic_count + has_source_topic_count,
        )
        explicit_topic_coverage = safe_ratio(has_explicit_category_count, total_weibo_count)
        implicit_valid_topic_coverage = safe_ratio(int(group["implicit_topic_valid"].sum()), total_weibo_count)
        final_topic_coverage = safe_ratio(int(group["final_category_list"].map(bool).sum()), total_weibo_count)
        avg_signal_confidence = round(float(group["signal_confidence"].mean()), 4) if total_weibo_count else 0.0
        avg_final_topic_confidence = (
            round(float(group["final_topic_confidence"].mean()), 4) if total_weibo_count else 0.0
        )
        original_group = group[~group["is_repost"]]
        repost_group = group[group["is_repost"]]

        rows.append(
            {
                "user_id": user_id,
                "total_weibo_count": total_weibo_count,
                "repost_weibo_count": repost_weibo_count,
                "original_weibo_count": original_weibo_count,
                "top_user_topics": count_top_items(
                    original_group["user_topic_list"],
                    denominator=original_weibo_count,
                    top_n=10,
                ),
                "top_source_topics": count_top_items(
                    repost_group["source_topic_list"],
                    denominator=repost_weibo_count,
                    top_n=10,
                ),
                "top_all_topics": count_top_items(
                    group["all_topic_list"],
                    denominator=total_weibo_count,
                    top_n=10,
                ),
                "top_categories": count_top_items(
                    group["category_list"],
                    denominator=total_weibo_count,
                    top_n=10,
                ),
                "top_implicit_topic_labels": count_top_items(
                    group["implicit_topic_label_list"],
                    denominator=total_weibo_count,
                    top_n=10,
                ),
                "top_final_topic_categories": count_top_items(
                    group["final_category_list"],
                    denominator=total_weibo_count,
                    top_n=10,
                ),
                "final_category_distribution": count_top_items(
                    group["final_category_list"],
                    denominator=total_weibo_count,
                    top_n=None,
                ),
                "public_issue_topic_ratio": safe_ratio(public_issue_count, total_weibo_count),
                "final_public_issue_topic_ratio": safe_ratio(final_public_issue_count, total_weibo_count),
                "entertainment_topic_ratio": safe_ratio(entertainment_count, total_weibo_count),
                "final_entertainment_topic_ratio": safe_ratio(final_entertainment_count, total_weibo_count),
                "daily_life_topic_ratio": safe_ratio(daily_life_count, total_weibo_count),
                "final_daily_life_topic_ratio": safe_ratio(final_daily_life_count, total_weibo_count),
                "repost_topic_dependency": repost_topic_dependency,
                "explicit_topic_coverage": explicit_topic_coverage,
                "implicit_valid_topic_coverage": implicit_valid_topic_coverage,
                "final_topic_coverage": final_topic_coverage,
                "topic_source_balance_label": determine_topic_source_balance_label(repost_topic_dependency),
                "explicit_topic_profile_reliability": determine_topic_profile_reliability(
                    total_weibo_count,
                    explicit_topic_coverage,
                ),
                "final_topic_profile_reliability": determine_final_topic_profile_reliability(
                    total_weibo_count,
                    final_topic_coverage,
                    avg_final_topic_confidence,
                ),
                "avg_signal_confidence": avg_signal_confidence,
                "avg_final_topic_confidence": avg_final_topic_confidence,
            }
        )

    profile_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return profile_df.sort_values("user_id").reset_index(drop=True)


def validate_profile_df(profile_df: pd.DataFrame, expected_user_count: int) -> None:
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in profile_df.columns]
    if missing_columns:
        raise AssertionError(f"Output dataframe is missing required columns: {missing_columns}")
    if len(profile_df) != expected_user_count:
        raise AssertionError(f"Output row count {len(profile_df)} does not match expected {expected_user_count}")

    ratio_columns = [
        "public_issue_topic_ratio",
        "entertainment_topic_ratio",
        "daily_life_topic_ratio",
        "final_public_issue_topic_ratio",
        "final_entertainment_topic_ratio",
        "final_daily_life_topic_ratio",
        "repost_topic_dependency",
        "explicit_topic_coverage",
        "implicit_valid_topic_coverage",
        "final_topic_coverage",
        "avg_signal_confidence",
        "avg_final_topic_confidence",
    ]
    for column in ratio_columns:
        if not profile_df[column].between(0.0, 1.0, inclusive="both").all():
            raise AssertionError(f"{column} contains values outside [0, 1]")


def save_outputs(profile_df: pd.DataFrame, output_path: Path) -> None:
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet output is supported, got: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_parquet(output_path, index=False)
    LOGGER.info("Saved %d user topic profile rows to %s", len(profile_df), output_path)


def log_summary(input_df: pd.DataFrame, profile_df: pd.DataFrame) -> None:
    avg_weibo_per_user = round(float(input_df.groupby("user_id").size().mean()), 4) if len(profile_df) else 0.0

    LOGGER.info("输入微博级主题表行数: %d", len(input_df))
    LOGGER.info("用户数量: %d", input_df["user_id"].nunique())
    LOGGER.info("输出用户级主题画像表行数: %d", len(profile_df))
    LOGGER.info("平均每个用户微博数: %.4f", avg_weibo_per_user)
    LOGGER.info(
        "public_issue_topic_ratio describe:\n%s",
        profile_df["public_issue_topic_ratio"].describe().to_string(),
    )
    LOGGER.info(
        "entertainment_topic_ratio describe:\n%s",
        profile_df["entertainment_topic_ratio"].describe().to_string(),
    )
    LOGGER.info(
        "daily_life_topic_ratio describe:\n%s",
        profile_df["daily_life_topic_ratio"].describe().to_string(),
    )
    LOGGER.info(
        "repost_topic_dependency describe:\n%s",
        profile_df["repost_topic_dependency"].describe().to_string(),
    )
    LOGGER.info(
        "explicit_topic_coverage describe:\n%s",
        profile_df["explicit_topic_coverage"].describe().to_string(),
    )
    LOGGER.info(
        "final_topic_coverage describe:\n%s",
        profile_df["final_topic_coverage"].describe().to_string(),
    )
    LOGGER.info(
        "topic_source_balance_label value_counts:\n%s",
        profile_df["topic_source_balance_label"].value_counts(dropna=False).to_string(),
    )
    LOGGER.info(
        "explicit_topic_profile_reliability value_counts:\n%s",
        profile_df["explicit_topic_profile_reliability"].value_counts(dropna=False).to_string(),
    )
    LOGGER.info(
        "final_topic_profile_reliability value_counts:\n%s",
        profile_df["final_topic_profile_reliability"].value_counts(dropna=False).to_string(),
    )
    LOGGER.info("top_user_topics 为空的用户数量: %d", int(profile_df["top_user_topics"].isna().sum()))
    LOGGER.info("top_source_topics 为空的用户数量: %d", int(profile_df["top_source_topics"].isna().sum()))
    LOGGER.info("top_all_topics 为空的用户数量: %d", int(profile_df["top_all_topics"].isna().sum()))
    LOGGER.info("top_categories 为空的用户数量: %d", int(profile_df["top_categories"].isna().sum()))
    LOGGER.info(
        "top_implicit_topic_labels 为空的用户数量: %d",
        int(profile_df["top_implicit_topic_labels"].isna().sum()),
    )
    LOGGER.info(
        "top_final_topic_categories 为空的用户数量: %d",
        int(profile_df["top_final_topic_categories"].isna().sum()),
    )
    LOGGER.info(
        "final_category_distribution 为空的用户数量: %d",
        int(profile_df["final_category_distribution"].isna().sum()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate weibo-level fused topic signals into user topic profiles.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH, help="Input weibo topic fusion parquet path.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output final user topic profile parquet path.",
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

    input_df = load_input_dataframe(args.input_path)
    profile_df = build_user_topic_profile(input_df)
    validate_profile_df(profile_df, expected_user_count=input_df["user_id"].nunique())
    log_summary(input_df, profile_df)
    save_outputs(profile_df, args.output_path)


if __name__ == "__main__":
    main()
