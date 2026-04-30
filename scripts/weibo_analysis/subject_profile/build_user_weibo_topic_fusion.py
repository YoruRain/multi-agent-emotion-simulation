from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPLICIT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "user_weibo_topic_signals.parquet"
)
DEFAULT_IMPLICIT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "implicit_topic_clustering"
    / "implicit_topic_clustering_k20.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "user_weibo_topic_fusion.parquet"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "user_weibo_topic_fusion.log"
DEFAULT_PREVIEW_ROWS = 500

EXPLICIT_COLUMNS = [
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
    "explicit_keywords",
    "explicit_topic_categories",
    "signal_confidence",
]
IMPLICIT_SOURCE_COLUMNS = [
    "weibo_id",
    "user_id",
    "analysis_text",
    "analysis_text_length",
    "cluster_id",
    "distance_to_center",
]
OUTPUT_COLUMNS = [
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
    "explicit_keywords",
    "explicit_topic_categories",
    "signal_confidence",
    "implicit_cluster_id",
    "implicit_analysis_text",
    "implicit_analysis_text_length",
    "distance_to_center",
    "implicit_topic_label",
    "implicit_topic_category",
    "implicit_topic_confidence_level",
    "implicit_topic_base_score",
    "cluster_distance_quantile_group",
    "distance_factor",
    "implicit_topic_confidence_score",
    "implicit_topic_valid",
    "final_topic_categories",
    "final_topic_labels",
    "topic_signal_source",
    "final_topic_confidence",
]
NULLABLE_TEXT_OUTPUT_COLUMNS = [
    "implicit_analysis_text",
    "implicit_topic_label",
    "implicit_topic_category",
    "implicit_topic_confidence_level",
    "cluster_distance_quantile_group",
    "final_topic_categories",
    "final_topic_labels",
]

CONFIDENCE_BASE = {
    "high": 0.85,
    "medium_high": 0.75,
    "medium": 0.60,
    "low": 0.35,
}

CLUSTER_TOPIC_MAPPING = {
    0: {"implicit_topic_label": "小说、影视剧与剧情评价", "implicit_topic_category": "娱乐文化", "implicit_topic_confidence_level": "high"},
    1: {"implicit_topic_label": "消费购物、商品信息与商业内容", "implicit_topic_category": "广告营销", "implicit_topic_confidence_level": "medium_high"},
    2: {"implicit_topic_label": "娱乐舆论、粉圈争议与明星事件", "implicit_topic_category": "娱乐文化", "implicit_topic_confidence_level": "medium_high"},
    3: {"implicit_topic_label": "公益活动、账号运营与推广内容", "implicit_topic_category": "广告营销", "implicit_topic_confidence_level": "medium"},
    4: {"implicit_topic_label": "萌宠、亲子、生活与趣味分享", "implicit_topic_category": "日常生活", "implicit_topic_confidence_level": "medium_high"},
    5: {"implicit_topic_label": "音乐、演出与文艺娱乐内容", "implicit_topic_category": "娱乐文化", "implicit_topic_confidence_level": "medium_high"},
    6: {"implicit_topic_label": "旅行、地点打卡与图文分享", "implicit_topic_category": "日常生活", "implicit_topic_confidence_level": "high"},
    7: {"implicit_topic_label": "人生感悟、价值思考与自我成长", "implicit_topic_category": "情感表达", "implicit_topic_confidence_level": "high"},
    8: {"implicit_topic_label": "吐槽、愤怒与社会事件短评", "implicit_topic_category": "情感表达", "implicit_topic_confidence_level": "medium"},
    9: {"implicit_topic_label": "明星应援、偶像物料与粉圈内容", "implicit_topic_category": "娱乐文化", "implicit_topic_confidence_level": "high"},
    10: {"implicit_topic_label": "节日祝福、新年愿望与年度记录", "implicit_topic_category": "日常生活", "implicit_topic_confidence_level": "high"},
    11: {"implicit_topic_label": "游戏、赛事、票务与线上娱乐互动", "implicit_topic_category": "游戏动漫", "implicit_topic_confidence_level": "medium"},
    12: {"implicit_topic_label": "粉圈冲突、娱乐八卦与平台舆论", "implicit_topic_category": "娱乐文化", "implicit_topic_confidence_level": "medium_high"},
    13: {"implicit_topic_label": "日常疲惫、失眠、通勤与生活烦恼", "implicit_topic_category": "情感表达", "implicit_topic_confidence_level": "high"},
    14: {"implicit_topic_label": "性别议题、家庭婚恋与社会公平讨论", "implicit_topic_category": "社会公共事件", "implicit_topic_confidence_level": "high"},
    15: {"implicit_topic_label": "恋爱、CP、追剧与亲密关系讨论", "implicit_topic_category": "娱乐文化", "implicit_topic_confidence_level": "medium_high"},
    16: {"implicit_topic_label": "睡眠、身体健康与心理压力", "implicit_topic_category": "日常生活", "implicit_topic_confidence_level": "high"},
    17: {"implicit_topic_label": "饮食、美食与食品健康", "implicit_topic_category": "日常生活", "implicit_topic_confidence_level": "high"},
    18: {"implicit_topic_label": "碎片化回忆、短句记录与低密度表达", "implicit_topic_category": "其他", "implicit_topic_confidence_level": "low"},
    19: {"implicit_topic_label": "积极情绪、美好记录与审美表达", "implicit_topic_category": "情感表达", "implicit_topic_confidence_level": "high"},
}


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


def load_parquet(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported, got: {path}")

    try:
        df = pd.read_parquet(path, columns=required_columns)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow or fastparquet in the active Python environment."
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Failed to read parquet file: {path}. "
            "The file may be corrupted or incompatible with the current pyarrow runtime."
        ) from exc

    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Input dataframe {path} is missing required columns: {missing}")
    return df


def clip_score(value: Any) -> float:
    score = pd.to_numeric(value, errors="coerce")
    if pd.isna(score):
        return 0.0
    return float(min(max(score, 0.0), 1.0))


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def has_non_empty_text(value: Any) -> bool:
    return bool(normalize_text(value))


def validate_unique_weibo_ids(df: pd.DataFrame, name: str) -> None:
    duplicate_mask = df["weibo_id"].duplicated(keep=False)
    if not duplicate_mask.any():
        return

    duplicate_ids = df.loc[duplicate_mask, "weibo_id"].dropna().astype(str).unique().tolist()
    preview = duplicate_ids[:10]
    raise ValueError(
        f"{name} contains duplicated weibo_id values, which would break one-to-one merge. "
        f"duplicate_count={len(duplicate_ids)}, sample={preview}"
    )


def prepare_explicit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["signal_confidence"] = prepared["signal_confidence"].map(clip_score)
    return prepared


def prepare_implicit_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    prepared = df.copy().rename(
        columns={
            "cluster_id": "implicit_cluster_id",
            "analysis_text": "implicit_analysis_text",
            "analysis_text_length": "implicit_analysis_text_length",
        }
    )
    prepared["distance_to_center"] = pd.to_numeric(prepared["distance_to_center"], errors="coerce")
    prepared["implicit_cluster_id"] = pd.to_numeric(prepared["implicit_cluster_id"], errors="coerce")

    unknown_cluster_ids = sorted(
        {
            int(cluster_id)
            for cluster_id in prepared["implicit_cluster_id"].dropna().tolist()
            if int(cluster_id) not in CLUSTER_TOPIC_MAPPING
        }
    )
    if unknown_cluster_ids:
        LOGGER.warning(
            "Found cluster_id values missing from CLUSTER_TOPIC_MAPPING: count=%d ids=%s",
            len(unknown_cluster_ids),
            unknown_cluster_ids,
        )

    mapping_df = (
        pd.DataFrame.from_dict(CLUSTER_TOPIC_MAPPING, orient="index")
        .rename_axis("implicit_cluster_id")
        .reset_index()
    )
    mapping_df["implicit_cluster_id"] = pd.to_numeric(mapping_df["implicit_cluster_id"], errors="coerce")
    prepared = prepared.merge(mapping_df, on="implicit_cluster_id", how="left")

    prepared["implicit_topic_label"] = prepared["implicit_topic_label"].fillna("")
    prepared["implicit_topic_category"] = prepared["implicit_topic_category"].fillna("")
    prepared["implicit_topic_confidence_level"] = prepared["implicit_topic_confidence_level"].fillna("")
    prepared["implicit_topic_base_score"] = prepared["implicit_topic_confidence_level"].map(
        lambda value: float(CONFIDENCE_BASE.get(normalize_text(value), 0.0))
    )

    quantiles = prepared.groupby("implicit_cluster_id", dropna=True)["distance_to_center"].quantile(
        [0.25, 0.50, 0.75, 0.90]
    ).unstack()
    quantiles = quantiles.rename(columns={0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"})
    prepared = prepared.merge(quantiles, on="implicit_cluster_id", how="left")

    prepared["cluster_distance_quantile_group"] = ""
    prepared["distance_factor"] = 0.0

    valid_mask = prepared["implicit_cluster_id"].notna() & prepared["distance_to_center"].notna()
    core_mask = valid_mask & (prepared["distance_to_center"] <= prepared["q25"])
    typical_mask = valid_mask & (prepared["distance_to_center"] > prepared["q25"]) & (prepared["distance_to_center"] <= prepared["q50"])
    middle_mask = valid_mask & (prepared["distance_to_center"] > prepared["q50"]) & (prepared["distance_to_center"] <= prepared["q75"])
    edge_mask = valid_mask & (prepared["distance_to_center"] > prepared["q75"]) & (prepared["distance_to_center"] <= prepared["q90"])
    far_edge_mask = valid_mask & (prepared["distance_to_center"] > prepared["q90"])

    prepared.loc[core_mask, ["cluster_distance_quantile_group", "distance_factor"]] = ["center_core", 1.10]
    prepared.loc[typical_mask, ["cluster_distance_quantile_group", "distance_factor"]] = ["center_typical", 1.00]
    prepared.loc[middle_mask, ["cluster_distance_quantile_group", "distance_factor"]] = ["middle", 0.90]
    prepared.loc[edge_mask, ["cluster_distance_quantile_group", "distance_factor"]] = ["edge", 0.75]
    prepared.loc[far_edge_mask, ["cluster_distance_quantile_group", "distance_factor"]] = ["far_edge", 0.60]

    prepared["implicit_topic_confidence_score"] = (
        prepared["implicit_topic_base_score"] * pd.to_numeric(prepared["distance_factor"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    prepared["implicit_topic_valid"] = (
        prepared["implicit_topic_category"].map(has_non_empty_text)
        & (prepared["implicit_topic_category"] != "其他")
        & (prepared["implicit_topic_confidence_score"] >= 0.45)
    )

    prepared = prepared.drop(columns=["q25", "q50", "q75", "q90"])
    return prepared, unknown_cluster_ids


def build_final_topic_fields(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["final_topic_categories"] = ""
    result["final_topic_labels"] = ""
    result["topic_signal_source"] = "无主题"
    result["final_topic_confidence"] = 0.0

    explicit_mask = result["explicit_topic_categories"].map(has_non_empty_text)
    implicit_label_mask = result["implicit_topic_label"].map(has_non_empty_text)
    implicit_valid_mask = result["implicit_topic_valid"].fillna(False).astype(bool)

    result.loc[explicit_mask, "final_topic_categories"] = result.loc[explicit_mask, "explicit_topic_categories"].fillna("")
    result.loc[explicit_mask, "final_topic_labels"] = result.loc[explicit_mask, "explicit_topic_categories"].fillna("")
    result.loc[explicit_mask, "topic_signal_source"] = "显式主题"
    result.loc[explicit_mask, "final_topic_confidence"] = result.loc[explicit_mask, "signal_confidence"].map(clip_score)

    implicit_valid_only_mask = (~explicit_mask) & implicit_valid_mask
    result.loc[implicit_valid_only_mask, "final_topic_categories"] = result.loc[implicit_valid_only_mask, "implicit_topic_category"].fillna("")
    result.loc[implicit_valid_only_mask, "final_topic_labels"] = result.loc[implicit_valid_only_mask, "implicit_topic_label"].fillna("")
    result.loc[implicit_valid_only_mask, "topic_signal_source"] = "隐式主题"
    result.loc[implicit_valid_only_mask, "final_topic_confidence"] = result.loc[
        implicit_valid_only_mask, "implicit_topic_confidence_score"
    ].map(clip_score)

    implicit_low_conf_mask = (~explicit_mask) & (~implicit_valid_mask) & implicit_label_mask
    result.loc[implicit_low_conf_mask, "final_topic_categories"] = ""
    result.loc[implicit_low_conf_mask, "final_topic_labels"] = result.loc[implicit_low_conf_mask, "implicit_topic_label"].fillna("")
    result.loc[implicit_low_conf_mask, "topic_signal_source"] = "隐式低置信度"
    result.loc[implicit_low_conf_mask, "final_topic_confidence"] = result.loc[
        implicit_low_conf_mask, "implicit_topic_confidence_score"
    ].map(clip_score)

    result["final_topic_confidence"] = result["final_topic_confidence"].map(clip_score)
    return result


def replace_empty_strings_with_none(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = result[column].map(lambda value: None if normalize_text(value) == "" else value)
    return result


def normalize_bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: bool(value) if pd.notna(value) else False)


def build_fusion_table(df_explicit: pd.DataFrame, df_implicit: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_unique_weibo_ids(df_explicit, "Explicit topic dataframe")
    validate_unique_weibo_ids(df_implicit, "Implicit clustering dataframe")

    explicit_prepared = prepare_explicit_dataframe(df_explicit)
    implicit_prepared, unknown_cluster_ids = prepare_implicit_dataframe(df_implicit)

    implicit_merge_columns = [
        "weibo_id",
        "implicit_cluster_id",
        "implicit_analysis_text",
        "implicit_analysis_text_length",
        "distance_to_center",
        "implicit_topic_label",
        "implicit_topic_category",
        "implicit_topic_confidence_level",
        "implicit_topic_base_score",
        "cluster_distance_quantile_group",
        "distance_factor",
        "implicit_topic_confidence_score",
        "implicit_topic_valid",
    ]
    merged = explicit_prepared.merge(implicit_prepared.loc[:, implicit_merge_columns], on="weibo_id", how="left")
    if len(merged) != len(explicit_prepared):
        raise AssertionError(
            f"Output row count after left merge changed from {len(explicit_prepared)} to {len(merged)}."
        )

    for column in [
        "implicit_analysis_text",
        "implicit_topic_label",
        "implicit_topic_category",
        "implicit_topic_confidence_level",
        "cluster_distance_quantile_group",
    ]:
        merged[column] = merged[column].fillna("")

    for column in [
        "implicit_topic_base_score",
        "distance_factor",
        "implicit_topic_confidence_score",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

    merged["implicit_topic_valid"] = normalize_bool_series(merged["implicit_topic_valid"])
    merged = build_final_topic_fields(merged)
    merged = replace_empty_strings_with_none(merged, NULLABLE_TEXT_OUTPUT_COLUMNS)
    merged = merged.loc[:, OUTPUT_COLUMNS].copy()

    final_topic_non_empty_count = int(merged["final_topic_categories"].map(has_non_empty_text).sum())
    stats = {
        "explicit_rows": int(len(df_explicit)),
        "implicit_rows": int(len(df_implicit)),
        "matched_implicit_count": int(merged["implicit_cluster_id"].notna().sum()),
        "unknown_cluster_count": int(len(unknown_cluster_ids)),
        "unknown_cluster_ids": unknown_cluster_ids,
        "explicit_non_empty_count": int(merged["explicit_topic_categories"].map(has_non_empty_text).sum()),
        "implicit_valid_count": int(merged["implicit_topic_valid"].sum()),
        "topic_signal_source_counts": merged["topic_signal_source"].value_counts(dropna=False),
        "final_topic_non_empty_count": final_topic_non_empty_count,
        "final_topic_coverage": float(final_topic_non_empty_count / len(merged)) if len(merged) else 0.0,
        "final_topic_confidence_describe": merged["final_topic_confidence"].describe(),
    }
    return merged, stats


def validate_output(df_output: pd.DataFrame, expected_rows: int) -> None:
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in df_output.columns]
    if missing_columns:
        raise AssertionError(f"Output dataframe is missing required columns: {missing_columns}")
    if len(df_output) != expected_rows:
        raise AssertionError(f"Output row count {len(df_output)} does not match expected {expected_rows}")
    if not df_output["final_topic_confidence"].between(0.0, 1.0, inclusive="both").all():
        raise AssertionError("final_topic_confidence contains values outside [0, 1]")
    if not df_output["implicit_topic_confidence_score"].between(0.0, 1.0, inclusive="both").all():
        raise AssertionError("implicit_topic_confidence_score contains values outside [0, 1]")


def save_output(df: pd.DataFrame, output_path: Path, preview_rows: int = DEFAULT_PREVIEW_ROWS) -> Path:
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet output is supported, got: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    preview_path = output_path.with_name(f"{output_path.stem}_preview.csv")
    df.head(preview_rows).to_csv(preview_path, index=False, encoding="utf-8-sig")

    LOGGER.info("Saved %d rows to %s", len(df), output_path)
    LOGGER.info("Saved preview CSV (%d rows) to %s", min(len(df), preview_rows), preview_path)
    return preview_path


def log_summary(stats: dict[str, Any], output_path: Path, preview_path: Path) -> None:
    LOGGER.info("显式主题表行数: %d", stats["explicit_rows"])
    LOGGER.info("隐式聚类表行数: %d", stats["implicit_rows"])
    LOGGER.info("成功匹配到隐式聚类结果的微博数: %d", stats["matched_implicit_count"])
    LOGGER.info(
        "cluster_id 未出现在 CLUSTER_TOPIC_MAPPING 中的数量: %d, ID 列表: %s",
        stats["unknown_cluster_count"],
        stats["unknown_cluster_ids"],
    )
    LOGGER.info("explicit_topic_categories 非空微博数: %d", stats["explicit_non_empty_count"])
    LOGGER.info("implicit_topic_valid 为 True 的微博数: %d", stats["implicit_valid_count"])
    LOGGER.info("topic_signal_source value_counts():\n%s", stats["topic_signal_source_counts"].to_string())
    LOGGER.info("final_topic_categories 非空微博数: %d", stats["final_topic_non_empty_count"])
    LOGGER.info("final_topic_coverage: %.4f", stats["final_topic_coverage"])
    LOGGER.info("final_topic_confidence describe:\n%s", stats["final_topic_confidence_describe"].to_string())
    LOGGER.info("输出 parquet 路径: %s", output_path)
    LOGGER.info("输出 csv 预览路径: %s", preview_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse explicit weibo topic signals with implicit clustering topics."
    )
    parser.add_argument(
        "--explicit_input_path",
        type=Path,
        default=DEFAULT_EXPLICIT_INPUT_PATH,
        help="Explicit topic parquet path.",
    )
    parser.add_argument(
        "--implicit_input_path",
        type=Path,
        default=DEFAULT_IMPLICIT_INPUT_PATH,
        help="Implicit clustering parquet path.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output fusion parquet path.",
    )
    parser.add_argument(
        "--preview_rows",
        type=int,
        default=DEFAULT_PREVIEW_ROWS,
        help="Number of rows to save into preview CSV.",
    )
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for log files.")
    parser.add_argument("--log_file", type=Path, default=None, help="Optional explicit log file path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, log_dir=args.log_dir, log_file=args.log_file)
    LOGGER.info("Using explicit_input_path=%s", args.explicit_input_path)
    LOGGER.info("Using implicit_input_path=%s", args.implicit_input_path)
    LOGGER.info("Using output_path=%s", args.output_path)

    df_explicit = load_parquet(args.explicit_input_path, EXPLICIT_COLUMNS)
    df_implicit = load_parquet(args.implicit_input_path, IMPLICIT_SOURCE_COLUMNS)
    fusion_df, stats = build_fusion_table(df_explicit, df_implicit)
    validate_output(fusion_df, expected_rows=len(df_explicit))
    preview_path = save_output(fusion_df, args.output_path, preview_rows=max(args.preview_rows, 1))
    log_summary(stats, output_path=args.output_path, preview_path=preview_path)


if __name__ == "__main__":
    main()
