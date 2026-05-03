from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USER_WEIBO_PATH = PROJECT_ROOT / "data" / "cleaned" / "user_weibo.parquet"
DEFAULT_USER_INFO_PATH = PROJECT_ROOT / "data" / "high_quality" / "user_info.parquet"
DEFAULT_SOURCE_CREATOR_PATH = PROJECT_ROOT / "data" / "high_quality" / "source_creator_info.parquet"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "propagation_profile"
    / "user_propagation_profile.parquet"
)

USER_INFO_REQUIRED_COLUMNS = {
    "user_id",
    "weibo_hq_count",
    "active_days",
    "original_ratio",
    "follower_count",
    "verified",
    "user_rank",
}
USER_WEIBO_REQUIRED_COLUMNS = {
    "weibo_id",
    "user_id",
    "content",
    "cleaned_content",
    "text_quality",
    "engagement",
    "is_repost",
    "reposted_weibo_id",
}
SOURCE_CREATOR_REQUIRED_COLUMNS = {
    "user_id",
    "follower_count",
    "verified_type_name",
    "user_rank",
}
OUTPUT_COLUMNS = [
    "user_id",
    "weibo_hq_count",
    "active_days",
    "propagation_activity_level",
    "original_ratio",
    "repost_ratio",
    "repost_with_comment_ratio",
    "source_media_ratio",
    "source_government_ratio",
    "source_institution_ratio",
    "source_personal_verified_ratio",
    "source_high_follower_ratio",
    "high_personal_verified_ratio",
    "media_dependency_score",
    "kol_sensitivity_score",
    "avg_engagement",
    "high_engagement_weibo_ratio",
    "influence_score",
    "influence_level",
    "propagation_role",
]
RATIO_COLUMNS = [
    "original_ratio",
    "repost_ratio",
    "repost_with_comment_ratio",
    "source_media_ratio",
    "source_government_ratio",
    "source_institution_ratio",
    "source_personal_verified_ratio",
    "source_high_follower_ratio",
    "high_personal_verified_ratio",
    "media_dependency_score",
    "kol_sensitivity_score",
    "high_engagement_weibo_ratio",
    "influence_score",
]

LOW_INFORMATION_PATTERNS = [
    (
        re.compile(
            r"^\s*("
            r"[转轉][发發]?(至?微博)?|"
            r"Repost|"
            r"[存码马](住|下|克)?|"
            r"转一个|必须转|"
            r"签到|"
            r"收藏(了)?"
            r")\s*$",
            re.IGNORECASE,
        ),
        "占位/功能互动",
    ),
    (re.compile(r"^#[^#]+#(\s*#[^#]+#)*$"), "纯话题占位"),
    (re.compile(r"^\d+$"), "纯数字"),
    (re.compile(r"^[^\w\u4e00-\u9fff\U00010000-\U0010FFFF]+$"), "纯符号"),
    (re.compile(r"^[a-zA-Z]+$"), "纯英文字母"),
    (re.compile(r"^(@[^\s@]+\s*)+$"), "纯@用户"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build user-level Weibo propagation profile.")
    parser.add_argument("--user-weibo-path", type=Path, default=DEFAULT_USER_WEIBO_PATH)
    parser.add_argument("--user-info-path", type=Path, default=DEFAULT_USER_INFO_PATH)
    parser.add_argument("--source-creator-path", type=Path, default=DEFAULT_SOURCE_CREATOR_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported: {path}")
    return pd.read_parquet(path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet output is supported: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def require_columns(df: pd.DataFrame, required_columns: set[str], name: str) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def safe_log1p(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return np.log1p(numeric.clip(lower=0))


def minmax_normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return ((numeric - min_value) / (max_value - min_value)).clip(0, 1)


def assign_level_by_quantile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    q33 = numeric.quantile(0.33)
    q67 = numeric.quantile(0.67)
    if pd.isna(q33) or pd.isna(q67):
        return pd.Series("low", index=series.index, dtype="object")
    if q33 == q67:
        fallback = "low" if q33 <= 0 else "medium"
        return pd.Series(fallback, index=series.index, dtype="object")

    labels = pd.Series("medium", index=series.index, dtype="object")
    labels.loc[numeric <= q33] = "low"
    labels.loc[numeric >= q67] = "high"
    return labels


def is_low_information_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(pattern.fullmatch(stripped) for pattern, _ in LOW_INFORMATION_PATTERNS)


def extract_repost_comment(text: str | None) -> str:
    if text is None or pd.isna(text):
        return ""
    comment = str(text).split("//", 1)[0].strip()
    if is_low_information_text(comment):
        return ""
    return comment


def normalize_verified_type(verified_type_name: Any) -> str:
    if verified_type_name is None or pd.isna(verified_type_name):
        return "未知"
    value = str(verified_type_name).strip()
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


def build_propagation_roles(
    row: pd.Series,
    influence_q85: float,
    kol_q75: float,
    high_engagement_q75: float,
) -> str:
    roles: list[str] = []
    if row["propagation_activity_level"] == "low":
        roles.append("低活跃观察者")
    if row["original_ratio"] >= 0.6:
        roles.append("原创表达者")
    if row["repost_ratio"] >= 0.75:
        roles.append("转发扩散者")
    if row["repost_with_comment_ratio"] >= 0.5:
        roles.append("转发评论者")
    if row["media_dependency_score"] >= 0.5:
        roles.append("媒体信息跟随者")
    if row["kol_sensitivity_score"] >= kol_q75:
        roles.append("KOL 敏感型用户")
    # 对照旧规则：row["influence_score"] >= influence_q85 and row["propagation_activity_level"] != "low"
    if row["influence_score"] >= influence_q85 and row["high_engagement_weibo_ratio"] >= high_engagement_q75:
        roles.append("潜在影响者")
    return ",".join(roles) if roles else "普通参与者"


def divide_or_zero(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return (num / den.replace(0, np.nan)).fillna(0.0)


def valid_repost_id_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & (numeric != -1)


def prepare_inputs(
    df_user_info: pd.DataFrame,
    df_user_weibo: pd.DataFrame,
    df_source_creator: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_info = df_user_info.copy()
    user_weibo_all = df_user_weibo.copy()
    source_creator = df_source_creator.copy()

    user_info["user_id"] = pd.to_numeric(user_info["user_id"], errors="coerce")
    user_weibo_all["user_id"] = pd.to_numeric(user_weibo_all["user_id"], errors="coerce")
    user_weibo_all["weibo_id"] = pd.to_numeric(user_weibo_all["weibo_id"], errors="coerce")
    user_weibo_all["reposted_weibo_id"] = pd.to_numeric(user_weibo_all["reposted_weibo_id"], errors="coerce")
    source_creator["user_id"] = pd.to_numeric(source_creator["user_id"], errors="coerce")

    for column in ["weibo_hq_count", "active_days", "original_ratio", "follower_count", "user_rank"]:
        user_info[column] = pd.to_numeric(user_info[column], errors="coerce")
    for column in ["text_quality", "engagement"]:
        user_weibo_all[column] = pd.to_numeric(user_weibo_all[column], errors="coerce")
    for column in ["follower_count", "user_rank"]:
        source_creator[column] = pd.to_numeric(source_creator[column], errors="coerce")

    user_info["verified"] = user_info["verified"].fillna(False).astype(bool)
    user_weibo_all["is_repost"] = user_weibo_all["is_repost"].fillna(False).astype(bool)

    return user_info, user_weibo_all, source_creator


def build_user_repost_stats(user_weibo: pd.DataFrame) -> pd.DataFrame:
    counts = user_weibo.groupby("user_id").agg(
        weibo_count=("weibo_id", "size"),
        original_weibo_count=("is_repost", lambda s: int((~s).sum())),
        repost_weibo_count=("is_repost", lambda s: int(s.sum())),
    )
    valid_reposts = user_weibo[user_weibo["is_repost"] & valid_repost_id_mask(user_weibo["reposted_weibo_id"])]
    valid_repost_count = valid_reposts.groupby("user_id").size().rename("valid_repost_count")

    if valid_reposts.empty:
        repost_with_comment_count = pd.Series(dtype="int64", name="repost_with_comment_count")
    else:
        valid_reposts = valid_reposts.copy()
        valid_reposts["repost_comment"] = valid_reposts["content"].map(extract_repost_comment)
        valid_reposts["has_valid_repost_comment"] = (
            valid_reposts["text_quality"].fillna(0).eq(3) & valid_reposts["repost_comment"].ne("")
        )
        repost_with_comment_count = (
            valid_reposts.groupby("user_id")["has_valid_repost_comment"].sum().rename("repost_with_comment_count")
        )

    stats = counts.join(valid_repost_count, how="left").join(repost_with_comment_count, how="left")
    return stats.fillna(0)


def build_source_relation_stats(
    user_weibo: pd.DataFrame,
    df_user_weibo: pd.DataFrame,
    df_source_creator: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int, int]:
    user_repost_weibo = user_weibo[
        user_weibo["is_repost"] & valid_repost_id_mask(user_weibo["reposted_weibo_id"])
    ].copy()
    valid_repost_records = len(user_repost_weibo)
    if user_repost_weibo.empty:
        empty = pd.DataFrame(
            columns=[
                "source_media_ratio",
                "source_government_ratio",
                "source_institution_ratio",
                "source_personal_verified_ratio",
                "source_high_follower_ratio",
                "high_personal_verified_ratio",
            ]
        )
        empty.index.name = "user_id"
        return empty, valid_repost_records, 0, 0

    source_weibo = (
        df_user_weibo[["weibo_id", "user_id"]]
        .dropna(subset=["weibo_id"])
        .drop_duplicates(subset=["weibo_id"])
        .rename(columns={"weibo_id": "source_weibo_id", "user_id": "source_user_id"})
    )
    relation = user_repost_weibo[
        ["user_id", "weibo_id", "reposted_weibo_id", "cleaned_content", "content", "text_quality", "engagement"]
    ].merge(source_weibo, left_on="reposted_weibo_id", right_on="source_weibo_id", how="left")
    source_weibo_matched_records = int(relation["source_user_id"].notna().sum())

    creator = df_source_creator[["user_id", "verified_type_name", "follower_count", "user_rank"]].rename(
        columns={
            "user_id": "source_user_id",
            "verified_type_name": "source_verified_type_name",
            "follower_count": "source_follower_count",
            "user_rank": "source_user_rank",
        }
    )
    creator["source_creator_matched"] = True
    relation = relation.merge(creator, on="source_user_id", how="left")
    relation["source_creator_matched"] = relation["source_creator_matched"].eq(True)
    source_creator_matched_records = int(relation["source_creator_matched"].sum())

    high_follower_threshold = df_source_creator["follower_count"].quantile(0.75)
    if pd.isna(high_follower_threshold):
        high_follower_threshold = np.inf

    matched_relation = relation[relation["source_creator_matched"]].copy()
    if matched_relation.empty:
        empty = pd.DataFrame(
            columns=[
                "source_media_ratio",
                "source_government_ratio",
                "source_institution_ratio",
                "source_personal_verified_ratio",
                "source_high_follower_ratio",
                "high_personal_verified_ratio",
            ]
        )
        empty.index.name = "user_id"
        return empty, valid_repost_records, source_weibo_matched_records, source_creator_matched_records

    matched_relation["source_type"] = matched_relation["source_verified_type_name"].map(normalize_verified_type)
    matched_relation["source_high_follower"] = (
        pd.to_numeric(matched_relation["source_follower_count"], errors="coerce").fillna(-1) >= high_follower_threshold
    )
    matched_relation["high_personal_verified"] = (
        matched_relation["source_high_follower"] & matched_relation["source_type"].eq("个人认证")
    )

    stats = matched_relation.groupby("user_id").agg(
        source_media_ratio=("source_type", lambda s: float((s == "媒体").mean())),
        source_government_ratio=("source_type", lambda s: float((s == "政府").mean())),
        source_institution_ratio=("source_type", lambda s: float((s == "机构").mean())),
        source_personal_verified_ratio=("source_type", lambda s: float((s == "个人认证").mean())),
        source_high_follower_ratio=("source_high_follower", "mean"),
        high_personal_verified_ratio=("high_personal_verified", "mean"),
    )
    return stats, valid_repost_records, source_weibo_matched_records, source_creator_matched_records


def build_engagement_stats(user_weibo: pd.DataFrame) -> pd.DataFrame:
    if user_weibo.empty:
        empty = pd.DataFrame(columns=["avg_engagement", "high_engagement_weibo_ratio"])
        empty.index.name = "user_id"
        return empty

    engagement = pd.to_numeric(user_weibo["engagement"], errors="coerce").fillna(0.0)
    high_engagement_threshold = engagement.quantile(0.75)
    prepared = user_weibo[["user_id"]].copy()
    prepared["engagement"] = engagement
    prepared["is_high_engagement_weibo"] = prepared["engagement"] >= high_engagement_threshold
    return prepared.groupby("user_id").agg(
        avg_engagement=("engagement", "mean"),
        high_engagement_weibo_ratio=("is_high_engagement_weibo", "mean"),
    )


def build_user_propagation_profile(
    df_user_info: pd.DataFrame,
    df_user_weibo: pd.DataFrame,
    df_source_creator: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    user_info, user_weibo_all, source_creator = prepare_inputs(df_user_info, df_user_weibo, df_source_creator)
    modeling_user_ids = set(user_info["user_id"].dropna())
    user_weibo = user_weibo_all[user_weibo_all["user_id"].isin(modeling_user_ids)].copy()

    base = user_info[["user_id", "weibo_hq_count", "active_days", "original_ratio", "follower_count", "verified", "user_rank"]].copy()
    base["weibo_hq_count"] = base["weibo_hq_count"].fillna(0.0)
    base["active_days"] = base["active_days"].fillna(0.0)

    repost_stats = build_user_repost_stats(user_weibo)
    source_stats, valid_repost_records, source_weibo_matched_records, source_creator_matched_records = (
        build_source_relation_stats(user_weibo, user_weibo_all, source_creator)
    )
    engagement_stats = build_engagement_stats(user_weibo)

    profile = base.merge(repost_stats, on="user_id", how="left")
    profile = profile.merge(source_stats, on="user_id", how="left")
    profile = profile.merge(engagement_stats, on="user_id", how="left")
    profile = profile.fillna(
        {
            "weibo_count": 0,
            "original_weibo_count": 0,
            "repost_weibo_count": 0,
            "valid_repost_count": 0,
            "repost_with_comment_count": 0,
            "source_media_ratio": 0,
            "source_government_ratio": 0,
            "source_institution_ratio": 0,
            "source_personal_verified_ratio": 0,
            "source_high_follower_ratio": 0,
            "high_personal_verified_ratio": 0,
            "avg_engagement": 0,
            "high_engagement_weibo_ratio": 0,
        }
    )

    computed_original_ratio = divide_or_zero(profile["original_weibo_count"], profile["weibo_count"])
    profile["original_ratio"] = pd.to_numeric(profile["original_ratio"], errors="coerce")
    profile["original_ratio"] = profile["original_ratio"].where(profile["original_ratio"].notna(), computed_original_ratio)
    profile["original_ratio"] = profile["original_ratio"].fillna(0.0).clip(0, 1)
    profile["repost_ratio"] = (1.0 - profile["original_ratio"]).clip(0, 1)
    profile["repost_with_comment_ratio"] = divide_or_zero(
        profile["repost_with_comment_count"], profile["valid_repost_count"]
    ).clip(0, 1)

    profile["media_dependency_score"] = (
        profile["source_media_ratio"]
        + 0.7 * profile["source_government_ratio"]
        + 0.5 * profile["source_institution_ratio"]
    ).clip(0, 1)
    profile["kol_sensitivity_score"] = (
        0.7 * profile["high_personal_verified_ratio"] + 0.3 * profile["source_high_follower_ratio"]
    ).clip(0, 1)

    activity_score = 0.5 * minmax_normalize(profile["weibo_hq_count"]) + 0.5 * minmax_normalize(profile["active_days"])
    profile["propagation_activity_level"] = assign_level_by_quantile(activity_score)

    follower_score = minmax_normalize(safe_log1p(profile["follower_count"]))
    engagement_score = minmax_normalize(safe_log1p(profile["avg_engagement"]))
    verified_score = profile["verified"].fillna(False).astype(bool).astype(float)
    user_rank_score = minmax_normalize(profile["user_rank"])
    profile["influence_score"] = (
        0.4 * follower_score + 0.3 * engagement_score + 0.2 * verified_score + 0.1 * user_rank_score
    ).clip(0, 1)
    profile["influence_level"] = assign_level_by_quantile(profile["influence_score"])
    influence_q85 = float(profile["influence_score"].quantile(0.85))
    kol_q75 = float(profile["kol_sensitivity_score"].quantile(0.75))
    high_engagement_q75 = float(profile["high_engagement_weibo_ratio"].quantile(0.75))
    profile["propagation_role"] = profile.apply(
        build_propagation_roles,
        axis=1,
        influence_q85=influence_q85,
        kol_q75=kol_q75,
        high_engagement_q75=high_engagement_q75,
    )

    for column in RATIO_COLUMNS:
        profile[column] = pd.to_numeric(profile[column], errors="coerce").fillna(0.0).clip(0, 1)
    profile["avg_engagement"] = pd.to_numeric(profile["avg_engagement"], errors="coerce").fillna(0.0)
    profile["weibo_hq_count"] = pd.to_numeric(profile["weibo_hq_count"], errors="coerce").fillna(0.0)
    profile["active_days"] = pd.to_numeric(profile["active_days"], errors="coerce").fillna(0.0)

    output = profile[OUTPUT_COLUMNS].copy()
    check_quality(output, expected_user_count=len(user_info))

    checks = {
        "valid_repost_records": valid_repost_records,
        "source_weibo_matched_records": source_weibo_matched_records,
        "source_creator_matched_records": source_creator_matched_records,
    }
    return output, checks


def check_quality(df: pd.DataFrame, expected_user_count: int) -> None:
    if len(df) != expected_user_count:
        raise ValueError(f"Output user count mismatch: expected {expected_user_count}, got {len(df)}")

    bad_ratio_columns = []
    for column in RATIO_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            bad_ratio_columns.append(column)
    if bad_ratio_columns:
        raise ValueError(f"Ratio columns outside [0, 1] or null: {bad_ratio_columns}")

    for column in ["propagation_activity_level", "influence_level", "propagation_role"]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains empty values")


def print_checks(
    df_user_info: pd.DataFrame,
    df_profile: pd.DataFrame,
    relation_checks: dict[str, int],
    output_path: Path,
) -> None:
    print(f"df_user_info 用户数: {len(df_user_info)}")
    print(f"输出画像用户数: {len(df_profile)}")
    print(f"有效转发记录数: {relation_checks['valid_repost_records']}")
    print(f"成功连接源微博的转发记录数: {relation_checks['source_weibo_matched_records']}")
    print(f"成功连接源作者信息的转发记录数: {relation_checks['source_creator_matched_records']}")
    print("propagation_activity_level 分布:")
    print(df_profile["propagation_activity_level"].value_counts(dropna=False).to_string())
    print("influence_level 分布:")
    print(df_profile["influence_level"].value_counts(dropna=False).to_string())
    print("propagation_role 高频统计前 10 项:")
    role_counts = (
        df_profile["propagation_role"]
        .str.split(",")
        .explode()
        .value_counts()
        .head(10)
    )
    print(role_counts.to_string())
    print(f"输出文件路径: {output_path}")


def main() -> None:
    args = parse_args()
    df_user_weibo = load_dataframe(args.user_weibo_path)
    df_user_info = load_dataframe(args.user_info_path)
    df_source_creator = load_dataframe(args.source_creator_path)

    require_columns(df_user_info, USER_INFO_REQUIRED_COLUMNS, "df_user_info")
    require_columns(df_user_weibo, USER_WEIBO_REQUIRED_COLUMNS, "df_user_weibo")
    require_columns(df_source_creator, SOURCE_CREATOR_REQUIRED_COLUMNS, "df_source_creator")

    df_user_propagation_profile, relation_checks = build_user_propagation_profile(
        df_user_info=df_user_info,
        df_user_weibo=df_user_weibo,
        df_source_creator=df_source_creator,
    )
    save_dataframe(df_user_propagation_profile, args.output_path)
    print_checks(df_user_info, df_user_propagation_profile, relation_checks, args.output_path)


if __name__ == "__main__":
    main()
