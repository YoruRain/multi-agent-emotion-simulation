from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORE_INPUT_PATH = PROJECT_ROOT / "data" / "high_quality" / "user_weibo.parquet"
DEFAULT_ALL_INPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "user_weibo.parquet"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "user_weibo_topic_signals.parquet"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "user_weibo_topic_signals.log"

BASE_COLUMNS = [
    "weibo_id",
    "user_id",
    "content",
    "is_repost",
    "reposted_weibo_id",
    "source_content",
]
OUTPUT_COLUMNS = [
    "weibo_id",
    "user_id",
    "content",
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
CORE_REQUIRED_COLUMNS = {
    "weibo_id",
    "user_id",
    "content",
    "topics",
    "is_repost",
    "reposted_weibo_id",
    "text_quality",
}
ALL_REQUIRED_COLUMNS = {"weibo_id", "content", "topics"}

# The dictionary stays intentionally compact and readable so downstream checks
# can still trace why a category was assigned.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "社会公共事件": [
        "警方", "通报", "事故", "地震", "火灾", "爆炸",
        "车祸", "救援", "调查", "回应", "热搜", "维权",
        "判决", "法院", "检察院", "公安", "消防", "律师", 
        "法官", "公众"
    ],
    "政策民生": [
        "政策", "医保", "社保", "教育", "就业", "工资",
        "房价", "租房", "学校", "医院", "高考", "考研",
        "公务员", "养老金", "户口", "补贴", "应届"
    ],
    "时事政治": [
        "政治", "外交", "国际", "中美","美国", "台湾", 
        "日本", "俄罗斯", "乌克兰", "以色列", "巴勒斯坦", 
        "选举", "总统", "议员", "制裁", "关税","战争", 
        "边境", "领土", "主权", "民族主义", "阅兵", "祖国", 
        "战区", "朝鲜", "特朗普", "伊朗"
    ], 
    "娱乐文化": [
        "明星", "演员", "歌手", "演唱会", "电影", "电视剧", 
        "综艺","粉丝", "塌房", "票房", "官宣", "剧组", 
        "偶像", "剧情", "首播", "古装剧", "晚会", "广播剧",
        "巡演", "上映", "关注超话", "代言人", "新剧", "短剧",
        "爱豆", "影帝", "球迷", "假唱", "内娱", "台词", "时装周", 
        "春晚", "我担", "恋综", "编剧", "热播"
    ],
    "日常生活": [
        "上班", "下班", "睡觉", "吃饭", "天气", "早安",
        "晚安", "回家", "出门", "旅游", "学习", "考试",
        "宿舍", "加班", "周末", "同事", "同学", "好吃", 
        "生活手记", "做饭"
    ],
    "游戏动漫": [
        "游戏", "抽卡", "皮肤", "角色", "原神", "王者荣耀",
        "崩坏", "明日方舟", "二次元", "动漫", "漫画", "番剧",
        "cos", "剑网", "恋与", "电竞", "英雄联盟", "金铲铲", 
        "第五人格", "鸣潮", "和平精英"
    ],
    "情感表达": [
        "无语", "崩溃", "破防", "开心", "难过", "生气", "烦死",
        "震惊", "心疼", "恶心", "离谱", "好笑", "哭了", "累了", "焦虑",
    ],
    "广告营销": [
        "点开红包", "现金红包", "微博红包", "随机抽奖",
        "抽奖详情", "领取优惠券", "购买请戳", "限时特卖",
        "分享有礼", "试试你的手气", "试手气", "抽奖平台", 
        "转发评论", "转发+评论", "转发关注", "转发+关注", 
        "关注转发", "关注+转发", "转关", "转+关", "好礼", 
        "我在参与", "免费围观", "森林驿站", "开放公测", 
        "上闲鱼", "微博抓马", "春节AI合拍", "微博渔场", 
        "解锁赛博年味", "旅行青蛙中国", "微博之夜", "粉丝福利", 
        "好运在此", "运气好到爆", "嗨抢", "欧气爆棚", 
        "抓马福", "马年接福", "集福袋", "惊喜福利", 
        "独家首播", "超话大赏", "爆款剧王", "星品", 
        "微博云包场", "复制口令", "红包活动"
    ],
    "媒体官方": [
        "记者", "报道", "据悉", "来源", "官方", "声明",
        "发布", "人民日报", "新华社", "央视", "澎湃",
        "观察者网", "环球时报",
    ],
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


def load_parquet(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported, got: {path}")

    try:
        df = pd.read_parquet(path, columns=sorted(required_columns))
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow or fastparquet in the active Python environment."
        ) from exc

    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Input dataframe {path} is missing required columns: {missing}")
    return df


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_topic_string(value: Any) -> str | None:
    if pd.isna(value):
        return None

    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return None

    deduped = list(dict.fromkeys(parts))
    return ",".join(deduped)


def join_unique_strings(values: list[str]) -> str | None:
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        return None
    return ",".join(dict.fromkeys(cleaned))


def build_source_lookup(df_all: pd.DataFrame) -> pd.DataFrame:
    lookup = df_all.loc[:, ["weibo_id", "content", "topics"]].copy()
    lookup = lookup.drop_duplicates(subset=["weibo_id"], keep="first")
    lookup = lookup.rename(
        columns={
            "content": "source_content",
            "topics": "source_topics_raw",
        }
    )
    return lookup


def determine_has_repost_comment(row: pd.Series) -> bool:
    if not bool(row["is_repost"]):
        return False

    text_quality = pd.to_numeric(row["text_quality"], errors="coerce")
    if pd.isna(text_quality) or float(text_quality) < 3:
        return False

    content = normalize_text(row["content"])
    if content.startswith("//"):
        return False

    return True


def build_match_text(content: str, source_content: str, is_repost: bool, has_repost_comment: bool) -> str:
    if not is_repost:
        return content

    if has_repost_comment:
        if source_content:
            return "\n".join([content, source_content]).strip()
        return content

    return source_content


def extract_keyword_hits(text: str) -> list[str]:
    if not text:
        return []

    matched_positions: list[tuple[int, str]] = []
    for keywords in TOPIC_KEYWORDS.values():
        for keyword in keywords:
            position = text.find(keyword)
            if position >= 0:
                matched_positions.append((position, keyword))

    matched_positions.sort(key=lambda item: (item[0], item[1]))
    hits: list[str] = []
    for _, keyword in matched_positions:
        if keyword not in hits:
            hits.append(keyword)
    return hits


def infer_topic_categories(
    user_content: str,
    source_content: str,
    user_topics: str,
    source_topics: str,
) -> list[str]:
    categories: list[str] = []
    combined = "\n".join(text for text in [user_content, source_content, user_topics, source_topics] if text).strip()
    if not combined:
        return categories

    for category, keywords in TOPIC_KEYWORDS.items():
        if category == "日常生活":
            # Restrict daily-life keyword matching to the user's own weibo text,
            # while still allowing topic-tag strings to contribute as before.
            category_text = "\n".join(
                text for text in [user_content, user_topics, source_topics] if text
            ).strip()
        else:
            category_text = combined

        if any(keyword in category_text for keyword in keywords):
            categories.append(category)
    return categories


def compute_signal_confidence(row: pd.Series) -> float:
    score = 0.0
    if row["user_topics"] is not None:
        score += 0.35
    if row["source_topics"] is not None:
        score += 0.20
    if row["explicit_keywords"] is not None:
        score += 0.30
    if row["explicit_topic_categories"] is not None:
        score += 0.15
    if bool(row["is_repost"]) and bool(row["has_repost_comment"]):
        score += 0.10
    if bool(row["is_repost"]) and not bool(row["has_repost_comment"]):
        score -= 0.15

    text_quality = pd.to_numeric(row["text_quality"], errors="coerce")
    if not normalize_text(row["content"]) or pd.isna(text_quality) or float(text_quality) < 3:
        score -= 0.10

    return round(min(max(score, 0.0), 1.0), 4)


def build_topic_signal_table(df_core: pd.DataFrame, df_all: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    core = df_core.copy()
    core["content"] = core["content"].map(normalize_text)
    core["user_topics"] = core["topics"].map(normalize_topic_string)
    core["has_repost_comment"] = core.apply(determine_has_repost_comment, axis=1)

    source_lookup = build_source_lookup(df_all)
    merged = core.merge(
        source_lookup,
        how="left",
        left_on="reposted_weibo_id",
        right_on="weibo_id",
        suffixes=("", "_source"),
    )
    merged = merged.rename(columns={"weibo_id_source": "matched_source_weibo_id"})
    merged["source_content"] = merged["source_content"].map(normalize_text)
    merged["source_topics"] = merged["source_topics_raw"].map(normalize_topic_string)
    merged["source_content"] = merged["source_content"]
    merged.loc[~merged["is_repost"], "source_content"] = None
    merged.loc[~merged["is_repost"], "source_topics"] = None
    merged.loc[~merged["is_repost"], "source_content"] = ""

    keyword_strings: list[str | None] = []
    category_strings: list[str | None] = []
    for row in merged.itertuples(index=False):
        match_text = build_match_text(
            content=row.content,
            source_content=row.source_content,
            is_repost=bool(row.is_repost),
            has_repost_comment=bool(row.has_repost_comment),
        )

        keyword_hits = extract_keyword_hits(match_text)
        keyword_strings.append(join_unique_strings(keyword_hits))

        categories = infer_topic_categories(
            user_content=row.content,
            source_content=row.source_content,
            user_topics=row.user_topics or "",
            source_topics=row.source_topics or "",
        )
        category_strings.append(join_unique_strings(categories))

    merged["explicit_keywords"] = keyword_strings
    merged["explicit_topic_categories"] = category_strings
    merged["signal_confidence"] = merged.apply(compute_signal_confidence, axis=1)

    output_df = merged.loc[:, BASE_COLUMNS].copy()
    output_df["has_repost_comment"] = merged["has_repost_comment"].astype(bool)
    output_df["user_topics"] = merged["user_topics"]
    output_df["source_topics"] = merged["source_topics"]
    output_df["explicit_keywords"] = merged["explicit_keywords"]
    output_df["explicit_topic_categories"] = merged["explicit_topic_categories"]
    output_df["signal_confidence"] = merged["signal_confidence"].astype(float)
    output_df = output_df.loc[:, OUTPUT_COLUMNS]

    matched_source_mask = merged["is_repost"] & merged["matched_source_weibo_id"].notna()
    unmatched_source_mask = merged["is_repost"] & merged["matched_source_weibo_id"].isna()
    stats = {
        "core_rows": int(len(df_core)),
        "all_rows": int(len(df_all)),
        "output_rows": int(len(output_df)),
        "original_count": int((~output_df["is_repost"]).sum()),
        "repost_count": int(output_df["is_repost"].sum()),
        "valid_repost_comment_count": int(output_df["has_repost_comment"].sum()),
        "invalid_repost_comment_count": int((output_df["is_repost"] & ~output_df["has_repost_comment"]).sum()),
        "matched_source_count": int(matched_source_mask.sum()),
        "unmatched_source_count": int(unmatched_source_mask.sum()),
        "user_topics_ratio": float(output_df["user_topics"].notna().mean()),
        "source_topics_ratio": float(output_df["source_topics"].notna().mean()),
        "explicit_keywords_ratio": float(output_df["explicit_keywords"].notna().mean()),
        "explicit_topic_categories_ratio": float(output_df["explicit_topic_categories"].notna().mean()),
        "signal_confidence_describe": output_df["signal_confidence"].describe(),
    }
    return output_df, stats


def validate_output(df_output: pd.DataFrame, expected_rows: int) -> None:
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in df_output.columns]
    if missing_columns:
        raise AssertionError(f"Output dataframe is missing required columns: {missing_columns}")
    if len(df_output) != expected_rows:
        raise AssertionError(f"Output row count {len(df_output)} does not match expected {expected_rows}")
    if not df_output["signal_confidence"].between(0.0, 1.0, inclusive="both").all():
        raise AssertionError("signal_confidence contains values outside [0, 1]")


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet output is supported, got: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Saved %d rows to %s", len(df), output_path)


def log_summary(stats: dict[str, Any]) -> None:
    LOGGER.info("核心用户微博表行数: %d", stats["core_rows"])
    LOGGER.info("用户微博总表行数: %d", stats["all_rows"])
    LOGGER.info("输出表行数: %d", stats["output_rows"])
    LOGGER.info("原创微博数量: %d", stats["original_count"])
    LOGGER.info("转发微博数量: %d", stats["repost_count"])
    LOGGER.info("有效转发评论数量: %d", stats["valid_repost_comment_count"])
    LOGGER.info("无有效转发评论数量: %d", stats["invalid_repost_comment_count"])
    LOGGER.info("转发源微博匹配成功数量: %d", stats["matched_source_count"])
    LOGGER.info("转发源微博匹配失败数量: %d", stats["unmatched_source_count"])
    LOGGER.info("user_topics 非空比例: %.4f", stats["user_topics_ratio"])
    LOGGER.info("source_topics 非空比例: %.4f", stats["source_topics_ratio"])
    LOGGER.info("explicit_keywords 非空比例: %.4f", stats["explicit_keywords_ratio"])
    LOGGER.info("explicit_topic_categories 非空比例: %.4f", stats["explicit_topic_categories_ratio"])
    LOGGER.info("signal_confidence describe:\n%s", stats["signal_confidence_describe"].to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weibo-level explicit topic signals for core user weibos."
    )
    parser.add_argument(
        "--core_input_path",
        type=Path,
        default=DEFAULT_CORE_INPUT_PATH,
        help="Core user weibo parquet path.",
    )
    parser.add_argument(
        "--all_input_path",
        type=Path,
        default=DEFAULT_ALL_INPUT_PATH,
        help="All user weibo parquet path for repost source lookup.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path.",
    )
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for log files.")
    parser.add_argument("--log_file", type=Path, default=None, help="Optional explicit log file path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, log_dir=args.log_dir, log_file=args.log_file)
    LOGGER.info("Using core_input_path=%s", args.core_input_path)
    LOGGER.info("Using all_input_path=%s", args.all_input_path)
    LOGGER.info("Using output_path=%s", args.output_path)

    df_core = load_parquet(args.core_input_path, CORE_REQUIRED_COLUMNS)
    df_all = load_parquet(args.all_input_path, ALL_REQUIRED_COLUMNS)
    output_df, stats = build_topic_signal_table(df_core, df_all)
    validate_output(output_df, expected_rows=len(df_core))
    log_summary(stats)
    save_output(output_df, args.output_path)


if __name__ == "__main__":
    main()
