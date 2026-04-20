"""
Generate summaries for topic Weibo records with the DeepSeek API.

Input:
    D:/GraduationProject/data/cleaned/topic_weibo.parquet

Output:
    D:/GraduationProject/data/cleaned/topic_weibo_summary.parquet

Set DEEPSEEK_API_KEY in the project .env file before running:
    DEEPSEEK_API_KEY=sk-...
    python topic_weibo_summary/summarize_topic_weibo.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "topic_weibo.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "topic_weibo_summary.parquet"

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_BASE = "https://api.deepseek.com"
SUMMARY_THRESHOLD = 10

load_dotenv(ENV_PATH)

SYSTEM_PROMPT = """
你是一名微博话题文本摘要助手。

你的任务是：根据输入的话题微博文本，生成一段简洁、客观、信息完整的摘要。

请严格遵循以下要求：
1. 保留事件主体、关键事实、主要行为、争议焦点和结果信息。
2. 摘要必须体现该微博涉及的主要争议点或分歧（如：是否存在违规、是否属实、是否合理等）。
3. 如果原文表达了明确立场、批评、支持、质疑、讽刺、愤怒等态度，可以适度保留其总体倾向。
4. 不要照搬原文中的冗余修饰、感叹、重复表达、营销语或平台提示语。
5. 不得编造原文没有的信息，不得加入常识性扩写或主观评价。
6. 输出应简洁、通顺、便于后续模型快速理解该微博在讨论什么、争议点是什么。
7. 摘要长度控制在 50～120 字之间。
8. 只输出摘要正文，不要输出任何解释、标题、前缀或额外说明。
""".strip()

USER_PROMPT_TEMPLATE = """
请对下面的话题微博文本生成摘要。

要求：
- 用于后续评论的立场与情绪分析
- 关注“这条微博在说什么、争议点是什么、作者大致持什么态度”
- 只输出一段摘要正文

微博文本：
{content}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize topic Weibo texts with DeepSeek.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input parquet path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output parquet path.")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"), help="DeepSeek API key.")
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--threshold", type=int, default=SUMMARY_THRESHOLD)
    parser.add_argument("--save-every", type=int, default=1, help="Save after this many API successes/failures.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of pending rows to process, useful for a smoke run.",
    )
    return parser.parse_args()


def load_or_initialize(input_path: Path, output_path: Path, threshold: int) -> pd.DataFrame:
    if output_path.exists():
        df = pd.read_parquet(output_path)
        print(f"Loaded existing output for resume: {output_path}")
    else:
        df = pd.read_parquet(input_path)
        print(f"Loaded source data: {input_path}")

    required_columns = {"content", "text_length"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if "need_summary" not in df.columns:
        df["need_summary"] = df["text_length"].fillna(0) >= threshold
    else:
        df["need_summary"] = df["text_length"].fillna(0) >= threshold

    if "summary_text" not in df.columns:
        df["summary_text"] = None

    if "summary_status" not in df.columns:
        df["summary_status"] = "pending"

    if "analysis_context" not in df.columns:
        df["analysis_context"] = None

    short_mask = ~df["need_summary"]
    df.loc[short_mask, "summary_text"] = None
    df.loc[short_mask, "summary_status"] = "skipped"
    df.loc[short_mask, "analysis_context"] = df.loc[short_mask, "content"].fillna("")

    long_empty_mask = df["need_summary"] & df["summary_text"].isna()
    df.loc[long_empty_mask, "summary_text"] = None

    valid_statuses = ["success", "failed", "pending", "skipped"]
    long_missing_status_mask = df["need_summary"] & ~df["summary_status"].isin(valid_statuses)
    df.loc[long_missing_status_mask, "summary_status"] = "pending"

    long_success_mask = df["need_summary"] & (df["summary_status"] == "success")
    df.loc[long_success_mask, "analysis_context"] = df.loc[long_success_mask, "summary_text"]

    long_not_success_mask = df["need_summary"] & (df["summary_status"] != "success")
    df.loc[long_not_success_mask, "analysis_context"] = None

    return df


def save_progress(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def build_request_body(model: str, content: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(content=content)},
        ],
        "temperature": 0,
        "stream": False,
    }


def call_deepseek(
    *,
    api_key: str,
    api_base: str,
    model: str,
    content: str,
    timeout: int,
) -> str:
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    body = json.dumps(build_request_body(model, content), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    try:
        summary = payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected DeepSeek response: {payload}") from exc

    if not summary:
        raise RuntimeError("DeepSeek returned an empty summary.")
    return summary


def summarize_with_retries(
    *,
    api_key: str,
    api_base: str,
    model: str,
    content: str,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return call_deepseek(
                api_key=api_key,
                api_base=api_base,
                model=model,
                content=content,
                timeout=timeout,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def pending_indices(df: pd.DataFrame) -> list[int]:
    mask = df["need_summary"] & (df["summary_status"] != "success")
    return df.index[mask].tolist()


def main() -> None:
    args = parse_args()
    df = load_or_initialize(args.input, args.output, args.threshold)

    total = len(df)
    need_count = int(df["need_summary"].sum())
    pending = pending_indices(df)
    if args.limit is not None:
        pending = pending[: args.limit]

    save_progress(df, args.output)
    print(f"Rows: {total}; need summary: {need_count}; pending this run: {len(pending)}")

    if not pending:
        print(f"Nothing to process. Output is ready: {args.output}")
        return

    if not args.api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for pending long-text summaries.")

    processed_since_save = 0
    for position, index in enumerate(pending, start=1):
        content = str(df.at[index, "content"] or "").strip()
        row_id = df.at[index, "weibo_id"] if "weibo_id" in df.columns else index
        print(f"[{position}/{len(pending)}] Summarizing row index={index}, weibo_id={row_id}")

        try:
            summary = summarize_with_retries(
                api_key=args.api_key,
                api_base=args.api_base,
                model=args.model,
                content=content,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            df.at[index, "summary_text"] = summary
            df.at[index, "summary_status"] = "success"
            df.at[index, "analysis_context"] = summary
        except Exception as exc:
            df.at[index, "summary_text"] = None
            df.at[index, "summary_status"] = "failed"
            df.at[index, "analysis_context"] = None
            print(f"  Failed: {exc}")

        processed_since_save += 1
        if processed_since_save >= args.save_every:
            save_progress(df, args.output)
            processed_since_save = 0
            print(f"  Saved progress to {args.output}")

    save_progress(df, args.output)
    success_count = int((df["summary_status"] == "success").sum())
    skipped_count = int((df["summary_status"] == "skipped").sum())
    failed_count = int((df["summary_status"] == "failed").sum())
    print(f"Done. success={success_count}, skipped={skipped_count}, failed={failed_count}, output={args.output}")


if __name__ == "__main__":
    main()
