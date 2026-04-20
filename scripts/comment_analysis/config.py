"""Configuration, CLI parsing, and logging setup for comment analysis."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from comment_analysis.schema import MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_COMMENT_PATH = PROJECT_ROOT / "data" / "high_quality" / "topic_comment.parquet"
DEFAULT_WEIBO_PATH = PROJECT_ROOT / "data" / "high_quality" / "topic_weibo.parquet"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "data" / "analysis" / "comment_analysis_result.jsonl"
DEFAULT_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "analysis" / "comment_analysis_result.parquet"
DEFAULT_FAILED_JSONL = PROJECT_ROOT / "data" / "analysis" / "comment_analysis_failed.jsonl"
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "comment_analysis.log"


@dataclass(frozen=True)
class AnalysisConfig:
    api_key: str
    api_base: str
    model: str
    comment_path: Path
    weibo_path: Path
    output_jsonl: Path
    output_parquet: Path
    failed_jsonl: Path
    log_file: Path
    run_id: str
    concurrency: int
    request_timeout: float
    max_retries: int
    retry_base_delay: float
    retry_max_delay: float
    checkpoint_every: int
    limit: int | None
    random_sample: bool
    random_seed: int | None
    response_format_json: bool


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,
    )


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    load_dotenv(ENV_PATH)

    parser = argparse.ArgumentParser(
        description="Asynchronously analyze first-level comments with DeepSeek."
    )
    parser.add_argument("--comment-path", type=Path, default=DEFAULT_COMMENT_PATH)
    parser.add_argument("--weibo-path", type=Path, default=DEFAULT_WEIBO_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--failed-jsonl", type=Path, default=DEFAULT_FAILED_JSONL)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(os.getenv("COMMENT_ANALYSIS_LOG_FILE", DEFAULT_LOG_FILE)),
        help="Log file path. Defaults to project-root .log/comment_analysis.log.",
    )
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", MODEL_NAME))
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("COMMENT_ANALYSIS_CONCURRENCY", "5")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("COMMENT_ANALYSIS_TIMEOUT", "60")),
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("COMMENT_ANALYSIS_MAX_RETRIES", "3")),
        help="Retry count after the first attempt.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=float(os.getenv("COMMENT_ANALYSIS_RETRY_BASE_DELAY", "1.5")),
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=float(os.getenv("COMMENT_ANALYSIS_RETRY_MAX_DELAY", "30")),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.getenv("COMMENT_ANALYSIS_CHECKPOINT_EVERY", "50")),
        help="Save parquet after this many successes. Use 0 to disable checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of pending comments to process in this run.",
    )
    parser.add_argument(
        "--random-sample",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("COMMENT_ANALYSIS_RANDOM_SAMPLE", "1").lower()
        not in {"0", "false", "no", "off"},
        help="Randomly sample pending comments when --limit is set. Enabled by default.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=(
            int(os.environ["COMMENT_ANALYSIS_RANDOM_SEED"])
            if os.getenv("COMMENT_ANALYSIS_RANDOM_SEED")
            else None
        ),
        help="Optional random seed for reproducible --random-sample runs.",
    )
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Do not send OpenAI-compatible JSON response_format to DeepSeek.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")

    return AnalysisConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        comment_path=args.comment_path,
        weibo_path=args.weibo_path,
        output_jsonl=args.output_jsonl,
        output_parquet=args.output_parquet,
        failed_jsonl=args.failed_jsonl,
        log_file=args.log_file,
        run_id=make_run_id(),
        concurrency=args.concurrency,
        request_timeout=args.timeout,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
        checkpoint_every=args.checkpoint_every,
        limit=args.limit,
        random_sample=args.random_sample,
        random_seed=args.random_seed,
        response_format_json=not args.no_response_format,
    )
