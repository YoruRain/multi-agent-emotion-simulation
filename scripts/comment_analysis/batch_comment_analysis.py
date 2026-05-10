"""
Asynchronously analyze Weibo comments with the DeepSeek API.

Default inputs:
    data/high_quality/topic_comment.parquet
    data/high_quality/topic_weibo.parquet

Default outputs:
    data/profile/comments/comment_analysis_result.jsonl
    data/profile/comments/comment_analysis_result.parquet
    data/profile/comments/comment_analysis_failed.jsonl

Run from the project root:
    python scripts/comment_analysis/batch_comment_analysis.py --limit 20
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import httpx


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from comment_analysis.config import AnalysisConfig, build_config, parse_args, setup_logging
from comment_analysis.data_io import (
    CommentSample,
    ResultWriter,
    archive_existing_failed_log,
    compact_jsonl,
    load_completed_ids,
    load_samples,
    make_failure_record,
    save_parquet_from_jsonl,
)
from comment_analysis.deepseek_client import analyze_with_retries


async def process_sample(
    client: httpx.AsyncClient,
    config: AnalysisConfig,
    writer: ResultWriter,
    semaphore: asyncio.Semaphore,
    sample: CommentSample,
) -> bool:
    async with semaphore:
        try:
            record = await analyze_with_retries(client, config, sample)
            await writer.write_success(record)
            return True
        except Exception as exc:
            logging.error(
                "Failed comment_id=%s after %s retries: %s",
                sample.comment_id,
                config.max_retries,
                exc,
            )
            await writer.write_failure(
                make_failure_record(
                    sample,
                    exc,
                    retry_count=config.max_retries,
                    run_id=config.run_id,
                )
            )
            return False


async def run_batch(config: AnalysisConfig, samples: list[CommentSample]) -> ResultWriter:
    timeout = httpx.Timeout(config.request_timeout)
    limits = httpx.Limits(
        max_connections=config.concurrency,
        max_keepalive_connections=config.concurrency,
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    writer = ResultWriter(config)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            asyncio.create_task(process_sample(client, config, writer, semaphore, sample))
            for sample in samples
        ]

        done_count = 0
        for task in asyncio.as_completed(tasks):
            await task
            done_count += 1
            if done_count == 1 or done_count % 10 == 0 or done_count == len(samples):
                logging.info(
                    "Progress: %s/%s finished, success=%s, failed=%s",
                    done_count,
                    len(samples),
                    writer.success_count,
                    writer.failure_count,
                )

    return writer


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    config = build_config(args)
    logging.info("Logging to file: %s", config.log_file)
    logging.info("Run id: %s", config.run_id)
    logging.info(
        "Sampling config: comment_level=%s, limit=%s, random_sample=%s, random_seed=%s",
        config.comment_level,
        config.limit,
        config.random_sample,
        config.random_seed,
    )

    if not config.api_key:
        raise ValueError("DEEPSEEK_API_KEY is required. Set it in .env or pass --api-key.")

    archived_failed_log = archive_existing_failed_log(config.failed_jsonl, config.run_id)
    if archived_failed_log:
        logging.info("Archived previous failure log to: %s", archived_failed_log)

    compact_jsonl(config.output_jsonl)
    completed_ids = load_completed_ids(config.output_jsonl, config.output_parquet)
    samples, selected_comment_count, skipped_count = load_samples(config, completed_ids)

    if not samples:
        row_count = save_parquet_from_jsonl(config.output_jsonl, config.output_parquet)
        logging.info("Nothing pending. Parquet refreshed with %s rows.", row_count)
        return

    writer = asyncio.run(run_batch(config, samples))
    final_rows = save_parquet_from_jsonl(config.output_jsonl, config.output_parquet)

    logging.info("Final summary")
    logging.info("Selected comment count: %s", selected_comment_count)
    logging.info("Skipped completed: %s", skipped_count)
    logging.info("New successes: %s", writer.success_count)
    logging.info("New failures: %s", writer.failure_count)
    logging.info("Final parquet rows: %s", final_rows)
    logging.info("JSONL output: %s", config.output_jsonl)
    logging.info("Parquet output: %s", config.output_parquet)
    logging.info("Failure log: %s", config.failed_jsonl)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error while running comment analysis")
        raise
