"""Add parent comment fields to existing comment analysis output files.

Run from the project root:
    python scripts/comment_analysis/migrate_comment_analysis_outputs.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from comment_analysis.config import (  # noqa: E402
    DEFAULT_COMMENT_PATH,
    DEFAULT_FAILED_JSONL,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_OUTPUT_PARQUET,
)
from comment_analysis.data_io import (  # noqa: E402
    PARQUET_OUTPUT_FIELDS,
    as_clean_str,
    is_first_level_parent,
    read_jsonl_records,
    write_jsonl_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate comment analysis outputs to include parent_id and parent_comment_text."
    )
    parser.add_argument("--comment-path", type=Path, default=DEFAULT_COMMENT_PATH)
    parser.add_argument("--jsonl-path", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--failed-jsonl-path", type=Path, default=DEFAULT_FAILED_JSONL)
    parser.add_argument(
        "--skip-failed-jsonl",
        action="store_true",
        help="Do not migrate the failed JSONL file.",
    )
    return parser.parse_args()


def load_comment_parent_maps(comment_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    df_comment = pd.read_parquet(comment_path, columns=["comment_id", "parent_id", "content"])
    df_comment = df_comment.copy()
    df_comment["comment_id"] = df_comment["comment_id"].map(as_clean_str)
    df_comment["parent_id"] = df_comment["parent_id"].map(as_clean_str)
    df_comment["content"] = df_comment["content"].map(as_clean_str)
    df_comment = df_comment[df_comment["comment_id"].ne("")]
    df_comment = df_comment.drop_duplicates(subset=["comment_id"], keep="first")

    parent_id_by_comment_id = dict(zip(df_comment["comment_id"], df_comment["parent_id"]))
    content_by_comment_id = dict(zip(df_comment["comment_id"], df_comment["content"]))
    return parent_id_by_comment_id, content_by_comment_id


def resolve_parent_fields(
    record: dict[str, Any],
    parent_id_by_comment_id: dict[str, str],
    content_by_comment_id: dict[str, str],
) -> tuple[str, str]:
    comment_id = as_clean_str(record.get("comment_id"))
    parent_id = as_clean_str(record.get("parent_id"))
    if not parent_id:
        parent_id = parent_id_by_comment_id.get(comment_id, "-1")
    if not parent_id:
        parent_id = "-1"

    if is_first_level_parent(parent_id):
        return "-1", ""

    parent_comment_text = as_clean_str(record.get("parent_comment_text"))
    if not parent_comment_text:
        parent_comment_text = content_by_comment_id.get(parent_id, "")
    return parent_id, parent_comment_text


def ordered_jsonl_record(record: dict[str, Any], parent_id: str, parent_comment_text: str) -> dict[str, Any]:
    migrated: dict[str, Any] = {}
    for field in ("comment_id", "weibo_id"):
        if field in record:
            migrated[field] = record[field]
    migrated["parent_id"] = parent_id
    if "content" in record:
        migrated["content"] = record["content"]
    migrated["parent_comment_text"] = parent_comment_text

    for key, value in record.items():
        if key not in migrated and key not in {"parent_id", "parent_comment_text"}:
            migrated[key] = value
    return migrated


def migrate_jsonl(
    path: Path,
    parent_id_by_comment_id: dict[str, str],
    content_by_comment_id: dict[str, str],
) -> int:
    records = read_jsonl_records(path)
    if not records:
        logging.info("No JSONL records to migrate: %s", path)
        return 0

    migrated_records = []
    for record in records:
        parent_id, parent_comment_text = resolve_parent_fields(
            record,
            parent_id_by_comment_id,
            content_by_comment_id,
        )
        migrated_records.append(ordered_jsonl_record(record, parent_id, parent_comment_text))

    write_jsonl_records(path, migrated_records)
    logging.info("Migrated JSONL records: %s -> %s", len(migrated_records), path)
    return len(migrated_records)


def migrate_parquet(
    path: Path,
    parent_id_by_comment_id: dict[str, str],
    content_by_comment_id: dict[str, str],
) -> int:
    if not path.exists():
        logging.info("No parquet file to migrate: %s", path)
        return 0

    df = pd.read_parquet(path)
    if df.empty:
        df.to_parquet(path, index=False)
        logging.info("Parquet file is empty: %s", path)
        return 0

    if "parent_id" not in df.columns:
        df["parent_id"] = ""
    if "parent_comment_text" not in df.columns:
        df["parent_comment_text"] = ""

    parent_ids: list[str] = []
    parent_texts: list[str] = []
    for record in df.to_dict(orient="records"):
        parent_id, parent_comment_text = resolve_parent_fields(
            record,
            parent_id_by_comment_id,
            content_by_comment_id,
        )
        parent_ids.append(parent_id)
        parent_texts.append(parent_comment_text)

    df["parent_id"] = parent_ids
    df["parent_comment_text"] = parent_texts

    ordered_columns = [column for column in PARQUET_OUTPUT_FIELDS if column in df.columns]
    extra_columns = [column for column in df.columns if column not in ordered_columns]
    df = df[ordered_columns + extra_columns]
    df.to_parquet(path, index=False)
    logging.info("Migrated parquet rows: %s -> %s", len(df), path)
    return len(df)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    parent_id_by_comment_id, content_by_comment_id = load_comment_parent_maps(args.comment_path)

    migrate_jsonl(args.jsonl_path, parent_id_by_comment_id, content_by_comment_id)
    migrate_parquet(args.parquet_path, parent_id_by_comment_id, content_by_comment_id)
    if not args.skip_failed_jsonl:
        migrate_jsonl(args.failed_jsonl_path, parent_id_by_comment_id, content_by_comment_id)


if __name__ == "__main__":
    main()
