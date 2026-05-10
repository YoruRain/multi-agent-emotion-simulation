"""Data loading, normalization, and result persistence helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from comment_analysis.config import AnalysisConfig
from comment_analysis.schema import CommentAnalysis


ANALYSIS_FIELDS = [
    "emotion_target_type",
    "emotion_target_text",
    "stance_target_type",
    "stance_target_text",
    "cause_or_stimulus",
    "target_explicit",
    "focus_type",
    "responsibility",
    "control",
    "norm_violation",
    "emotion_label",
    "emotion_intensity",
    "stance_label",
    "stance_intensity",
    "argument_type",
    "confidence",
    "emotion_evidence",
    "stance_evidence",
    "needs_more_context",
    "low_confidence_reason",
    "semantic_validation_warnings",
]

BASE_OUTPUT_FIELDS = [
    "comment_id",
    "weibo_id",
    "parent_id",
    "content",
    "parent_comment_text",
    "analysis_context",
    "topic",
]

TRACKING_OUTPUT_FIELDS = [
    "model_name",
    "analyzed_at",
    "run_id",
]

PARQUET_OUTPUT_FIELDS = BASE_OUTPUT_FIELDS + ANALYSIS_FIELDS + TRACKING_OUTPUT_FIELDS


@dataclass(frozen=True)
class CommentSample:
    comment_id: str
    weibo_id: str
    parent_id: str
    content: str
    parent_comment_text: str
    topic: str
    analysis_context: str
    row: dict[str, Any]


def as_clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def require_columns(df: pd.DataFrame, columns: set[str], source_name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")


def is_first_level_parent(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return as_clean_str(value) == "-1"


def is_high_quality_text(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return as_clean_str(value) == "3"


def comment_level_label(comment_level: str) -> str:
    return "first-level" if comment_level == "first" else "second-level"


def load_samples(config: AnalysisConfig, completed_ids: set[str]) -> tuple[list[CommentSample], int, int]:
    logging.info("Loading comments: %s", config.comment_path)
    logging.info("Loading topic weibos: %s", config.weibo_path)
    df_comment = pd.read_parquet(config.comment_path)
    df_weibo = pd.read_parquet(config.weibo_path)

    require_columns(
        df_comment,
        {"comment_id", "weibo_id", "parent_id", "text_quality", "content"},
        "topic_comment",
    )
    require_columns(
        df_weibo,
        {"weibo_id", "topic", "analysis_context"},
        "topic_weibo",
    )

    original_count = len(df_comment)
    df_comment["comment_id"] = df_comment["comment_id"].map(as_clean_str)
    df_comment["weibo_id"] = df_comment["weibo_id"].map(as_clean_str)
    df_comment["parent_id"] = df_comment["parent_id"].map(as_clean_str)
    df_comment["content"] = df_comment["content"].map(as_clean_str)

    parent_lookup = (
        df_comment[["comment_id", "content"]]
        .drop_duplicates(subset=["comment_id"], keep="first")
        .rename(columns={"comment_id": "parent_id", "content": "parent_comment_text"})
    )

    if config.comment_level == "first":
        level_mask = df_comment["parent_id"].apply(is_first_level_parent)
    else:
        level_mask = ~df_comment["parent_id"].apply(is_first_level_parent)

    df_comment = df_comment[level_mask].copy()
    selected_comment_count = len(df_comment)
    df_comment = df_comment[df_comment["text_quality"].apply(is_high_quality_text)].copy()
    high_quality_selected_count = len(df_comment)
    df_comment = df_comment[df_comment["comment_id"].ne("") & df_comment["content"].ne("")]
    df_comment = df_comment.drop_duplicates(subset=["comment_id"], keep="first")

    if config.comment_level == "first":
        df_comment["parent_id"] = df_comment["parent_id"].where(
            df_comment["parent_id"].ne(""),
            "-1",
        )
        df_comment["parent_comment_text"] = ""
    else:
        df_comment = df_comment.merge(parent_lookup, on="parent_id", how="left")
        df_comment["parent_comment_text"] = df_comment["parent_comment_text"].map(as_clean_str)
        missing_parent_count = int(df_comment["parent_comment_text"].eq("").sum())
        if missing_parent_count:
            logging.warning(
                "Second-level comments without matched parent comment text: %s",
                missing_parent_count,
            )

    df_weibo = df_weibo.copy()
    df_weibo["weibo_id"] = df_weibo["weibo_id"].map(as_clean_str)
    weibo_context = df_weibo[["weibo_id", "topic", "analysis_context"]].drop_duplicates(
        subset=["weibo_id"], keep="first"
    )
    merged = df_comment.merge(weibo_context, on="weibo_id", how="left", suffixes=("", "_weibo"))
    merged["topic"] = merged["topic"].map(as_clean_str)
    merged["analysis_context"] = merged["analysis_context"].map(as_clean_str)

    skipped_count = int(merged["comment_id"].isin(completed_ids).sum())
    pending = merged[~merged["comment_id"].isin(completed_ids)].copy()
    if config.limit is not None:
        sample_size = min(config.limit, len(pending))
        if config.random_sample and sample_size > 0:
            pending = pending.sample(n=sample_size, random_state=config.random_seed)
            logging.info(
                "Randomly sampled %s pending comments for this run; random_seed=%s",
                sample_size,
                config.random_seed,
            )
        else:
            pending = pending.head(sample_size)
            logging.info("Selected first %s pending comments for this run.", sample_size)

    samples: list[CommentSample] = []
    for record in pending.to_dict(orient="records"):
        samples.append(
            CommentSample(
                comment_id=as_clean_str(record.get("comment_id")),
                weibo_id=as_clean_str(record.get("weibo_id")),
                parent_id=as_clean_str(record.get("parent_id")),
                content=as_clean_str(record.get("content")),
                parent_comment_text=as_clean_str(record.get("parent_comment_text")),
                topic=as_clean_str(record.get("topic")),
                analysis_context=as_clean_str(record.get("analysis_context")),
                row=record,
            )
        )

    level_label = comment_level_label(config.comment_level)
    logging.info("Total comments: %s", original_count)
    logging.info("Selected %s comments: %s", level_label, selected_comment_count)
    logging.info(
        "%s comments with text_quality=3: %s",
        level_label,
        high_quality_selected_count,
    )
    logging.info("Already completed and skipped: %s", skipped_count)
    logging.info("Pending in this run: %s", len(samples))
    return samples, selected_comment_count, skipped_count


def model_dump(instance: Any) -> dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(mode="json")
    return instance.dict()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def validate_target_text_source(record: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    text = as_clean_str(record.get("content"))
    role = record.get("role_layer") or {}

    if role.get("target_explicit") is True:
        for key in ["emotion_target_text", "stance_target_text"]:
            target_text = as_clean_str(role.get(key))
            if target_text and target_text not in text:
                warnings.append(
                    f"{key}={target_text!r} not found in comment_text while target_explicit=True"
                )
    return warnings


def make_jsonl_record(
    sample: CommentSample,
    analysis: CommentAnalysis,
    config: AnalysisConfig,
    analyzed_at: str,
) -> dict[str, Any]:
    analysis_data: dict[str, Any] = model_dump(analysis)
    record: dict[str, Any] = {
        "comment_id": sample.comment_id,
        "weibo_id": sample.weibo_id,
        "parent_id": sample.parent_id,
        "content": sample.content,
        "parent_comment_text": sample.parent_comment_text,
        "analysis_context": sample.analysis_context,
        "topic": sample.topic,
    }
    record.update(analysis_data)
    semantic_validation_warnings = validate_target_text_source(record)
    if semantic_validation_warnings:
        logging.warning(
            "Semantic validation warnings for comment_id=%s: %s",
            sample.comment_id,
            "; ".join(semantic_validation_warnings),
        )
    record["semantic_validation_warnings"] = semantic_validation_warnings
    record["analysis_info"] = {
        "model_name": config.model,
        "analyzed_at": analyzed_at,
        "run_id": config.run_id,
    }
    return record


def flatten_analysis_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "comment_id": as_clean_str(record.get("comment_id")),
        "weibo_id": as_clean_str(record.get("weibo_id")),
        "parent_id": as_clean_str(record.get("parent_id")) or "-1",
        "content": as_clean_str(record.get("content")),
        "parent_comment_text": as_clean_str(record.get("parent_comment_text")),
        "analysis_context": as_clean_str(record.get("analysis_context")),
    }

    if "topic" in record:
        flat["topic"] = as_clean_str(record.get("topic"))

    role_layer = record.get("role_layer") or {}
    appraisal_layer = record.get("appraisal_layer") or {}
    task_output_layer = record.get("task_output_layer") or {}
    evidence = record.get("evidence") or {}
    review_flags = record.get("review_flags") or {}
    analysis_info = record.get("analysis_info") or {}
    semantic_validation_warnings = record.get("semantic_validation_warnings") or []
    if isinstance(semantic_validation_warnings, list):
        semantic_validation_warnings_text = "; ".join(map(str, semantic_validation_warnings))
    else:
        semantic_validation_warnings_text = as_clean_str(semantic_validation_warnings)

    flat.update(
        {
            "emotion_target_type": role_layer.get("emotion_target_type"),
            "emotion_target_text": role_layer.get("emotion_target_text"),
            "stance_target_type": role_layer.get("stance_target_type"),
            "stance_target_text": role_layer.get("stance_target_text"),
            "cause_or_stimulus": role_layer.get("cause_or_stimulus"),
            "target_explicit": role_layer.get("target_explicit"),
            "focus_type": appraisal_layer.get("focus_type"),
            "responsibility": appraisal_layer.get("responsibility"),
            "control": appraisal_layer.get("control"),
            "norm_violation": appraisal_layer.get("norm_violation"),
            "emotion_label": task_output_layer.get("emotion_label"),
            "emotion_intensity": task_output_layer.get("emotion_intensity"),
            "stance_label": task_output_layer.get("stance_label"),
            "stance_intensity": task_output_layer.get("stance_intensity"),
            "argument_type": task_output_layer.get("argument_type"),
            "confidence": task_output_layer.get("confidence"),
            "emotion_evidence": evidence.get("emotion_evidence"),
            "stance_evidence": evidence.get("stance_evidence"),
            "needs_more_context": review_flags.get("needs_more_context"),
            "low_confidence_reason": review_flags.get("low_confidence_reason"),
            "semantic_validation_warnings": semantic_validation_warnings_text,
            "model_name": analysis_info.get("model_name"),
            "analyzed_at": analysis_info.get("analyzed_at"),
            "run_id": analysis_info.get("run_id"),
        }
    )

    for field in ANALYSIS_FIELDS:
        flat.setdefault(field, None)
    return flat


def make_failure_record(
    sample: CommentSample,
    error: Exception,
    retry_count: int,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "comment_id": sample.comment_id,
        "weibo_id": sample.weibo_id,
        "parent_id": sample.parent_id,
        "analysis_context": sample.analysis_context,
        "content": sample.content,
        "parent_comment_text": sample.parent_comment_text,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "retry_count": retry_count,
        "failed_at": now_iso(),
    }


def archive_existing_failed_log(failed_jsonl: Path, run_id: str) -> Path | None:
    if not failed_jsonl.exists() or failed_jsonl.stat().st_size == 0:
        return None

    archive_path = failed_jsonl.with_name(
        f"{failed_jsonl.stem}_{run_id}_previous{failed_jsonl.suffix}"
    )
    counter = 1
    while archive_path.exists():
        archive_path = failed_jsonl.with_name(
            f"{failed_jsonl.stem}_{run_id}_previous_{counter}{failed_jsonl.suffix}"
        )
        counter += 1

    failed_jsonl.replace(archive_path)
    return archive_path


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSONL line %s in %s", line_number, path)
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def dedupe_records_by_comment_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        comment_id = as_clean_str(record.get("comment_id"))
        if comment_id:
            deduped[comment_id] = record
    return list(deduped.values())


def strip_redundant_jsonl_fields(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    cleaned.pop("comment_metadata", None)
    return cleaned


def compact_jsonl(path: Path) -> int:
    records = read_jsonl_records(path)
    if not records:
        return 0

    cleaned_records = [strip_redundant_jsonl_fields(record) for record in records]
    deduped = dedupe_records_by_comment_id(cleaned_records)
    if len(deduped) != len(records) or cleaned_records != records:
        write_jsonl_records(path, deduped)
        logging.info(
            "Compacted %s from %s to %s unique cleaned records",
            path,
            len(records),
            len(deduped),
        )
    return len(deduped)


def load_completed_ids(jsonl_path: Path, parquet_path: Path) -> set[str]:
    completed_ids: set[str] = set()
    for record in read_jsonl_records(jsonl_path):
        comment_id = as_clean_str(record.get("comment_id"))
        if comment_id:
            completed_ids.add(comment_id)

    if parquet_path.exists():
        try:
            df_existing = pd.read_parquet(parquet_path, columns=["comment_id"])
            completed_ids.update(df_existing["comment_id"].map(as_clean_str).tolist())
        except Exception as exc:
            logging.warning("Could not read completed ids from parquet: %s", exc)

    return completed_ids


def save_parquet_from_jsonl(jsonl_path: Path, parquet_path: Path) -> int:
    records = dedupe_records_by_comment_id(read_jsonl_records(jsonl_path))
    flat_records = [flatten_analysis_record(record) for record in records]
    frames: list[pd.DataFrame] = []
    if parquet_path.exists():
        try:
            frames.append(pd.read_parquet(parquet_path))
        except Exception as exc:
            logging.warning("Could not merge existing parquet before saving: %s", exc)
    if flat_records:
        frames.append(pd.DataFrame(flat_records))

    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not df.empty:
        df["comment_id"] = df["comment_id"].map(as_clean_str)
        df = df.drop_duplicates(subset=["comment_id"], keep="last")
        for column in PARQUET_OUTPUT_FIELDS:
            if column not in df.columns:
                df[column] = None
        df = df[PARQUET_OUTPUT_FIELDS]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return len(df)


class ResultWriter:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.lock = asyncio.Lock()
        self.success_count = 0
        self.failure_count = 0

    async def write_success(self, record: dict[str, Any]) -> None:
        async with self.lock:
            self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.config.output_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            self.success_count += 1
            if (
                self.config.checkpoint_every > 0
                and self.success_count % self.config.checkpoint_every == 0
            ):
                row_count = save_parquet_from_jsonl(
                    self.config.output_jsonl,
                    self.config.output_parquet,
                )
                logging.info(
                    "Checkpoint parquet saved: %s rows -> %s",
                    row_count,
                    self.config.output_parquet,
                )

    async def write_failure(self, record: dict[str, Any]) -> None:
        async with self.lock:
            self.config.failed_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.config.failed_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.failure_count += 1
