from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
ID_COLUMN = "weibo_id"


def load_task_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "task_config.json"
    if not config_path.exists():
        LOGGER.warning("task_config.json not found at %s; using safe defaults.", config_path)
        return {
            "num_class": 3,
            "max_len": 64,
            "label2id": {"Negative": 0, "Neutral": 1, "Positive": 2},
        }

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    config.setdefault("num_class", 3)
    config.setdefault("max_len", 64)
    config.setdefault("label2id", {"Negative": 0, "Neutral": 1, "Positive": 2})
    return config


def load_input_dataframe(
    input_path: Path,
    min_text_quality: float = 3,
    use_quality_filter: bool = True,
    original_only: bool = True,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported, got: {input_path}")

    try:
        df = pd.read_parquet(input_path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow or fastparquet in the active Python "
            "environment. Install one of them before running this script."
        ) from exc
    LOGGER.info("Loaded %d rows from %s", len(df), input_path)

    if use_quality_filter:
        if "text_quality" not in df.columns:
            LOGGER.warning("Column text_quality not found; quality filter was skipped.")
        else:
            before = len(df)
            df = df[pd.to_numeric(df["text_quality"], errors="coerce").fillna(-1) >= min_text_quality].copy()
            LOGGER.info(
                "Applied text_quality >= %s filter: %d -> %d rows",
                min_text_quality,
                before,
                len(df),
            )

    if original_only:
        if "is_repost" not in df.columns:
            LOGGER.warning("Column is_repost not found; original-only filter was skipped.")
        else:
            before = len(df)
            df = df[df["is_repost"] == False].copy()
            LOGGER.info("Applied original-only filter (is_repost == False): %d -> %d rows", before, len(df))

    return df.reset_index(drop=True)


def _read_existing_output(output_path: Path) -> pd.DataFrame:
    suffix = output_path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(output_path, orient="records", lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(output_path)
    raise ValueError(f"Unsupported output suffix {suffix!r}; use .jsonl or .parquet.")


def load_existing_analyzed_ids(output_path: Path, id_column: str = ID_COLUMN) -> set[str]:
    if not output_path.exists():
        LOGGER.info("No existing output found at %s; resume starts from scratch.", output_path)
        return set()

    existing = _read_existing_output(output_path)
    if id_column not in existing.columns:
        LOGGER.warning("Existing output %s has no %s column; resume was skipped.", output_path, id_column)
        return set()

    analyzed_ids = set(existing[id_column].dropna().astype(str))
    LOGGER.info("Loaded %d analyzed %s values from %s", len(analyzed_ids), id_column, output_path)
    return analyzed_ids


def filter_analyzed_records(df: pd.DataFrame, analyzed_ids: set[str], id_column: str = ID_COLUMN) -> pd.DataFrame:
    if not analyzed_ids:
        return df.reset_index(drop=True)
    if id_column not in df.columns:
        LOGGER.warning("Input dataframe has no %s column; resume filtering was skipped.", id_column)
        return df.reset_index(drop=True)

    before = len(df)
    df = df[~df[id_column].astype(str).isin(analyzed_ids)].copy()
    LOGGER.info("Filtered analyzed records by %s: %d -> %d rows", id_column, before, len(df))
    return df.reset_index(drop=True)


def sample_input_dataframe(
    df: pd.DataFrame,
    max_records: int | None = None,
    random_sample: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    if max_records is None:
        return df.reset_index(drop=True)
    if max_records < 0:
        raise ValueError(f"max_records must be non-negative, got: {max_records}")

    before = len(df)
    sample_size = min(max_records, before)
    if random_sample:
        df = df.sample(n=sample_size, random_state=seed).copy()
        LOGGER.info("Applied random max_records=%d sample: %d -> %d rows", max_records, before, len(df))
    else:
        df = df.head(sample_size).copy()
        LOGGER.info("Applied max_records=%d limit: %d -> %d rows", max_records, before, len(df))

    return df.reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, output_path: Path, append_existing: bool = False) -> None:
    if output_path.suffix == "":
        output_path = output_path / "user_weibo_emotion_profile.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    save_df = df

    if append_existing and output_path.exists():
        existing = _read_existing_output(output_path)
        save_df = pd.concat([existing, df], ignore_index=True)
        if ID_COLUMN in save_df.columns:
            before = len(save_df)
            save_df = save_df.drop_duplicates(subset=[ID_COLUMN], keep="first")
            LOGGER.info("Merged with existing output and deduplicated %s: %d -> %d rows", ID_COLUMN, before, len(save_df))

    if suffix == ".jsonl":
        save_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif suffix == ".parquet":
        save_df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output suffix {suffix!r}; use .jsonl or .parquet.")

    LOGGER.info("Saved %d new rows, %d total rows to %s", len(df), len(save_df), output_path)
