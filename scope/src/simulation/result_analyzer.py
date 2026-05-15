from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def _dominant(values: pd.Series) -> str:
    filtered = [str(value) for value in values.dropna().tolist() if str(value).strip()]
    if not filtered:
        return ""
    return Counter(filtered).most_common(1)[0][0]


def _distribution_json(values: pd.Series) -> str:
    filtered = [str(value) for value in values.dropna().tolist() if str(value).strip()]
    return json.dumps(dict(Counter(filtered)), ensure_ascii=False)


def analyze_results(
    reactions_path: Path,
    summary_report_path: Path,
    write_group_reports: bool = True,
) -> pd.DataFrame:
    """Analyze agent reaction JSONL and write summary CSV files."""

    records = _load_jsonl(reactions_path)
    if not records:
        summary = pd.DataFrame(
            [
                {
                    "event_id": "",
                    "topic": "",
                    "total_agents": 0,
                    "success_count": 0,
                    "failed_count": 0,
                    "participation_rate": 0.0,
                    "comment_count": 0,
                    "repost_count": 0,
                    "repost_with_comment_count": 0,
                    "ignore_count": 0,
                    "dominant_emotion": "",
                    "dominant_stance": "",
                },
            ],
        )
        summary_report_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_report_path, index=False, encoding="utf-8-sig")
        return summary

    df = pd.DataFrame(records)
    success_mask = df["parse_status"].eq("success") if "parse_status" in df else pd.Series(False, index=df.index)
    success_df = df[success_mask].copy()

    total_agents = len(df)
    success_count = int(success_mask.sum())
    failed_count = total_agents - success_count
    participation_count = int(success_df.get("participate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    action_counts = Counter(success_df.get("action_type", pd.Series(dtype=str)).dropna().astype(str).tolist())

    summary = pd.DataFrame(
        [
            {
                "event_id": str(df["event_id"].dropna().iloc[0]) if "event_id" in df and not df["event_id"].dropna().empty else "",
                "topic": str(df["topic"].dropna().iloc[0]) if "topic" in df and not df["topic"].dropna().empty else "",
                "total_agents": total_agents,
                "success_count": success_count,
                "failed_count": failed_count,
                "participation_rate": round(participation_count / success_count, 4) if success_count else 0.0,
                "comment_count": action_counts.get("comment", 0),
                "repost_count": action_counts.get("repost", 0),
                "repost_with_comment_count": action_counts.get("repost_with_comment", 0),
                "ignore_count": action_counts.get("ignore", 0),
                "dominant_emotion": _dominant(success_df.get("emotion_label", pd.Series(dtype=str))),
                "dominant_stance": _dominant(success_df.get("stance_label", pd.Series(dtype=str))),
                "emotion_intensity_mean": round(float(success_df.get("emotion_intensity", pd.Series(dtype=float)).mean()), 4)
                if success_count
                else 0.0,
                "stance_intensity_mean": round(float(success_df.get("stance_intensity", pd.Series(dtype=float)).mean()), 4)
                if success_count
                else 0.0,
                "action_type_distribution": _distribution_json(success_df.get("action_type", pd.Series(dtype=str))),
                "emotion_label_distribution": _distribution_json(success_df.get("emotion_label", pd.Series(dtype=str))),
                "stance_label_distribution": _distribution_json(success_df.get("stance_label", pd.Series(dtype=str))),
                "parse_failed_count": int(df.get("parse_status", pd.Series(dtype=str)).eq("parse_failed").sum()),
                "failed_status_count": int(df.get("parse_status", pd.Series(dtype=str)).eq("failed").sum()),
            },
        ],
    )

    summary_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_report_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote summary report to %s", summary_report_path)

    if write_group_reports:
        _write_group_reports(df, summary_report_path.parent)
    return summary


def _write_group_reports(df: pd.DataFrame, output_dir: Path) -> None:
    success_df = df[df.get("parse_status", pd.Series(dtype=str)).eq("success")].copy()
    for column in ["memory_user_level", "influence_level", "action_type"]:
        if column not in success_df:
            continue
        grouped = (
            success_df.groupby(column, dropna=False)
            .agg(
                total_agents=("agent_id", "count"),
                participation_rate=("participate", "mean"),
                emotion_intensity_mean=("emotion_intensity", "mean"),
                stance_intensity_mean=("stance_intensity", "mean"),
            )
            .reset_index()
        )
        grouped.to_csv(output_dir / f"{column}_summary.csv", index=False, encoding="utf-8-sig")
