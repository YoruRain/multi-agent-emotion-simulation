from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "implicit_topic_clustering"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "implicit_topic_cluster_review.log"
REQUIRED_COLUMNS = {"cluster_id", "analysis_text", "distance_to_center"}


def configure_logging(
    verbose: bool = False,
    log_dir: Path = DEFAULT_LOG_DIR,
    log_file: Path | None = None,
) -> Path:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build review CSV files for implicit topic clustering outputs."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing implicit_topic_clustering_k*.parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for review outputs.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Review output format. Defaults to parquet.",
    )
    parser.add_argument(
        "--center_example_count",
        type=int,
        default=10,
        help="Number of nearest-to-center examples to save per cluster.",
    )
    parser.add_argument(
        "--random_example_count",
        type=int,
        default=10,
        help="Number of random examples to save per cluster.",
    )
    parser.add_argument(
        "--edge_example_count",
        type=int,
        default=10,
        help="Number of edge examples to save per cluster.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for random example sampling.",
    )
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for log files.")
    parser.add_argument("--log_file", type=Path, default=None, help="Optional explicit log file path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_k_from_path(path: Path) -> int:
    match = re.search(r"_k(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Cannot infer k from filename: {path.name}")
    return int(match.group(1))


def list_clustering_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("implicit_topic_clustering_k*.parquet"))
    if not files:
        raise FileNotFoundError(f"No clustering parquet files found in: {input_dir}")
    return files


def load_clustering_result(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow or fastparquet in the active Python environment."
        ) from exc

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input dataframe {path} is missing required columns: {missing}")

    prepared = df.copy()
    prepared["analysis_text"] = prepared["analysis_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    prepared["distance_to_center"] = pd.to_numeric(prepared["distance_to_center"], errors="coerce")
    prepared["cluster_id"] = pd.to_numeric(prepared["cluster_id"], errors="raise").astype(int)
    prepared = prepared.dropna(subset=["distance_to_center"]).reset_index(drop=True)
    return prepared


def serialize_examples(df: pd.DataFrame) -> str:
    payload = []
    for row in df.itertuples(index=False):
        payload.append(
            {
                "distance": round(float(row.distance_to_center), 6),
                "text": row.analysis_text,
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def build_review_dataframe(
    df: pd.DataFrame,
    *,
    center_example_count: int,
    random_example_count: int,
    edge_example_count: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for cluster_id, group in df.groupby("cluster_id", sort=True):
        cluster = group.sort_values("distance_to_center", ascending=True).reset_index(drop=True)
        cluster_size = int(len(cluster))
        center_examples = cluster.head(center_example_count)
        edge_examples = cluster.sort_values("distance_to_center", ascending=False).head(edge_example_count)
        random_examples = cluster.sample(
            n=min(random_example_count, cluster_size),
            random_state=random_state + int(cluster_id),
            replace=False,
        )

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "mean_distance": round(float(cluster["distance_to_center"].mean()), 6),
                "min_distance": round(float(cluster["distance_to_center"].min()), 6),
                "max_distance": round(float(cluster["distance_to_center"].max()), 6),
                "center_examples": serialize_examples(center_examples),
                "random_examples": serialize_examples(random_examples),
                "edge_examples": serialize_examples(edge_examples),
            }
        )

    review_df = pd.DataFrame(rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    return review_df


def save_review_csv(review_df: pd.DataFrame, output_path: Path) -> None:
    review_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved %d review rows to %s", len(review_df), output_path)


def save_review_parquet(review_df: pd.DataFrame, output_path: Path) -> None:
    review_df.to_parquet(output_path, index=False)
    LOGGER.info("Saved %d review rows to %s", len(review_df), output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, log_dir=args.log_dir, log_file=args.log_file)

    input_dir = args.input_dir.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve())
    LOGGER.info("Using input_dir=%s", input_dir)
    LOGGER.info("Using output_dir=%s", output_dir)
    LOGGER.info("Using output_format=%s", args.output_format)

    clustering_files = list_clustering_files(input_dir)
    LOGGER.info("Found %d clustering files", len(clustering_files))

    for clustering_file in clustering_files:
        k = extract_k_from_path(clustering_file)
        LOGGER.info("Building review CSV for k=%d from %s", k, clustering_file)
        clustering_df = load_clustering_result(clustering_file)
        review_df = build_review_dataframe(
            clustering_df,
            center_example_count=args.center_example_count,
            random_example_count=args.random_example_count,
            edge_example_count=args.edge_example_count,
            random_state=args.random_state,
        )
        if args.output_format == "csv":
            output_path = output_dir / f"implicit_topic_clustering_review_k{k}.csv"
            save_review_csv(review_df, output_path)
        else:
            output_path = output_dir / f"implicit_topic_clustering_review_k{k}.parquet"
            save_review_parquet(review_df, output_path)
        LOGGER.info(
            "k=%d summary: clusters=%d min_cluster_size=%d max_cluster_size=%d",
            k,
            len(review_df),
            int(review_df["cluster_size"].min()),
            int(review_df["cluster_size"].max()),
        )


if __name__ == "__main__":
    main()
