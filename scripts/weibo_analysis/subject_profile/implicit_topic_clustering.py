from __future__ import annotations

"""
Example (cmd.exe):
python scripts/weibo_analysis/implicit_topic_clustering.py ^
  --input data/profile/weibos/subject_profile/candidate_weibos_for_implicit_topic.parquet ^
  --output_dir data/profile/weibos/subject_profile/implicit_topic_clustering ^
  --model_name BAAI/bge-small-zh-v1.5 ^
  --batch_size 64 ^
  --max_length 256 ^
  --k_values 10 15 20 25 30

Example (PowerShell):
python scripts/weibo_analysis/implicit_topic_clustering.py --input data/profile/weibos/subject_profile/candidate_weibos_for_implicit_topic.parquet --output_dir data/profile/weibos/subject_profile/implicit_topic_clustering --model_name BAAI/bge-small-zh-v1.5 --batch_size 64 --max_length 256 --k_values 10 15 20 25 30
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "candidate_weibos_for_implicit_topic.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "subject_profile"
    / "implicit_topic_clustering"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "implicit_topic_clustering.log"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_FILE_NAME = "candidate_embeddings_bge_small_zh_v1_5.npy"
DEFAULT_EMBEDDING_META_FILE_NAME = "candidate_embeddings_meta.json"
DEFAULT_EVAL_FILE_NAME = "cluster_eval_summary.csv"
REQUIRED_COLUMNS = {"weibo_id", "user_id", "content"}
DEFAULT_K_VALUES = [10, 15, 20, 25, 30]
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 3000


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
        description="Generate BGE embeddings and run MiniBatchKMeans experiments for implicit topic clustering."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Candidate weibo parquet path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for embedding cache and clustering outputs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Embedding model name or local model path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum text length passed to the embedding model.",
    )
    parser.add_argument(
        "--force_rebuild_embeddings",
        action="store_true",
        help="Force rebuild embeddings even if cache files already exist.",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="List of k values to try.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for clustering and silhouette sampling.",
    )
    parser.add_argument(
        "--kmeans_batch_size",
        type=int,
        default=1024,
        help="MiniBatchKMeans batch_size parameter.",
    )
    parser.add_argument(
        "--silhouette_sample_size",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE_SIZE,
        help="Maximum number of samples used for silhouette scoring.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for log files.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional explicit log file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def load_and_prepare_candidates(input_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported, got: {input_path}")

    try:
        df = pd.read_parquet(input_path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow or fastparquet in the active Python environment."
        ) from exc

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input dataframe {input_path} is missing required columns: {missing}")

    raw_count = int(len(df))
    working = df.copy()
    working["analysis_text"] = working["content"].map(clean_text)
    working["analysis_text_length"] = working["analysis_text"].map(len).astype(int)

    empty_mask = working["analysis_text_length"] == 0
    short_mask = working["analysis_text_length"] < 2
    removed_empty_count = int(empty_mask.sum())
    removed_short_count = int((short_mask & ~empty_mask).sum())

    prepared = working.loc[~short_mask].copy()
    prepared["analysis_text_length"] = prepared["analysis_text_length"].astype(int)
    prepared = prepared.reset_index(drop=True)

    stats = {
        "raw_count": raw_count,
        "cleaned_count": int(len(prepared)),
        "removed_empty_count": removed_empty_count,
        "removed_short_count": removed_short_count,
    }
    return prepared, stats


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    normalized = normalize(embeddings, norm="l2", axis=1, copy=False)
    return np.asarray(normalized, dtype=np.float32)


def load_embedding_cache(
    embedding_path: Path,
    meta_path: Path,
    input_path: Path,
    model_name: str,
    expected_count: int,
) -> np.ndarray | None:
    if not embedding_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embeddings = np.load(embedding_path)
    except Exception as exc:
        LOGGER.warning("Failed to load embedding cache. Will rebuild. reason=%s", exc)
        return None

    if embeddings.ndim != 2:
        LOGGER.warning("Cached embeddings have invalid shape %s. Will rebuild.", embeddings.shape)
        return None

    if int(embeddings.shape[0]) != expected_count:
        LOGGER.warning(
            "Cached embedding count %d does not match candidate count %d. Will rebuild.",
            embeddings.shape[0],
            expected_count,
        )
        return None

    if meta.get("embedding_model") != model_name:
        LOGGER.warning(
            "Cached embedding model %s does not match requested model %s. Will rebuild.",
            meta.get("embedding_model"),
            model_name,
        )
        return None

    if meta.get("input_file") != str(input_path.resolve()):
        LOGGER.warning(
            "Cached input_file %s does not match requested input %s. Will rebuild.",
            meta.get("input_file"),
            str(input_path.resolve()),
        )
        return None

    if not bool(meta.get("normalize_embeddings", False)):
        LOGGER.warning("Cached embeddings are marked as non-normalized. Will rebuild.")
        return None

    return np.asarray(embeddings, dtype=np.float32)


def build_flag_model(model_name: str) -> Any:
    try:
        from FlagEmbedding import FlagModel
    except ImportError as exc:
        raise ImportError(
            "FlagEmbedding is required to build embeddings. Install it in the active environment first."
        ) from exc

    base_kwargs = {
        "query_instruction_for_retrieval": "",
        "normalize_embeddings": False,
        "use_fp16": True,
    }

    try:
        return FlagModel(model_name, **base_kwargs)
    except TypeError:
        # Some FlagEmbedding versions expose a smaller constructor surface.
        LOGGER.warning("FlagModel constructor signature differs from the current environment. Falling back.")
        try:
            return FlagModel(model_name, query_instruction_for_retrieval="", use_fp16=True)
        except TypeError:
            return FlagModel(model_name, use_fp16=True)
    except Exception as exc:
        LOGGER.warning("Failed to initialize FlagModel with fp16. Falling back to fp32. reason=%s", exc)
        fallback_kwargs = dict(base_kwargs)
        fallback_kwargs["use_fp16"] = False
        try:
            return FlagModel(model_name, **fallback_kwargs)
        except TypeError:
            try:
                return FlagModel(model_name, query_instruction_for_retrieval="", use_fp16=False)
            except TypeError:
                return FlagModel(model_name, use_fp16=False)


def encode_texts(model: Any, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    batches: list[np.ndarray] = []
    total = len(texts)
    iterator = range(0, total, batch_size)
    for start in tqdm(iterator, total=(total + batch_size - 1) // batch_size, desc="Encoding", unit="batch"):
        batch_texts = texts[start : start + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=True,
        )
        batches.append(np.asarray(batch_embeddings, dtype=np.float32))
    return np.vstack(batches)


def write_embedding_metadata(
    meta_path: Path,
    *,
    model_name: str,
    candidate_count: int,
    embedding_dim: int,
    input_path: Path,
    embedding_path: Path,
    normalized: bool,
) -> None:
    metadata = {
        "embedding_model": model_name,
        "candidate_count": int(candidate_count),
        "embedding_dim": int(embedding_dim),
        "normalize_embeddings": bool(normalized),
        "input_file": str(input_path.resolve()),
        "embedding_file": str(embedding_path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_or_build_embeddings(
    texts: list[str],
    output_dir: Path,
    input_path: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    force_rebuild: bool,
) -> tuple[np.ndarray, bool]:
    embedding_path = output_dir / DEFAULT_EMBEDDING_FILE_NAME
    meta_path = output_dir / DEFAULT_EMBEDDING_META_FILE_NAME

    if not force_rebuild:
        cached = load_embedding_cache(
            embedding_path=embedding_path,
            meta_path=meta_path,
            input_path=input_path,
            model_name=model_name,
            expected_count=len(texts),
        )
        if cached is not None:
            LOGGER.info("Embedding cache hit: %s", embedding_path)
            return cached, True

    LOGGER.info("Building embeddings with model=%s", model_name)
    model = build_flag_model(model_name)
    embeddings = encode_texts(
        model=model,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
    )
    embeddings = normalize_embeddings(embeddings)
    np.save(embedding_path, embeddings)
    write_embedding_metadata(
        meta_path,
        model_name=model_name,
        candidate_count=len(texts),
        embedding_dim=embeddings.shape[1],
        input_path=input_path,
        embedding_path=embedding_path,
        normalized=True,
    )
    LOGGER.info("Saved embedding cache to %s", embedding_path)
    LOGGER.info("Saved embedding metadata to %s", meta_path)
    return embeddings, False


def create_minibatch_kmeans(k: int, random_state: int, batch_size: int) -> MiniBatchKMeans:
    try:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            n_init="auto",
        )
    except TypeError:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            n_init=10,
        )


def run_minibatch_kmeans(
    embeddings: np.ndarray,
    k: int,
    random_state: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        model = create_minibatch_kmeans(k=k, random_state=random_state, batch_size=batch_size)
        labels = model.fit_predict(embeddings)
    except ValueError as exc:
        if "n_init" not in str(exc):
            raise
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            n_init=10,
        )
        labels = model.fit_predict(embeddings)

    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    inertia = float(model.inertia_)
    return labels.astype(int), centers, inertia


def compute_distance_to_center(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    assigned_centers = centers[labels]
    distances = np.linalg.norm(embeddings - assigned_centers, axis=1)
    return np.asarray(distances, dtype=np.float32)


def compute_silhouette_sample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        LOGGER.warning("Silhouette score skipped because fewer than 2 clusters were produced.")
        return float("nan")

    sample_count = min(len(labels), sample_size)
    if sample_count < 2:
        LOGGER.warning("Silhouette score skipped because fewer than 2 samples are available.")
        return float("nan")

    try:
        if len(labels) > sample_count:
            rng = np.random.default_rng(random_state)
            sample_indices = rng.choice(len(labels), size=sample_count, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = labels[sample_indices]
        else:
            sample_embeddings = embeddings
            sample_labels = labels

        if len(np.unique(sample_labels)) < 2:
            LOGGER.warning("Silhouette score sample skipped because the sample only contains one cluster.")
            return float("nan")

        score = silhouette_score(sample_embeddings, sample_labels, metric="euclidean")
        return float(score)
    except Exception as exc:
        LOGGER.warning("Silhouette score computation failed. Returning NaN. reason=%s", exc)
        return float("nan")


def save_clustering_result(
    df_candidates: pd.DataFrame,
    labels: np.ndarray,
    distances: np.ndarray,
    k: int,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"implicit_topic_clustering_k{k}.parquet"
    result_df = df_candidates.loc[:, ["weibo_id", "user_id", "analysis_text", "analysis_text_length"]].copy()
    result_df["cluster_id"] = labels.astype(int)
    result_df["distance_to_center"] = distances.astype(float)
    result_df.to_parquet(output_path, index=False)
    return output_path


def build_cluster_eval_row(
    *,
    k: int,
    labels: np.ndarray,
    inertia: float,
    silhouette: float,
    random_state: int,
    batch_size: int,
) -> dict[str, Any]:
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "k": int(k),
        "candidate_count": int(len(labels)),
        "cluster_count": int(cluster_sizes.shape[0]),
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
        "mean_cluster_size": float(cluster_sizes.mean()),
        "median_cluster_size": float(cluster_sizes.median()),
        "inertia": float(inertia),
        "silhouette_score_sample": float(silhouette),
        "random_state": int(random_state),
        "batch_size": int(batch_size),
    }


def sanitize_k_values(k_values: list[int], candidate_count: int) -> list[int]:
    sanitized: list[int] = []
    for k in dict.fromkeys(k_values):
        if k < 2:
            LOGGER.warning("Skipping invalid k=%d because k must be at least 2.", k)
            continue
        if k > candidate_count:
            LOGGER.warning(
                "Skipping invalid k=%d because it exceeds candidate_count=%d.",
                k,
                candidate_count,
            )
            continue
        sanitized.append(k)
    if not sanitized:
        raise ValueError("No valid k values remain after validation.")
    return sanitized


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, log_dir=args.log_dir, log_file=args.log_file)

    output_dir = ensure_output_dir(args.output_dir)
    LOGGER.info("Input file: %s", args.input.resolve())
    LOGGER.info("Output directory: %s", output_dir.resolve())

    df_candidates, prep_stats = load_and_prepare_candidates(args.input)
    LOGGER.info("Raw candidate count: %d", prep_stats["raw_count"])
    LOGGER.info("Cleaned candidate count: %d", prep_stats["cleaned_count"])
    LOGGER.info("Removed empty text rows: %d", prep_stats["removed_empty_count"])
    LOGGER.info("Removed short text rows (<2 chars): %d", prep_stats["removed_short_count"])

    texts = df_candidates["analysis_text"].tolist()
    embeddings, used_cache = load_or_build_embeddings(
        texts=texts,
        output_dir=output_dir,
        input_path=args.input,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        force_rebuild=args.force_rebuild_embeddings,
    )
    LOGGER.info("Embedding model: %s", args.model_name)
    LOGGER.info("Embedding shape: %s", embeddings.shape)
    LOGGER.info("Embedding cache used: %s", used_cache)

    k_values = sanitize_k_values(args.k_values, candidate_count=len(df_candidates))
    eval_rows: list[dict[str, Any]] = []

    for k in k_values:
        LOGGER.info("Running MiniBatchKMeans for k=%d", k)
        labels, centers, inertia = run_minibatch_kmeans(
            embeddings=embeddings,
            k=k,
            random_state=args.random_state,
            batch_size=args.kmeans_batch_size,
        )
        distances = compute_distance_to_center(
            embeddings=embeddings,
            labels=labels,
            centers=centers,
        )
        silhouette = compute_silhouette_sample(
            embeddings=embeddings,
            labels=labels,
            random_state=args.random_state,
            sample_size=args.silhouette_sample_size,
        )
        output_path = save_clustering_result(
            df_candidates=df_candidates,
            labels=labels,
            distances=distances,
            k=k,
            output_dir=output_dir,
        )
        eval_row = build_cluster_eval_row(
            k=k,
            labels=labels,
            inertia=inertia,
            silhouette=silhouette,
            random_state=args.random_state,
            batch_size=args.kmeans_batch_size,
        )
        eval_rows.append(eval_row)

        LOGGER.info("Saved clustering result for k=%d to %s", k, output_path)
        LOGGER.info(
            "k=%d stats: clusters=%d min=%d max=%d mean=%.2f median=%.2f inertia=%.4f silhouette=%.6f",
            k,
            eval_row["cluster_count"],
            eval_row["min_cluster_size"],
            eval_row["max_cluster_size"],
            eval_row["mean_cluster_size"],
            eval_row["median_cluster_size"],
            eval_row["inertia"],
            eval_row["silhouette_score_sample"],
        )

    eval_df = pd.DataFrame(eval_rows).sort_values("k").reset_index(drop=True)
    eval_output_path = output_dir / DEFAULT_EVAL_FILE_NAME
    eval_df.to_csv(eval_output_path, index=False)
    LOGGER.info("Saved cluster evaluation summary to %s", eval_output_path)


if __name__ == "__main__":
    main()
