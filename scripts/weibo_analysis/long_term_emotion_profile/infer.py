from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

try:
    from .features import compute_rule_intensity, compute_text_weight, prepare_texts
    from .io_utils import (
        filter_analyzed_records,
        load_existing_analyzed_ids,
        load_input_dataframe,
        load_task_config,
        sample_input_dataframe,
        save_dataframe,
    )
except ImportError:
    from features import compute_rule_intensity, compute_text_weight, prepare_texts
    from io_utils import (
        filter_analyzed_records,
        load_existing_analyzed_ids,
        load_input_dataframe,
        load_task_config,
        sample_input_dataframe,
        save_dataframe,
    )

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "macbert_finetuned_sentiment"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "high_quality" / "user_weibo.parquet"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "profile"
    / "weibos"
    / "emotion_profile"
    / "user_weibo_emotion_analysis.parquet"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"
DEFAULT_LOG_FILE_NAME = "user_weibo_emotion.log"

LABEL_EN_BY_ID = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_ZH_BY_EN = {"Negative": "消极", "Neutral": "中性", "Positive": "积极"}


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

    LOGGER.info(
        "Logging to %s",
        log_file,
    )
    return log_file


def set_reproducible_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    model_dir: Path,
    task_config: dict[str, Any],
    device: Any,
) -> tuple[Any, Any]:
    try:
        from transformers import BertModel, BertTokenizer

        try:
            from .model import MacBertSentimentClassifier, load_classifier_state
        except ImportError:
            from model import MacBertSentimentClassifier, load_classifier_state
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing ML dependency. Please install transformers and torch in the active "
            "Python environment before running inference."
        ) from exc

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    bert = BertModel.from_pretrained(model_dir)
    model = MacBertSentimentClassifier(
        bert=bert,
        num_class=int(task_config.get("num_class", 3)),
        dropout=float(task_config.get("dropout", 0.3)),
    )
    load_classifier_state(model, model_dir / "model_state.pt", device)

    model.to(device)
    model.eval()
    LOGGER.info("Loaded MacBERT sentiment model on %s", device)
    return model, tokenizer


def _ordered_label_ids(task_config: dict[str, Any]) -> list[int]:
    label2id = task_config.get("label2id", {"Negative": 0, "Neutral": 1, "Positive": 2})
    return [int(label2id.get(label, index)) for index, label in enumerate(("Negative", "Neutral", "Positive"))]


def predict_sentiment(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    max_len: int,
    batch_size: int = 64,
    task_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    import torch

    if not texts:
        return pd.DataFrame(columns=["p_neg", "p_neu", "p_pos", "pred_id"])

    label_ids = _ordered_label_ids(task_config or {})
    probabilities: list[list[float]] = []
    pred_ids: list[int] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Predicting sentiment"):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            probabilities.extend(probs[:, label_ids].detach().cpu().tolist())
            pred_ids.extend(probs.argmax(dim=1).detach().cpu().tolist())

    result = pd.DataFrame(probabilities, columns=["p_neg", "p_neu", "p_pos"])
    result["pred_id"] = pred_ids
    return result


def _get_column_or_default(df: pd.DataFrame, column: str, default: Any = None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def build_weibo_emotion_table(
    df: pd.DataFrame,
    texts: list[str],
    predictions: pd.DataFrame,
    feature_texts: list[str] | None = None,
) -> pd.DataFrame:
    if len(df) != len(predictions) or len(df) != len(texts):
        raise ValueError("Dataframe, prepared texts, and predictions must have the same length.")
    if feature_texts is None:
        feature_texts = texts
    if len(df) != len(feature_texts):
        raise ValueError("Feature texts must have the same length as the dataframe.")

    output = pd.DataFrame(index=df.index)
    output["weibo_id"] = _get_column_or_default(df, "weibo_id")
    output["user_id"] = _get_column_or_default(df, "user_id")
    output["content"] = [
        row["content"] if "content" in df.columns and pd.notna(row.get("content")) else text
        for text, (_, row) in zip(texts, df.iterrows())
    ]
    output["cleaned_text_length"] = _get_column_or_default(df, "cleaned_text_length")
    output["year"] = _get_column_or_default(df, "year")
    output["is_repost"] = _get_column_or_default(df, "is_repost", False)

    output[["p_neg", "p_neu", "p_pos"]] = predictions[["p_neg", "p_neu", "p_pos"]].to_numpy()
    output["sentiment_label_en"] = predictions["pred_id"].map(LABEL_EN_BY_ID).fillna("Neutral")
    output["sentiment_label"] = output["sentiment_label_en"].map(LABEL_ZH_BY_EN).fillna("中性")
    output["model_confidence"] = output[["p_neg", "p_neu", "p_pos"]].max(axis=1)
    output["polarity_score"] = output["p_pos"] - output["p_neg"]
    output["model_intensity"] = 1.0 - output["p_neu"]
    output["rule_intensity"] = [compute_rule_intensity(text) for text in feature_texts]
    output["emotion_intensity_score"] = 0.5 * output["model_intensity"] + 0.5 * output["rule_intensity"]
    output["text_weight"] = [
        compute_text_weight(text, is_repost, model_confidence)
        for text, is_repost, model_confidence in zip(
            feature_texts,
            output["is_repost"].tolist(),
            output["model_confidence"].tolist(),
        )
    ]

    ordered_columns = [
        "weibo_id",
        "user_id",
        "content",
        "cleaned_text_length",
        "year",
        "is_repost",
        "sentiment_label_en",
        "sentiment_label",
        "p_neg",
        "p_neu",
        "p_pos",
        "model_confidence",
        "polarity_score",
        "model_intensity",
        "rule_intensity",
        "emotion_intensity_score",
        "text_weight",
    ]
    return output[ordered_columns].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MacBERT sentiment inference for user weibo records.",
    )
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH, help="Input parquet file.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output .parquet or .jsonl path.")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="Fine-tuned MacBERT model directory.")
    parser.add_argument("--min_text_quality", type=float, default=3, help="Minimum text_quality to keep.")
    parser.add_argument("--no_quality_filter", action="store_true", help="Disable text_quality filtering.")
    parser.add_argument(
        "--include_reposts",
        action="store_true",
        help="Analyze reposted weibos too. By default, only original weibos (is_repost == False) are analyzed.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument(
        "--max_records",
        "--limit",
        dest="max_records",
        type=int,
        default=None,
        help="Analyze at most N rows after quality filtering and resume filtering.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample rows when --max_records is set instead of taking the first N rows.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not skip weibo_id values already present in the output file.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Overwrite the output file instead of merging new rows with existing output.",
    )
    parser.add_argument("--max_len", type=int, default=128, help="Override max_len from task_config.json.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible inference.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for run log files.")
    parser.add_argument("--log_file", type=Path, default=None, help="Optional explicit log file path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, log_dir=args.log_dir, log_file=args.log_file)

    task_config = load_task_config(args.model_dir)
    max_len = int(args.max_len or task_config.get("max_len", 64))

    LOGGER.info("Using model_dir=%s", args.model_dir)
    LOGGER.info("Using max_len=%s batch_size=%s", max_len, args.batch_size)
    LOGGER.info("Original-only analysis=%s", not args.include_reposts)

    df = load_input_dataframe(
        input_path=args.input_path,
        min_text_quality=args.min_text_quality,
        use_quality_filter=not args.no_quality_filter,
        original_only=not args.include_reposts,
    )
    if not args.no_resume:
        analyzed_ids = load_existing_analyzed_ids(args.output_path)
        df = filter_analyzed_records(df, analyzed_ids)

    df = sample_input_dataframe(
        df,
        max_records=args.max_records,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    if df.empty:
        LOGGER.info("No records left to analyze after filtering and sampling; exiting.")
        return

    raw_texts = _get_column_or_default(df, "content").fillna("").astype(str).tolist()
    model_texts = prepare_texts(df)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing ML dependency. Please install torch in the active Python environment "
            "before running inference."
        ) from exc

    set_reproducible_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(args.model_dir, task_config, device)
    predictions = predict_sentiment(
        texts=model_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=max_len,
        batch_size=args.batch_size,
        task_config=task_config,
    )
    result = build_weibo_emotion_table(df, raw_texts, predictions, feature_texts=model_texts)
    save_dataframe(result, args.output_path, append_existing=not args.overwrite_output)


if __name__ == "__main__":
    main()
