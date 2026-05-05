from __future__ import annotations

import ast
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / ".log"


def configure_logging(
    verbose: bool = False,
    log_file_name: str = "agent_data_preparation.log",
    log_dir: Path = DEFAULT_LOG_DIR,
) -> Path:
    level = logging.DEBUG if verbose else logging.INFO
    log_path = log_dir / log_file_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)

    LOGGER.info("日志写入 %s", log_path)
    return log_path


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.error("输入文件不存在: %s", path)
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path, encoding="utf-8")
        elif suffix == ".jsonl":
            df = pd.read_json(path, lines=True, encoding="utf-8")
        elif suffix == ".json":
            df = pd.read_json(path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported input suffix: {suffix}")
    except Exception:
        LOGGER.exception("读取失败: %s", path)
        raise

    LOGGER.info("读取成功: %s, 行数=%d, 字段数=%d", path, len(df), len(df.columns))
    return df


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        LOGGER.info("输出文件已存在，将覆盖: %s", path)

    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    LOGGER.info("写入 JSONL 完成: %s, 行数=%d", path, count)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_id(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def safe_get(row: pd.Series | dict[str, Any], field: str, default: Any = None) -> Any:
    value = row.get(field, default)
    return default if is_missing(value) else value


def safe_str(value: Any, default: str = "未知") -> str:
    if is_missing(value):
        return default
    text = str(value).strip()
    return text if text else default


def safe_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return default
    return result


def parse_bool(value: Any, default: bool = False) -> bool:
    if is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是", "真"}:
        return True
    if text in {"false", "0", "no", "n", "否", "假"}:
        return False
    return default


def parse_list_like(value: Any) -> list[Any]:
    if is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        return [item.item() if hasattr(item, "item") else item for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return list(value)

    text = str(value).strip()
    if not text:
        return []

    if text.startswith(("[", "(", "{")):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return list(parsed.keys())
        if isinstance(parsed, (list, tuple, set)):
            items: list[Any] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and item:
                    items.append(item[0])
                else:
                    items.append(item)
            return items

    return [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]


def parse_mapping_like(value: Any) -> dict[str, Any]:
    if is_missing(value):
        return {}
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): val for key, val in parsed.items()}


def stringify_list(value: Any) -> str:
    return ",".join(safe_str(item, "").strip() for item in parse_list_like(value) if safe_str(item, "").strip())
