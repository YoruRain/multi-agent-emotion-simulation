from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from transformers import BertModel

LOGGER = logging.getLogger(__name__)


class MacBertSentimentClassifier(nn.Module):
    """Classifier compatible with the training notebook's BERT class.

    The notebook defines:
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_class)
    """

    def __init__(self, bert: BertModel, num_class: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_class)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(cls_output))


def _unwrap_state_dict(raw_state: object) -> Mapping[str, torch.Tensor]:
    if isinstance(raw_state, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            nested = raw_state.get(key)
            if isinstance(nested, Mapping):
                return nested
        return raw_state
    raise TypeError("model_state.pt must contain a PyTorch state_dict or a checkpoint mapping.")


def _normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> "OrderedDict[str, torch.Tensor]":
    model_keys = set(model.state_dict().keys())
    normalized: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    classifier_aliases = (
        ("classifier.", "fc."),
        ("linear.", "fc."),
        ("head.", "fc."),
        ("classification_head.", "fc."),
        ("module.", ""),
        ("model.", ""),
    )

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix, replacement in classifier_aliases:
                if new_key.startswith(prefix):
                    new_key = replacement + new_key[len(prefix) :]
                    changed = True

        if new_key in model_keys:
            normalized[new_key] = value
        else:
            normalized[key] = value

    return normalized


def load_classifier_state(model: nn.Module, state_path: Path, device: torch.device) -> None:
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing model weights: {state_path}. "
            "Please save the fine-tuned checkpoint as model_state.pt before inference."
        )

    raw_state = torch.load(state_path, map_location=device)
    state_dict = _unwrap_state_dict(raw_state)
    normalized = _normalize_state_dict_keys(state_dict, model)
    missing, unexpected = model.load_state_dict(normalized, strict=False)

    allowed_missing = {key for key in missing if key.startswith("bert.pooler.")}
    real_missing = [key for key in missing if key not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            "Failed to load model_state.pt cleanly. "
            f"Missing keys: {real_missing}; unexpected keys: {unexpected}. "
            "The notebook classifier head is expected to use fc.weight/fc.bias."
        )

    if allowed_missing:
        LOGGER.warning("Ignored non-critical missing BERT pooler keys: %s", sorted(allowed_missing))
    LOGGER.info("Loaded classifier weights from %s", state_path)

