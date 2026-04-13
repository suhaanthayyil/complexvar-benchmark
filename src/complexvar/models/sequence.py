"""Sequence-only baseline models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from complexvar.features import mutation_descriptor

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None


ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


@dataclass(frozen=True)
class SequenceEncoding:
    vector: np.ndarray
    source: str


def simple_sequence_window_embedding(
    sequence: str, position: int, window: int = 5
) -> SequenceEncoding:
    start = max(0, position - window)
    end = min(len(sequence), position + window + 1)
    snippet = sequence[start:end]
    vector = np.array([ord(char) % 31 for char in snippet], dtype=float)
    if vector.size < (2 * window + 1):
        vector = np.pad(vector, (0, (2 * window + 1) - vector.size))
    return SequenceEncoding(vector=vector, source="window")


@lru_cache(maxsize=1)
def _load_esm2_components(model_name: str = ESM2_MODEL_NAME):
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise RuntimeError("transformers and torch are required for ESM2 encoding")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    forced_device = os.environ.get("COMPLEXVAR_SEQUENCE_DEVICE", "").strip().lower()
    if forced_device:
        model = model.to(forced_device)
    elif torch.cuda.is_available():
        model = model.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return tokenizer, model


def esm2_mutation_embedding(
    sequence: str,
    position: int,
    mutant: str,
    model_name: str = ESM2_MODEL_NAME,
) -> tuple[np.ndarray, np.ndarray]:
    if AutoTokenizer is None or AutoModel is None or torch is None:
        wt = simple_sequence_window_embedding(sequence, position)
        mutant_sequence = sequence[:position] + mutant + sequence[position + 1 :]
        mt = simple_sequence_window_embedding(mutant_sequence, position)
        return wt.vector, mt.vector

    tokenizer, model = _load_esm2_components(model_name)
    mutant_sequence = sequence[:position] + mutant + sequence[position + 1 :]
    with torch.no_grad():
        device = next(model.parameters()).device
        wt_tokens = {
            key: value.to(device)
            for key, value in tokenizer(sequence, return_tensors="pt").items()
        }
        mt_tokens = {
            key: value.to(device)
            for key, value in tokenizer(mutant_sequence, return_tensors="pt").items()
        }
        wt_outputs = model(**wt_tokens).last_hidden_state[0]
        mt_outputs = model(**mt_tokens).last_hidden_state[0]
        token_index = position + 1
        return (
            wt_outputs[token_index].cpu().numpy(),
            mt_outputs[token_index].cpu().numpy(),
        )


def build_sequence_feature_vector(
    sequence: str,
    position: int,
    wildtype: str,
    mutant: str,
    model_name: str = ESM2_MODEL_NAME,
) -> SequenceEncoding:
    wt_embedding, mt_embedding = esm2_mutation_embedding(
        sequence=sequence,
        position=position,
        mutant=mutant,
        model_name=model_name,
    )
    delta = mutation_descriptor(wildtype=wildtype, mutant=mutant)
    vector = np.concatenate(
        [wt_embedding, mt_embedding, np.asarray(list(delta.values()), dtype=float)]
    )
    source = "esm2" if AutoTokenizer is not None and AutoModel is not None else "window"
    return SequenceEncoding(vector=vector, source=source)


if nn is not None:

    class SequenceMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, int, int] = (256, 128, 64),
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LayerNorm(hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.LayerNorm(hidden_dims[2]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.classification_head = nn.Linear(hidden_dims[2], 1)
            self.regression_head = nn.Linear(hidden_dims[2], 1)

        def forward(self, x):
            hidden = self.encoder(x)
            return {
                "classification": self.classification_head(hidden).squeeze(-1),
                "regression": self.regression_head(hidden).squeeze(-1),
            }
