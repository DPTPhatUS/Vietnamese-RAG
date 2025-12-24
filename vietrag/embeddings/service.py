from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from vietrag.config import EmbeddingConfig


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        model_kwargs = {}
        if config.device:
            model_kwargs["device"] = config.device
        self._model = SentenceTransformer(config.model_name, **model_kwargs)
        self._batch_size = config.batch_size

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


__all__ = ["EmbeddingService"]
