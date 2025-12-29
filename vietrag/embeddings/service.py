from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

from vietrag.config import EmbeddingConfig


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        model_kwargs = {}
        if config.device:
            model_kwargs["devices"] = config.device
        self._model = BGEM3FlagModel(
            config.model_name, normalize_embeddings=True, use_fp16=True, **model_kwargs
        )
        self._batch_size = config.batch_size

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            max_length=8192,
            return_dense=True,
        )["dense_vecs"]

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


__all__ = ["EmbeddingService"]
