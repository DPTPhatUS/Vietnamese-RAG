from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vietrag.config import RerankerConfig
from vietrag.types import RetrievalDocument


class BGEReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        if config.device:
            self.model.to(config.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
        top_k: int | None = None,
    ) -> List[RetrievalDocument]:
        if not documents:
            return []
        pairs = [(query, doc.text) for doc in documents]
        scores = []
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            inputs = self.tokenizer(
                [q for q, _ in batch],
                [d for _, d in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            if self.config.device:
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits.squeeze(-1)
            scores.extend(logits.detach().cpu().tolist())
        enriched = [
            RetrievalDocument(text=doc.text, score=float(score), metadata=doc.metadata)
            for doc, score in zip(documents, scores, strict=True)
        ]
        enriched.sort(key=lambda doc: doc.score, reverse=True)
        return enriched if top_k is None else enriched[:top_k]


__all__ = ["BGEReranker"]
