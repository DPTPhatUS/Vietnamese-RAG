from __future__ import annotations

import logging
from typing import Callable, List, Optional

from vietrag.config import AppConfig
from vietrag.data.chunker import chunk_corpus, chunks_to_dataframe
from vietrag.embeddings.service import EmbeddingService
from vietrag.llm.qwen import QwenClient
from vietrag.retrieval.raptor import RaptorIndex

logger = logging.getLogger(__name__)

SummaryFn = Callable[[List[str], int], str]


def run_ingestion(config: AppConfig) -> None:
    cfg = config.prepare()
    logger.info("Starting ingestion pipeline")
    chunks = chunk_corpus(
        cfg.paths.data_root,
        cfg.raptor.max_leaf_chars,
        cfg.raptor.recursion_char_threshold,
    )
    logger.info("Chunked %d passages", len(chunks))
    df = chunks_to_dataframe(chunks)
    df.to_parquet(cfg.paths.chunks_path, index=False)
    embedding_service = EmbeddingService(cfg.embeddings)
    embeddings = embedding_service.embed_texts(df["text"].tolist())
    summary_fn: Optional[SummaryFn] = None
    if cfg.raptor.use_llm_summary:
        summary_fn = _build_llm_summarizer(cfg)
    raptor_index = RaptorIndex(cfg.raptor, cfg.paths.raptor_dir, summarizer=summary_fn)
    raptor_index.build(chunks, embeddings)
    logger.info("Stored RAPTOR index at %s", cfg.paths.raptor_dir)


def _build_llm_summarizer(cfg: AppConfig) -> SummaryFn:
    qwen = QwenClient(cfg.qwen)
    system_prompt = (
        "Bạn là chuyên gia Y học cổ truyền Việt Nam. "
        "Tóm tắt cô đọng các đoạn văn thành 2-3 câu nêu chủ đề chung và điểm khác biệt quan trọng."
    )

    def summarize(passages: List[str], level: int) -> str:
        limited = passages[: cfg.raptor.summary_max_segments]
        if not limited:
            return ""
        context = "\n\n".join(f"[{idx + 1}] {text}" for idx, text in enumerate(limited))
        user_prompt = (
            f"Các đoạn dưới đây thuộc cấp độ cụm {level} trong một cây RAPTOR.\n"
            f"{context}\n\n"
            f"Tổng hợp thành tối đa {cfg.raptor.summary_target_words} từ bằng tiếng Việt, tránh lặp lại nhiều lần."
        )
        try:
            return qwen.generate(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("Qwen summary failed at level %s: %s", level, exc)
            return ""

    return summarize


__all__ = ["run_ingestion"]
