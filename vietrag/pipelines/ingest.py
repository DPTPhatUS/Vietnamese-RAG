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
    summary_fn: Optional[SummaryFn] = None
    if cfg.raptor.use_llm_summary:
        summary_fn = _build_llm_summarizer(cfg)
    raptor_index = RaptorIndex(
        cfg.raptor,
        cfg.paths.raptor_dir,
        summarizer=summary_fn,
        embedding_service=embedding_service,
    )
    raptor_index.build(chunks)
    logger.info("Stored RAPTOR index at %s", cfg.paths.raptor_dir)


def _build_llm_summarizer(cfg: AppConfig) -> SummaryFn:
    qwen = QwenClient(cfg.qwen)
    system_prompt = "Bạn là trợ lý đọc hiểu tiếng Việt. Hãy tóm tắt ngắn gọn các ý chính từ nhiều đoạn văn."

    def summarize(passages: List[str], level: int) -> str:
        limited = passages[: cfg.raptor.summary_max_segments]
        if not limited:
            return ""
        context = "\n\n".join(f"[{idx + 1}] {text}" for idx, text in enumerate(limited))
        user_prompt = (
            "Tóm tắt các ý chính được nêu trong những đoạn sau bằng tiếng Việt. "
            f"Giới hạn trong {cfg.raptor.summary_target_words} từ và không lặp lại thông tin.\n"
            f"Đoạn tham khảo:\n{context}"
        )
        try:
            return qwen.generate(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("Qwen summary failed at level %s: %s", level, exc)
            return ""

    return summarize


__all__ = ["run_ingestion"]
