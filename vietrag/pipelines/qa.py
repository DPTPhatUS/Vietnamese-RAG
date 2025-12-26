from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from vietrag.config import AppConfig
from vietrag.embeddings.service import EmbeddingService
from vietrag.llm.qwen import QwenClient
from vietrag.rerank.bge import BGEReranker
from vietrag.retrieval.kg import VietMedKGRetriever
from vietrag.retrieval.raptor import RaptorIndex
from vietrag.retrieval.router import RetrievalRouter, RoutingAgent
from vietrag.types import RetrievalBatch, RetrievalDocument, RetrievalMode

logger = logging.getLogger(__name__)


def _load_chunk_lookup(chunks_path: Path) -> Dict[str, dict]:
    if not chunks_path.exists():
        raise FileNotFoundError("Chunk cache missing. Run ingestion first.")
    df = pd.read_parquet(chunks_path)
    return {
        str(row["chunk_id"]): {col: row[col] for col in df.columns}
        for _, row in df.iterrows()
    }


class QAPipeline:
    def __init__(self, config: AppConfig):
        self.config = config.prepare()
        self.embedding_service = EmbeddingService(self.config.embeddings)
        self.qwen = QwenClient(self.config.qwen)
        self.reranker = BGEReranker(self.config.reranker)
        self.chunk_lookup = _load_chunk_lookup(self.config.paths.chunks_path)
        self.raptor_index = RaptorIndex.load(self.config.raptor, self.config.paths.raptor_dir)
        self.kg_retriever = VietMedKGRetriever(self.config.neo4j, llm=self.qwen)
        routing_agent = RoutingAgent(self.qwen, self.config.router)
        self.router = RetrievalRouter(
            embedding_service=self.embedding_service,
            raptor_index=self.raptor_index,
            kg_retriever=self.kg_retriever,
            reranker=self.reranker,
            chunk_lookup=self.chunk_lookup,
            routing_agent=routing_agent,
        )
        self.max_context_chars = 8000
        self.system_prompt = (
            "Bạn là một trợ lý tận tâm am hiểu về Y học Cổ truyền Việt Nam. "
            "Chỉ sử dụng thông tin có trong các nguồn đã được truy xuất để trả lời. "
            "Nếu thông tin không có trong nguồn, hãy trả lời rằng bạn không đủ dữ liệu để kết luận. "
            "Không suy đoán và không sử dụng kiến thức bên ngoài. "
            "Trả lời ngắn gọn, rõ ràng và chính xác."
        )

    def answer(self, query: str, mode: RetrievalMode, top_k: int = 5) -> RetrievalBatch:
        documents = self.router.retrieve(query, mode, top_k=top_k)
        response = self._compose_answer(query, documents)
        return RetrievalBatch(query=query, answer=response, documents=documents, mode=mode)

    def _compose_answer(self, query: str, documents: list[RetrievalDocument]) -> str:
        if not documents:
            return "Xin lỗi, tôi không tìm được thông tin phù hợp trong cơ sở tri thức hiện có."
        context = self._build_context(documents)
        if not context:
            return "Xin lỗi, tôi không tìm được thông tin phù hợp trong cơ sở tri thức hiện có."
        user_prompt = (
            f"Nguồn tham chiếu:\n{context}\n\n"
            f"Câu hỏi: {query}\n"
            "Trả lời bằng tiếng Việt."
        )
        answer = self.qwen.generate(self.system_prompt, user_prompt)
        return answer

    def _build_context(self, documents: list[RetrievalDocument]) -> str:
        sections: list[str] = []
        total_chars = 0
        for doc in documents:
            text = (doc.text or "").strip()
            if not text:
                continue
            remaining = self.max_context_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                sections.append(text[:remaining])
                total_chars += remaining
                break
            sections.append(text)
            total_chars += len(text)
        return "\n\n".join(sections)

    def shutdown(self) -> None:
        if self.kg_retriever:
            self.kg_retriever.close()


__all__ = ["QAPipeline"]
