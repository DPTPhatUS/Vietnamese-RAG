from __future__ import annotations

import json
from typing import Dict, List, Optional

from vietrag.config import RouterConfig
from vietrag.embeddings.service import EmbeddingService
from vietrag.llm.qwen import QwenClient
from vietrag.rerank.bge import BGEReranker
from vietrag.retrieval.kg import VietMedKGRetriever
from vietrag.retrieval.raptor import RaptorIndex
from vietrag.types import RetrievalDocument, RetrievalMode


class RoutingAgent:
    def __init__(self, llm: QwenClient, config: RouterConfig):
        self.llm = llm
        self.config = config
        self.system_prompt = (
            "You decide whether a Vietnamese Traditional Medicine query should be answered with "
            "a vector store of book passages (RAPTOR) or a Neo4j knowledge graph (KG)."
        )

    def decide(self, query: str) -> RetrievalMode:
        user_prompt = (
            "Respond with JSON like {\"mode\": \"RAPTOR\"} choosing between RAPTOR or KG.\n"
            f"Query: {query}"
        )
        try:
            raw = self.llm.generate(self.system_prompt, user_prompt)
            candidate = raw
            if "{" in raw and "}" in raw:
                candidate = raw[raw.index("{") : raw.rindex("}") + 1]
            data = json.loads(candidate)
            mode = data.get("mode", "").strip().lower()
            if mode == "kg":
                return RetrievalMode.KNOWLEDGE_GRAPH
            if mode == "raptor":
                return RetrievalMode.RAPTOR
        except Exception:
            pass
        fallback = self.config.default_mode
        if fallback == "kg":
            return RetrievalMode.KNOWLEDGE_GRAPH
        if fallback == "raptor":
            return RetrievalMode.RAPTOR
        return RetrievalMode.HYBRID


class RetrievalRouter:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        raptor_index: Optional[RaptorIndex],
        kg_retriever: Optional[VietMedKGRetriever],
        reranker: Optional[BGEReranker],
        chunk_lookup: Dict[str, dict],
        routing_agent: Optional[RoutingAgent],
    ):
        self.embedding_service = embedding_service
        self.raptor_index = raptor_index
        self.kg_retriever = kg_retriever
        self.reranker = reranker
        self.chunk_lookup = chunk_lookup
        self.routing_agent = routing_agent

    def retrieve(self, query: str, mode: RetrievalMode, top_k: int = 5) -> List[RetrievalDocument]:
        if mode is RetrievalMode.ROUTED:
            if not self.routing_agent:
                raise RuntimeError("Routing agent is not configured")
            mode = self.routing_agent.decide(query)
        query_embedding = self.embedding_service.embed_query(query)
        documents: List[RetrievalDocument] = []
        if mode is RetrievalMode.RAPTOR:
            documents = self._retrieve_raptor(query_embedding, top_k)
        elif mode is RetrievalMode.KNOWLEDGE_GRAPH:
            documents = self._retrieve_kg(query, top_k)
        elif mode is RetrievalMode.HYBRID:
            docs_a = self._retrieve_raptor(query_embedding, top_k)
            docs_b = self._retrieve_kg(query, top_k)
            documents = docs_a + docs_b
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        if self.reranker and documents:
            return self.reranker.rerank(query, documents, top_k=top_k)
        return documents[:top_k]

    def _retrieve_raptor(self, query_embedding, top_k: int) -> List[RetrievalDocument]:
        if not self.raptor_index:
            return []
        return self.raptor_index.search(query_embedding, self.chunk_lookup, top_k=top_k)

    def _retrieve_kg(self, query: str, top_k: int) -> List[RetrievalDocument]:
        if not self.kg_retriever:
            return []
        return self.kg_retriever.search(query, top_k=top_k)


__all__ = ["RetrievalRouter", "RoutingAgent"]
