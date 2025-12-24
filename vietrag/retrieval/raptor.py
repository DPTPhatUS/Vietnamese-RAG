from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans

from vietrag.config import RaptorConfig
from vietrag.types import Chunk, RetrievalDocument

if TYPE_CHECKING:
    from vietrag.embeddings.service import EmbeddingService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RaptorNode:
    node_id: str
    level: int
    text: str
    children: List[str]
    chunk_refs: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)


SummarizerFn = Callable[[List[str], int], str]


class RaptorIndex:
    def __init__(
        self,
        config: RaptorConfig,
        directory: Path,
        summarizer: Optional[SummarizerFn] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ):
        self.config = config
        self.directory = directory
        self.level_nodes: Dict[int, List[RaptorNode]] = {}
        self.level_embeddings: Dict[int, np.ndarray] = {}
        self.leaf_lookup: Dict[str, int] = {}
        self.dimension: int | None = None
        self._summarizer = summarizer
        self._embedding_service = embedding_service

    def build(self, chunks: Sequence[Chunk]) -> None:
        if not len(chunks):
            raise ValueError("Cannot build RAPTOR index without chunks")
        if not self._embedding_service:
            raise RuntimeError("Embedding service is required to build the RAPTOR index")
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_service.embed_texts(texts)
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk embedding generation failed")
        self.dimension = embeddings.shape[1]
        leaves: List[RaptorNode] = []
        for idx, chunk in enumerate(chunks):
            node = RaptorNode(
                node_id=chunk.chunk_id,
                level=0,
                text=chunk.text,
                children=[],
                chunk_refs=[chunk.chunk_id],
                metadata=chunk.metadata,
            )
            leaves.append(node)
            self.leaf_lookup[chunk.chunk_id] = idx
        self.level_nodes[0] = leaves
        self.level_embeddings[0] = embeddings.astype(np.float32)
        current_nodes = leaves
        current_embeddings = embeddings
        for level in range(1, self.config.max_depth + 1):
            if len(current_nodes) <= self.config.cluster_size:
                break
            parents, parent_embeddings = self._cluster_level(current_nodes, current_embeddings, level)
            self.level_nodes[level] = parents
            self.level_embeddings[level] = parent_embeddings
            current_nodes = parents
            current_embeddings = parent_embeddings
        self.save()

    def _cluster_level(
        self,
        nodes: Sequence[RaptorNode],
        embeddings: np.ndarray,
        level: int,
    ) -> tuple[List[RaptorNode], np.ndarray]:
        cluster_count = max(2, math.ceil(len(nodes) / self.config.cluster_size))
        kmeans = KMeans(n_clusters=cluster_count, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)
        parent_nodes: List[RaptorNode] = []
        parent_embeddings = np.zeros((len(clusters), embeddings.shape[1]), dtype=np.float32)
        total_clusters = len(clusters)
        if self._summarizer and total_clusters:
            logger.info("Summarizing level %s (%s clusters)", level, total_clusters)
        for completed, (label, member_indices) in enumerate(clusters.items(), start=1):
            member_nodes = [nodes[i] for i in member_indices]
            member_texts = [node.text for node in member_nodes]
            parent_text = self._summarize_cluster(member_texts, level)
            chunk_refs = sorted({ref for node in member_nodes for ref in node.chunk_refs})
            parent_node = RaptorNode(
                node_id=f"lvl{level}-" + uuid.uuid4().hex[:8],
                level=level,
                text=parent_text,
                children=[node.node_id for node in member_nodes],
                chunk_refs=chunk_refs,
                metadata={"cluster_label": str(label)},
            )
            parent_nodes.append(parent_node)
            parent_embeddings[completed - 1] = self._embed_summary(parent_text)
            if self._summarizer:
                logger.info("Level %s summarization %s/%s", level, completed, total_clusters)
        return parent_nodes, parent_embeddings

    def _embed_summary(self, text: str) -> np.ndarray:
        if not self._embedding_service:
            raise RuntimeError("Embedding service is required to embed summary nodes")
        vector = self._embedding_service.embed_texts([text])[0]
        return vector.astype(np.float32, copy=False)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "levels": list(self.level_nodes.keys()),
            "dimension": self.dimension,
            "leaf_lookup": self.leaf_lookup,
        }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for level, nodes in self.level_nodes.items():
            node_records = [
                {
                    "node_id": node.node_id,
                    "level": node.level,
                    "text": node.text,
                    "children": node.children,
                    "chunk_refs": node.chunk_refs,
                    "metadata": node.metadata,
                }
                for node in nodes
            ]
            (self.directory / f"level_{level}.json").write_text(
                json.dumps(node_records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            np.save(self.directory / f"level_{level}.npy", self.level_embeddings[level])

    @classmethod
    def load(cls, config: RaptorConfig, directory: Path) -> "RaptorIndex":
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("RAPTOR manifest not found. Run ingestion first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = cls(config=config, directory=directory)
        index.dimension = manifest["dimension"]
        index.leaf_lookup = {k: int(v) for k, v in manifest["leaf_lookup"].items()}
        for level in manifest["levels"]:
            nodes_json = json.loads((directory / f"level_{level}.json").read_text(encoding="utf-8"))
            nodes = [
                RaptorNode(
                    node_id=entry["node_id"],
                    level=entry["level"],
                    text=entry["text"],
                    children=entry["children"],
                    chunk_refs=entry["chunk_refs"],
                    metadata=entry.get("metadata", {}),
                )
                for entry in nodes_json
            ]
            index.level_nodes[int(level)] = nodes
            index.level_embeddings[int(level)] = np.load(directory / f"level_{level}.npy")
        return index

    def search(
        self,
        query_embedding: np.ndarray,
        chunk_text_lookup: Dict[str, dict],
        top_k: int = 5,
    ) -> List[RetrievalDocument]:
        if not self.level_embeddings:
            raise RuntimeError("RAPTOR index is not loaded")
        scored_documents: List[RetrievalDocument] = []
        for level, embeddings in self.level_embeddings.items():
            sims = embeddings @ query_embedding
            nodes = self.level_nodes[level]
            for idx, score in enumerate(sims):
                document = self._node_to_document(nodes[idx], float(score), chunk_text_lookup)
                if document:
                    scored_documents.append(document)
        scored_documents.sort(key=lambda doc: doc.score, reverse=True)
        return scored_documents[:top_k]

    def _node_to_document(
        self,
        node: RaptorNode,
        score: float,
        chunk_text_lookup: Dict[str, dict],
    ) -> Optional[RetrievalDocument]:
        if node.level == 0:
            payload = chunk_text_lookup.get(node.node_id)
            if not payload:
                return None
            metadata = {k: str(v) for k, v in payload.items() if k != "text"}
            metadata.update({"node_id": node.node_id, "level": str(node.level)})
            if node.chunk_refs:
                metadata["chunk_refs"] = ",".join(node.chunk_refs)
            metadata["source"] = "chunk"
            return RetrievalDocument(text=payload["text"], score=score, metadata=metadata)
        metadata = {"node_id": node.node_id, "level": str(node.level), "source": "summary"}
        if node.chunk_refs:
            metadata["chunk_refs"] = ",".join(node.chunk_refs)
        if node.metadata:
            metadata.update({k: str(v) for k, v in node.metadata.items()})
        return RetrievalDocument(text=node.text, score=score, metadata=metadata)

    def _summarize_cluster(self, texts: List[str], level: int) -> str:
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return ""
        if self._summarizer:
            try:
                summary = self._summarizer(clean_texts, level)
                if summary:
                    return summary.strip()
            except Exception as exc:
                logger.warning("LLM summarizer failed at level %s: %s", level, exc)
        fallback = "\n".join(clean_texts[:3])
        return fallback


__all__ = ["RaptorIndex"]
