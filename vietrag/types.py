from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    book_id: str
    source_path: Path
    order: int
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalDocument:
    text: str
    score: float
    metadata: Dict[str, str]


class RetrievalMode(str, Enum):
    RAPTOR = "raptor"
    KNOWLEDGE_GRAPH = "kg"
    HYBRID = "hybrid"
    ROUTED = "routed"


@dataclass(slots=True)
class RetrievalBatch:
    query: str
    answer: str
    documents: List[RetrievalDocument]
    mode: RetrievalMode


__all__ = [
    "Chunk",
    "RetrievalDocument",
    "RetrievalMode",
    "RetrievalBatch",
]
