from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from vietrag.types import Chunk

BREAK_MARKER = "</break>"


def discover_markdown_files(data_root: Path) -> List[Path]:
    return sorted(data_root.glob("book*/markdown/*.md"))


def _split_recursively(segment: str, max_chars: int) -> Iterable[str]:
    segment = segment.strip()
    if not segment:
        return []
    if len(segment) <= max_chars:
        return [segment]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", segment) if s.strip()]
    if len(sentences) <= 1:
        midpoint = max(1, len(segment) // 2)
        left = segment[:midpoint]
        right = segment[midpoint:]
        return list(_split_recursively(left, max_chars)) + list(_split_recursively(right, max_chars))
    half = len(sentences) // 2
    left = " ".join(sentences[:half]).strip()
    right = " ".join(sentences[half:]).strip()
    chunks: List[str] = []
    if left:
        chunks.extend(_split_recursively(left, max_chars))
    if right:
        chunks.extend(_split_recursively(right, max_chars))
    return chunks


def chunk_markdown_file(
    md_path: Path,
    max_chars: int,
    recursion_threshold: int,
    start_order: int = 0,
) -> List[Chunk]:
    text = md_path.read_text(encoding="utf-8")
    book_id = md_path.parent.parent.name
    base_segments = [seg.strip() for seg in text.split(BREAK_MARKER) if seg.strip()]
    chunks: List[Chunk] = []
    order = start_order
    for seg in base_segments:
        pieces = [seg] if len(seg) <= recursion_threshold else list(_split_recursively(seg, max_chars))
        for piece in pieces:
            chunk_id = f"{book_id}:{md_path.stem}:{order}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    book_id=book_id,
                    source_path=md_path,
                    order=order,
                    metadata={"file": md_path.name, "chunk_order": str(order)},
                )
            )
            order += 1
    return chunks


def chunk_corpus(data_root: Path, max_chars: int, recursion_threshold: int) -> List[Chunk]:
    markdown_files = discover_markdown_files(data_root)
    all_chunks: List[Chunk] = []
    order = 0
    for md_file in tqdm(markdown_files, desc="Chunking markdown"):
        chunks = chunk_markdown_file(md_file, max_chars, recursion_threshold, order)
        order += len(chunks)
        all_chunks.extend(chunks)
    return all_chunks


def chunks_to_dataframe(chunks: List[Chunk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "book_id": chunk.book_id,
                "source_path": str(chunk.source_path),
                "order": chunk.order,
                **chunk.metadata,
            }
            for chunk in chunks
        ]
    )


__all__ = ["chunk_corpus", "chunks_to_dataframe"]
