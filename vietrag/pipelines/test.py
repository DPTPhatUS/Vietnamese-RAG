from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from vietrag.config import AppConfig
from vietrag.pipelines.qa import QAPipeline
from vietrag.types import RetrievalMode

logger = logging.getLogger(__name__)


def run_test_suite(
    config: AppConfig,
    dataset_path: Path,
    output_path: Path,
    *,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    top_k: int = 5,
    limit: Optional[int] = None,
    start: Optional[int] = None,
) -> List[dict]:
    """Run QA pipeline against a dataset of question-answer pairs.

    Args:
        config: Application configuration prepared with credentials and paths.
        dataset_path: JSON file containing a list of objects with at least a "question" field.
        output_path: File to write the serialized evaluation results.
        mode: Retrieval strategy passed to the QA pipeline.
        top_k: Number of contexts to keep after re-ranking.
        limit: Optional cap on number of samples processed (useful for smoke tests).
        start: Optional zero-based offset indicating which sample to evaluate first.

    Returns:
        List of evaluation dictionaries written to ``output_path``.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    total_samples = len(dataset)
    start_index = 0 if start is None else start
    if start_index < 0:
        raise ValueError("Start index cannot be negative")
    if start_index > total_samples:
        raise ValueError(
            f"Start index {start_index} is beyond dataset size {total_samples}"
        )

    dataset = dataset[start_index:]
    if limit is not None:
        dataset = dataset[:limit]

    pipeline = QAPipeline(config)
    results: List[dict] = []
    try:
        for idx, sample in enumerate(dataset, start=start_index + 1):
            question = sample.get("question")
            ground_truth = sample.get("answer")
            if not question:
                logger.warning("Skipping sample %d without question", idx)
                continue

            batch = pipeline.answer(question, mode, top_k=top_k)
            result = {
                "question": question,
                "answer": batch.answer,
                "contexts": [doc.text for doc in batch.documents],
            }
            if ground_truth:
                result["ground_truth"] = ground_truth

            results.append(result)
            logger.info("Processed sample %d/%d", idx, len(dataset))
    finally:
        pipeline.shutdown()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    logger.info("Wrote %d test results to %s", len(results), output_path)
    return results


__all__ = ["run_test_suite"]
