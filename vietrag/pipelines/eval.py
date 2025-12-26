from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import llm_factory
from ragas.metrics.base import Metric
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_entities_recall import ContextEntityRecall
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._noise_sensitivity import NoiseSensitivity

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find results file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Results file must contain a list of QA outputs")
    return payload


def _prepare_samples(
    raw_samples: List[Dict[str, Any]],
    limit: Optional[int],
) -> tuple[List[Dict[str, Any]], int]:
    processed: List[Dict[str, Any]] = []
    skipped = 0
    for idx, sample in enumerate(raw_samples, start=1):
        question = (sample.get("question") or "").strip()
        answer = (sample.get("answer") or "").strip()
        ground_truth = (sample.get("ground_truth") or "").strip()
        contexts = [
            ctx.strip()
            for ctx in sample.get("contexts") or []
            if isinstance(ctx, str) and ctx.strip()
        ]
        if not question or not answer or not ground_truth or not contexts:
            skipped += 1
            logger.debug(
                "Skipping sample %d because of missing fields (question=%s, answer=%s, "
                "ground_truth=%s, contexts=%d)",
                idx,
                bool(question),
                bool(answer),
                bool(ground_truth),
                len(contexts),
            )
            continue
        processed.append(
            {
                "user_input": question,
                "response": answer,
                "reference": ground_truth,
                "retrieved_contexts": contexts,
            }
        )
        if limit is not None and len(processed) >= limit:
            break
    return processed, skipped


def _default_metrics() -> List[Metric]:
    return [
        ContextPrecision(),
        ContextRecall(),
        ContextEntityRecall(),
        NoiseSensitivity(),
        AnswerRelevancy(),
        Faithfulness(),
    ]


def run_ragas_eval(
    results_path: Path,
    *,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    metrics: Optional[Sequence[Metric]] = None,
    ollama_model: str = DEFAULT_LLM_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run Ragas evaluation over serialized QA outputs."""

    raw_results = _load_results(results_path)
    rows, skipped = _prepare_samples(raw_results, limit)
    if not rows:
        raise ValueError("No valid samples left after filtering for evaluation")

    dataset = EvaluationDataset.from_list(rows)
    metric_suite = list(metrics) if metrics else _default_metrics()

    llm = llm_factory(
        ollama_model,
        provider="ollama",
        base_url=ollama_base_url,
        temperature=temperature,
    )

    evaluation = evaluate(
        dataset,
        metrics=metric_suite,
        llm=llm,
    )
    results_df = evaluation.to_pandas()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info("Saved Ragas evaluation CSV to %s", output_path)

    return results_df


__all__ = ["run_ragas_eval"]
