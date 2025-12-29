from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Callable

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import Metric, MetricType
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_entities_recall import ContextEntityRecall
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._noise_sensitivity import NoiseSensitivity
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

MetricFactory = Callable[[], Metric]

DEFAULT_METRIC_FACTORIES: Dict[str, MetricFactory] = {
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "answer_relevancy": AnswerRelevancy,
    "faithfulness": Faithfulness,
}

AVAILABLE_RAGAS_METRICS = tuple(DEFAULT_METRIC_FACTORIES.keys())


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
    start: Optional[int],
) -> List[Dict[str, Any]]:
    start_index = 0 if start is None else start
    total_samples = len(raw_samples)
    if start_index < 0:
        raise ValueError("Start index cannot be negative")
    if start_index > total_samples:
        raise ValueError(
            f"Start index {start_index} is beyond dataset size {total_samples}"
        )

    processed: List[Dict[str, Any]] = []
    for sample in raw_samples[start_index:]:
        question = (sample.get("question") or "").strip()
        answer = (sample.get("answer") or "").strip()
        ground_truth = (sample.get("ground_truth") or "").strip()
        contexts = [
            ctx.strip()
            for ctx in sample.get("contexts") or []
            if isinstance(ctx, str) and ctx.strip()
        ]
        processed.append(
            {
                "user_input": question or None,
                "response": answer or None,
                "reference": ground_truth or None,
                "retrieved_contexts": contexts,
            }
        )
        if limit is not None and len(processed) >= limit:
            break
    return processed


def _default_metrics(selected_names: Optional[Sequence[str]] = None) -> List[Metric]:
    if selected_names is None:
        metric_names = list(DEFAULT_METRIC_FACTORIES.keys())
    else:
        metric_names = []
        seen: set[str] = set()
        for raw_name in selected_names:
            normalized = (raw_name or "").strip().lower()
            if not normalized:
                continue
            if normalized not in DEFAULT_METRIC_FACTORIES:
                available = ", ".join(DEFAULT_METRIC_FACTORIES.keys())
                raise ValueError(f"Unknown metric '{raw_name}'. Available metrics: {available}")
            if normalized in seen:
                continue
            seen.add(normalized)
            metric_names.append(normalized)
        if not metric_names:
            raise ValueError("No valid metric names provided for evaluation")

    return [DEFAULT_METRIC_FACTORIES[name]() for name in metric_names]


def _metric_required_columns(metric: Metric) -> set[str]:
    required_map = metric.get_required_columns()
    if MetricType.SINGLE_TURN.name in required_map:
        return set(required_map[MetricType.SINGLE_TURN.name])
    required: set[str] = set()
    for cols in required_map.values():
        required.update(cols)
    return required


def _value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return True


def _has_required_fields(sample: Dict[str, Any], required_columns: set[str]) -> bool:
    if not required_columns:
        return True
    for column in required_columns:
        if not _value_is_present(sample.get(column)):
            return False
    return True


def run_ragas_eval(
    results_path: Path,
    *,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    start: Optional[int] = None,
    metrics: Optional[Sequence[Metric]] = None,
    metric_names: Optional[Sequence[str]] = None,
    ollama_model: str = DEFAULT_LLM_MODEL,
    ollama_embed_model: str = DEFAULT_EMBED_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """Run Ragas evaluation over serialized QA outputs."""

    raw_results = _load_results(results_path)
    rows = _prepare_samples(raw_results, limit, start)
    if not rows:
        raise ValueError("No valid samples left after filtering for evaluation")
    metric_suite = list(metrics) if metrics else _default_metrics(metric_names)

    chat_llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=temperature,
    )
    llm = LangchainLLMWrapper(chat_llm)
    embedder = OllamaEmbeddings(model=ollama_embed_model, base_url=ollama_base_url)
    embeddings = LangchainEmbeddingsWrapper(embedder)

    metric_scores: List[Dict[str, Optional[float]]] = [dict() for _ in rows]
    executed_metrics: List[str] = []

    for metric in metric_suite:
        required_columns = _metric_required_columns(metric)
        applicable_rows: List[Dict[str, Any]] = []
        applicable_indices: List[int] = []
        for idx, sample in enumerate(rows):
            if _has_required_fields(sample, required_columns):
                applicable_rows.append(sample)
                applicable_indices.append(idx)

        if not applicable_rows:
            logger.info("Skipping metric '%s' because no samples satisfy its requirements", metric.name)
            continue

        dataset = EvaluationDataset.from_list(applicable_rows)
        evaluation = evaluate(
            dataset,
            metrics=[metric],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(
                max_workers=1,
                timeout=600,
                max_retries=1,
            ),
        )
        executed_metrics.append(metric.name)
        scores = evaluation.scores

        for local_idx, global_idx in enumerate(applicable_indices):
            raw_value = scores[local_idx].get(metric.name)
            sanitized: Optional[float]
            try:
                numeric = float(raw_value)
                sanitized = None if math.isnan(numeric) else numeric
            except (TypeError, ValueError):
                sanitized = None
            metric_scores[global_idx][metric.name] = sanitized

    output_rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(rows):
        contexts = sample.get("retrieved_contexts") or []
        row: Dict[str, Any] = {
            "sample_index": idx,
            "question": sample.get("user_input"),
            "answer": sample.get("response"),
            "ground_truth": sample.get("reference"),
            "num_contexts": len(contexts),
            "contexts": "\n\n---\n\n".join(contexts) if contexts else None,
        }
        for metric_name in executed_metrics:
            row[metric_name] = metric_scores[idx].get(metric_name)
        output_rows.append(row)

    results_df = pd.DataFrame(output_rows)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info("Saved Ragas evaluation CSV to %s", output_path)

    return results_df


__all__ = ["run_ragas_eval", "AVAILABLE_RAGAS_METRICS"]
