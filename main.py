from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vietrag.config import AppConfig
from vietrag.pipelines.eval import AVAILABLE_RAGAS_METRICS, run_ragas_eval
from vietrag.pipelines.ingest import run_ingestion
from vietrag.pipelines.qa import QAPipeline
from vietrag.pipelines.test import run_test_suite
from vietrag.types import RetrievalMode


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vietnamese Traditional Medicine RAG system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Chunk markdown files and build RAPTOR index")
    ingest_parser.add_argument("--config", default=None, help="Optional path to .env with overrides")

    qa_parser = subparsers.add_parser("qa", help="Run a single QA query")
    qa_parser.add_argument("question", help="User question in Vietnamese")
    qa_parser.add_argument("--config", default=None, help="Optional path to .env with overrides")
    qa_parser.add_argument(
        "--mode",
        choices=[m.value for m in RetrievalMode],
        default=RetrievalMode.HYBRID.value,
        help="Retrieval strategy",
    )
    qa_parser.add_argument("--top-k", type=int, default=5, help="Number of passages to keep")

    test_parser = subparsers.add_parser("test", help="Run batch QA evaluation on a dataset")
    test_parser.add_argument("--dataset", default="data/benchmark/test.json", help="Path to test dataset JSON")
    test_parser.add_argument(
        "--output",
        default="artifacts/test_results.json",
        help="Where to store the evaluation output",
    )
    test_parser.add_argument(
        "--mode",
        choices=[m.value for m in RetrievalMode],
        default=RetrievalMode.HYBRID.value,
        help="Retrieval strategy",
    )
    test_parser.add_argument("--top-k", type=int, default=5, help="Number of passages to keep")
    test_parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples to evaluate")
    test_parser.add_argument("--config", default=None, help="Optional path to .env with overrides")

    eval_parser = subparsers.add_parser("eval", help="Run Ragas metrics over QA logs")
    eval_parser.add_argument("--results", default="artifacts/test_results.json", help="Path to QA results JSON")
    eval_parser.add_argument(
        "--output",
        default="artifacts/ragas_metrics.csv",
        help="Path to the CSV file that will store all metric rows",
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many samples to score",
    )
    eval_parser.add_argument(
        "--ollama-model",
        default="qwen3:8b",
        help="Ollama chat model used by the evaluator",
    )
    eval_parser.add_argument(
        "--ollama-embed-model",
        default="qwen3-embedding:8b",
        help="Ollama embedding model exposed via the OpenAI embeddings API",
    )
    eval_parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Endpoint for the Ollama server",
    )
    eval_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for the evaluator model",
    )
    eval_parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(AVAILABLE_RAGAS_METRICS),
        default=None,
        metavar="METRIC",
        help="Subset of metrics to compute (choose from: "
        + ", ".join(AVAILABLE_RAGAS_METRICS)
    )
    eval_parser.add_argument("--config", default=None, help="Optional path to .env with overrides")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_kwargs = {"_env_file": args.config} if getattr(args, "config", None) else {}
    config = AppConfig(**config_kwargs)
    if args.command == "ingest":
        run_ingestion(config)
        return
    if args.command == "qa":
        pipeline = QAPipeline(config)
        batch = pipeline.answer(args.question, RetrievalMode(args.mode), top_k=args.top_k)
        print("=== Answer ===")
        print(batch.answer)
        print("\n=== Retrieved Documents ===")
        for idx, doc in enumerate(batch.documents, start=1):
            print(f"[{idx}] score={doc.score:.4f}")
            print(doc.text if len(doc.text) <= 400 else doc.text[:397] + "...")
            print("---")
        pipeline.shutdown()
        return
    if args.command == "test":
        dataset_path = Path(args.dataset)
        output_path = Path(args.output)
        run_test_suite(
            config,
            dataset_path,
            output_path,
            mode=RetrievalMode(args.mode),
            top_k=args.top_k,
            limit=args.limit,
        )
        return
    if args.command == "eval":
        results_path = Path(args.results)
        output_path = None if args.output in (None, "", "-") else Path(args.output)
        run_ragas_eval(
            results_path,
            output_path=output_path,
            limit=args.limit,
            metric_names=args.metrics,
            ollama_model=args.ollama_model,
            ollama_embed_model=args.ollama_embed_model,
            ollama_base_url=args.ollama_base_url,
            temperature=args.temperature,
        )
        return
    parser.error("Unknown command")


if __name__ == "__main__":
    main()
