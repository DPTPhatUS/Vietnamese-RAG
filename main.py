from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vietrag.config import AppConfig
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
    parser.error("Unknown command")


if __name__ == "__main__":
    main()
