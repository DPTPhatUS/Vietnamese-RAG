# Vietnamese Traditional Medicine RAG

This project builds a Retrieval-Augmented Generation (RAG) assistant for Vietnamese Traditional Medicine. It combines:

- **RAPTOR** hierarchical vector retrieval over scanned books.
- **VietMedKG** Neo4j knowledge graph lookups.
- **bge-reranker-v2-m3** for cross-encoder reranking.
- **Qwen3-4B-Instruct-2507** as the answering model.

## Quickstart

1. Install dependencies.

	```bash
	pip install -e .
	```

2. Prepare environment variables (Neo4j URI, credentials, Hugging Face token) in `.env` or pass with `--config`.

3. Run ingestion to chunk markdown files, embed them, and build the RAPTOR index.

	```bash
	python main.py ingest
	```

4. Ask a question with your preferred retrieval mode: `raptor`, `kg`, `hybrid`, or `routed` (LLM decides).

	```bash
	python main.py qa "Thành phần của bài thuốc Bổ trung ích khí là gì?" --mode routed
	```

### Batch QA + Ragas evaluation

1. Generate QA logs for the benchmark split (writes to `artifacts/test_results.json` by default):

	```bash
	python main.py test --dataset data/benchmark/test.json --output artifacts/test_results.json
	```

2. Make sure [Ollama](https://ollama.com/) is running locally with an evaluator chat model **and** an embedding model available to the Ollama LangChain integrations (defaults: `qwen3:8b` for chat, `nomic-embed-text` for embeddings). Pull both via `ollama pull <model>` before running the eval command.

3. Score the QA logs with Ragas, which writes a single CSV containing every metric row (use `--limit` for smoke tests or override the Ollama parameters as needed):

	```bash
	python main.py eval \
	  --results artifacts/test_results.json \
	  --output artifacts/ragas_metrics.csv \
	  --ollama-model qwen3:8b \
	  --ollama-embed-model nomic-embed-text
	```

## Setting Up VietMedKG

1. Clone the [HySonLab/VietMedKG](https://github.com/HySonLab/VietMedKG) repository and install its preprocessing requirements (notably `py2neo` and `pandas`).
2. Download or generate `data/data_translated.csv` from that project, then edit [preprocessing/kgraph/create_KG.py](https://github.com/HySonLab/VietMedKG/blob/main/preprocessing/kgraph/create_KG.py) so the `Graph(...)` URI and credentials match your Neo4j deployment (e.g. `bolt://localhost:7687`).
3. Run the script to clear the database and ingest every row into Neo4j:

	```bash
	python preprocessing/kgraph/create_KG.py
	```

4. Provide connection details via environment variables, for example:

	```bash
	cat <<'EOF' > .env
	RAG_NEO4J__URI=bolt://localhost:7687
	RAG_NEO4J__USERNAME=neo4j
	RAG_NEO4J__PASSWORD=your-password
	RAG_NEO4J__DATABASE=neo4j
	EOF
	```

5. Run `python main.py qa "..." --mode kg` to confirm Neo4j answers are coming through before switching to hybrid/routed modes.

## Project Layout

- `vietrag/config.py` – configuration models (paths, RAPTOR, Neo4j, models).
- `vietrag/data/chunker.py` – parses `data/book*/markdown/book*.md`, splits on `</break>`, and recursively chunks long passages.
- `vietrag/embeddings/service.py` – wraps the sentence-transformer embedder used by RAPTOR.
- `vietrag/retrieval/raptor.py` – RAPTOR tree builder and searcher.
- `vietrag/retrieval/kg.py` – VietMedKG Neo4j retriever.
- `vietrag/rerank/bge.py` – bge reranker module.
- `vietrag/llm/qwen.py` – Qwen3-4B inference helper.
- `vietrag/retrieval/router.py` – orchestrates RAPTOR, KG, hybrid, and routed retrieval modes.
- `vietrag/pipelines/ingest.py` – end-to-end preprocessing pipeline.
- `vietrag/pipelines/eval.py` – wraps Ragas metrics over recorded QA runs via Ollama.
- `vietrag/pipelines/qa.py` – QA orchestration (retrieval + generation).

Artifacts live under `artifacts/` by default (`chunks.parquet`, `raptor_index/`).

## Notes

- The reranker and LLM require a GPU for best performance.
- The router mode uses Qwen to decide whether to query RAPTOR or the knowledge graph based on user intent.
- You can reduce Qwen memory pressure by setting `RAG_QWEN__QUANTIZATION=4bit` (or `8bit`) and, for 8-bit CPU offload, enable `RAG_QWEN__INT8_CPU_OFFLOAD=true`; also override `RAG_QWEN__DEVICE_MAP` when you need a custom placement strategy.
- Qwen defaults follow vendor guidance (temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 0); bump `RAG_QWEN__MAX_NEW_TOKENS` up to 16384 for long-form answers or benchmarks.
- When you need consistent grading, standardize prompts: for math add `Hãy suy luận từng bước và đặt đáp án cuối cùng vào \boxed{}`; for multiple choice append `{"answer": "<letter>"}` instructions so the model returns a single letter.