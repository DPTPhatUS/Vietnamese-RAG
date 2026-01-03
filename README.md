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

### Ragas evaluation

1. Generate QA logs for the benchmark split (writes to `artifacts/test_results.json` by default). Use `--start <offset>` to resume from a specific sample and `--limit <count>` for quick smoke tests:

	```bash
	python main.py test --dataset data/benchmark/test.json --output artifacts/test_results.json
	```

2. Make sure [Ollama](https://ollama.com/) is running locally with an evaluator chat model **and** an embedding model available to the Ollama LangChain integrations (defaults: `qwen3:8b` for chat, `qwen3-embedding:8b` for embeddings). Pull both via `ollama pull <model>` before running the eval command.

3. Score the QA logs with Ragas, which writes a single CSV containing every metric row (combine `--start` and `--limit` as needed or override the Ollama parameters):

	```bash
	python main.py eval \
	  --results artifacts/test_results.json \
	  --output artifacts/ragas_metrics.csv \
	  --ollama-model qwen3:8b \
	  --ollama-embed-model qwen3-embedding:8b
	```

## Setting Up VietMedKG

1. Clone the [DPTPhatUS/VietMedKG](https://github.com/DPTPhatUS/VietMedKG) repository and install its preprocessing requirements (notably `py2neo` and `pandas`).
2. Edit [preprocessing/kgraph/create_KG.py](https://github.com/DPTPhatUS/VietMedKG/blob/main/preprocessing/kgraph/create_KG.py) so the `Graph(...)` URI and credentials match your Neo4j deployment (e.g. `bolt://localhost:7687`).
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