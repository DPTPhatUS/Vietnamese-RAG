# Vietnamese Traditional Medicine RAG

Retrieval-Augmented Generation (RAG) system for Vietnamese Traditional Medicine Question Answering.

## Prerequisites

- Python 3.11+ and a modern `pip`/`virtualenv` setup.
- A running Neo4j instance (local or remote) that will host VietMedKG.
- [Ollama](https://ollama.com/) with both a chat model and an embedding model pulled locally.

## Configuration

Copy the sample below into a local `.env` (or provide equivalent environment variables) so every CLI command can discover Neo4j and other overrides:

```bash
cat <<'EOF' > .env
RAG_NEO4J__URI=bolt://localhost:7687
RAG_NEO4J__USERNAME=neo4j
RAG_NEO4J__PASSWORD=your-password
RAG_NEO4J__DATABASE=neo4j
EOF
```

Pass `--config path/to/.env` to any `main.py` sub-command if you store overrides outside the project root.

## Setting Up

### VietMedKG Neo4j Graph

1. Clone the [DPTPhatUS/VietMedKG](https://github.com/DPTPhatUS/VietMedKG) repository and install its preprocessing requirements (notably `py2neo` and `pandas`).

2. Edit [preprocessing/kgraph/create_KG.py](https://github.com/DPTPhatUS/VietMedKG/blob/main/preprocessing/kgraph/create_KG.py) so the `Graph(...)` URI and credentials match your Neo4j deployment (e.g. `bolt://localhost:7687`).

3. Run the script to clear the database and ingest every row into Neo4j:

	```bash
	python preprocessing/kgraph/create_KG.py
	```

4. Provide connection details of the graph via the `.env` snippet in the **Configuration** section or your preferred secrets manager.

### Runtime Environment

1. Install dependencies into a clean virtual environment.

	```bash
	pip install -e .
	```

2. Run ingestion to chunk markdown files, embed them, and build the RAPTOR index.

	```bash
	python main.py ingest
	```

3. Ask a question with your preferred retrieval mode: `raptor`, `kg`, `hybrid`, or `routed` (LLM decides).

	```bash
	python main.py qa "Thành phần của bài thuốc Bổ trung ích khí là gì?" --mode hybrid
	```

4. Run `python main.py qa "..." --mode kg` to confirm Neo4j answers are coming through.

## Ragas evaluation

1. Generate QA logs for the benchmark (writes to `artifacts/test_results.json` by default). Use `--start <offset>` and `--limit <count>` to drill into a slice of samples:

	```bash
	python main.py test --dataset data/benchmark/test.json --output artifacts/test_results.json
	```

2. Make sure [Ollama](https://ollama.com/) is running locally with both an evaluator chat model and an embedding model. The defaults are `qwen3:8b` (chat) and `qwen3-embedding:8b` (embeddings); run `ollama pull <model>` for each one before launching the eval.

3. Score the QA logs with Ragas. The evaluator streams every scored sample to a single CSV (defaults to `artifacts/ragas_metrics.csv`, or pass `--output -` to print to stdout) and supports `--start`/`--limit` for partial runs, `--metrics` to select from `context_recall`, `answer_relevancy`, and `faithfulness`, plus knobs for Ollama networking and temperature:

	```bash
	python main.py eval \
	  --results artifacts/test_results.json \
	  --output artifacts/ragas_metrics.csv \
	  --metrics context_recall answer_relevancy faithfulness \
	  --ollama-model qwen3:8b \
	  --ollama-embed-model qwen3-embedding:8b \
	  --ollama-base-url http://localhost:11434 \
	  --temperature 0.0
	```

4. The resulting CSV includes the normalized question, answer, ground truth, flattened contexts, and one column per metric.

### Saved evaluation results

See the [evaluation report](docs/evaluation-results.md) for the results, retrieval coverage, representative retrieval and generation failures, and lessons for improving the pipeline.

Saved Ragas runs cover 400 questions per retrieval mode. The table below reports **mean score (number of valid scores out of 400)**, rounded to four decimal places. Higher scores are better; the highest mean for each metric is bolded.

| Retrieval mode | Answer relevancy | Context recall | Faithfulness |
| --- | --- | --- | --- |
| Hybrid | 0.6501 (399) | **0.6492** (387) | **0.8113** (346) |
| KG | 0.1236 (391) | 0.2118 (92) | 0.7122 (79) |
| RAPTOR | **0.7432** (399) | 0.2655 (377) | 0.6871 (357) |
| Routed | 0.6675 (397) | 0.3480 (340) | 0.6939 (313) |

- **Answer relevancy:** how well the generated answer addresses the question.
- **Context recall:** how much of the reference answer is supported by the retrieved contexts.
- **Faithfulness:** how much of the generated answer is supported by the retrieved contexts.

Hybrid has the highest mean context recall and faithfulness, while RAPTOR has the highest mean answer relevancy. Missing scores are excluded from the means, so the metrics and modes are averaged over different sample subsets. In particular, KG has only 92 valid context recall scores and 79 valid faithfulness scores; comparisons should account for this lower coverage.

The saved artifacts include:

- [Raw QA outputs](artifacts/test/): answers, retrieved contexts, and reference answers for all four modes.
- [Per-question metric scores](artifacts/evaluate/): one CSV per mode and metric.
- [Mean scores](artifacts/evaluate/summary/mean_scores.csv) and [score distributions and valid counts](artifacts/evaluate/summary/metric_summary.csv).

## Using the CLI

Each pipeline action is exposed as a sub-command on `main.py`. Add `--config .env` (or your path) anytime you need custom credentials.

### Command cheat sheet

| Command | Purpose | Helpful flags |
| --- | --- | --- |
| `python main.py ingest` | Chunk markdown, embed passages, and refresh the RAPTOR index. | `--config` for env overrides. |
| `python main.py qa "<question>" --mode hybrid` | Run interactive QA with RAPTOR, KG, hybrid, or routed retrieval. | `--mode`, `--top-k`. |
| `python main.py test --dataset data/benchmark/test.json` | Execute batch QA over the benchmark and capture raw logs. | `--mode`, `--limit`, `--start`, `--output`. |
| `python main.py eval --results artifacts/test_results.json` | Score saved QA logs with Ragas metrics. | `--metrics`, `--ollama-*`, `--start`, `--limit`, `--output`. |

## Data & Artifacts

- `data/benchmark/`: curated QA pairs plus the extracted markdown from the three Viet Medicine books and supporting Neo4j subgraphs.
- `artifacts/test_results.json`: raw QA outputs produced by `main.py test` for the chosen retrieval strategy.
- `artifacts/raptor_index/`: serialized RAPTOR hierarchy (`.json` manifests + `.npy` embeddings) consumed by the retriever.
- `artifacts/evaluate/`: CSV summaries (mean scores, metric breakdowns) derived from Ragas runs.
- `artifacts/ragas_metrics.csv`: metrics dump from the latest evaluation command.
