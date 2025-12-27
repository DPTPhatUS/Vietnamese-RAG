from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Hugging Face model for dense embeddings",
    )
    batch_size: int = 16
    device: Optional[str] = None


class Neo4jConfig(BaseModel):
    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="neo4j")
    database: str = Field(default="neo4j")


class QwenConfig(BaseModel):
    model_name: str = Field(default="Qwen/Qwen3-4B-Instruct-2507")
    max_new_tokens: int = Field(
        default=2048,
        description="Maximum number of new tokens to generate (set to 16384 for long-form benchmarks)",
    )
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    device: Optional[str] = None
    device_map: Optional[str] = Field(
        default=None,
        description="Override Hugging Face device_map (defaults to auto)",
    )
    quantization: Optional[Literal["4bit", "8bit"]] = Field(
        default=None,
        description="bitsandbytes quantization mode",
    )
    int8_cpu_offload: bool = Field(
        default=False,
        description="Enable fp32 CPU offload when using 8-bit quantization",
    )


class RerankerConfig(BaseModel):
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    batch_size: int = 16
    device: Optional[str] = None


class RaptorConfig(BaseModel):
    max_leaf_chars: int = 1500
    recursion_char_threshold: int = 2500
    cluster_size: int = 20
    max_depth: int = 3
    level_search_k: int = 5
    use_llm_summary: bool = True
    summary_max_segments: int = 8
    summary_target_words: int = 120


class PathConfig(BaseModel):
    data_root: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    chunks_path: Path = Path("artifacts/chunks.parquet")
    raptor_dir: Path = Path("artifacts/raptor_index")

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.raptor_dir.mkdir(parents=True, exist_ok=True)


class RouterConfig(BaseModel):
    routing_threshold: float = 0.2
    default_mode: Literal["raptor", "kg", "hybrid"] = "hybrid"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    paths: PathConfig = PathConfig()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    qwen: QwenConfig = QwenConfig()
    reranker: RerankerConfig = RerankerConfig()
    raptor: RaptorConfig = RaptorConfig()
    router: RouterConfig = RouterConfig()

    def prepare(self) -> "AppConfig":
        self.paths.ensure()
        return self


__all__ = ["AppConfig"]
