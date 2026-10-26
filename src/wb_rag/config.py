"""Central configuration. Every value can be overridden with a WB_RAG_* env var or .env entry."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
# Export .env (e.g. GEMINI_API_KEY) to the process; real env vars win.
load_dotenv(ROOT / ".env", override=False)

Strategy = Literal["hierarchical", "flat"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WB_RAG_", env_file=ROOT / ".env", extra="ignore")

    # Paths
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    index_root: Path = ROOT / "data"
    eval_file: Path = ROOT / "data" / "eval" / "sample_questions.jsonl"
    reports_dir: Path = ROOT / "eval" / "reports"

    # Document
    doc_id: str = "GEP-June-2025"
    doc_title: str = "Global Economic Prospects, June 2025"

    # Chunking. "hierarchical" = TOC-driven sections, small child chunks linked to parent
    # section windows, atomic figure/table/box chunks, section summaries.
    # "flat" = the baseline: fixed-size chunks with only page metadata.
    chunk_strategy: Strategy = "hierarchical"
    child_size: int = 220          # tokens; what gets embedded and matched
    child_max: int = 320
    child_overlap: int = 40
    parent_size: int = 900         # tokens; what the LLM sees after small-to-big expansion
    parent_max: int = 1200
    flat_size: int = 400           # baseline only
    flat_overlap: int = 60
    section_summaries: bool = True  # LLM summaries per section (cached; extractive fallback without a key)

    # Embeddings
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_cache_dir: Path = ROOT / ".cache" / "fastembed"
    embed_batch_size: int = 32
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Retrieval
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    candidate_k: int = 40          # per retriever, before fusion
    rrf_k: int = 60
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top: int = 30           # fused candidates sent to the cross-encoder
    metadata_routing: bool = True  # region / chapter / figure / table / box hints in the query
    route_boost: float = 0.35      # added to the rerank score of chunks that match the route
    exclude_references: bool = True
    query_rewrite: bool = False    # LLM multi-query decomposition (off by default; see ablation)
    top_n: int = 8                 # child hits kept after reranking

    # Context management
    expand_to_parent: bool = True
    context_budget: int = 6000     # tokens of evidence passed to the LLM
    max_sources: int = 10

    # Generation (Gemini REST)
    gemini_model: str = "gemini-3.5-flash"
    gemini_fallback_models: list[str] = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    summary_model: str = "gemini-3.5-flash-lite"
    judge_model: str = "gemini-3.5-flash"
    gemini_timeout: float = 90.0
    temperature: float = 0.1
    max_tokens: int = 2048

    # Evaluation
    eval_workers: int = 4

    @property
    def index_dir(self) -> Path:
        return self.index_root / f"index_{self.chunk_strategy}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
