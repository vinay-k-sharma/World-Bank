"""Data records shared across the pipeline."""

from typing import Literal

from pydantic import BaseModel

ElementKind = Literal["text", "table", "figure", "abstract", "references"]
ChunkKind = Literal["text", "table", "figure", "abstract", "summary", "references"]


class Section(BaseModel):
    """A node of the document outline (from the report's Contents pages)."""

    node_id: str                 # e.g. "ch2/east-asia-and-pacific/outlook"
    title: str
    level: int                   # 0 = front matter / chapter, 1 = section, 2 = subsection, ...
    path: list[str]              # titles from the chapter down to this node
    chapter: str                 # e.g. "Chapter 2: Regional Outlooks"
    region: str | None = None    # EAP / ECA / LAC / MNA / SAR / SSA for regional sections
    page_start: int              # 1-based physical PDF page
    page_end: int
    printed_page: str | None = None


class Element(BaseModel):
    """One structural block on a page, already assigned to a section."""

    page: int
    printed_page: str | None
    node_id: str
    section_path: list[str]
    chapter: str
    region: str | None
    kind: ElementKind
    label: str | None = None     # "FIGURE 1.1", "TABLE 2.1.1", "BOX 1.2"
    box: str | None = None       # set when the element sits inside a box ("BOX 1.2 Low-income countries...")
    heading: str | None = None   # nearest sub-heading below the TOC node, if any
    text: str


class Chunk(BaseModel):
    chunk_id: str
    kind: ChunkKind
    parent_id: str | None        # parent window this child expands to (None for parents/atomic)
    node_id: str
    section_path: list[str]
    chapter: str
    region: str | None
    label: str | None = None
    box: str | None = None
    page_start: int
    page_end: int
    printed_page: str | None
    text: str
    embed_text: str              # breadcrumb-prefixed text that is embedded / BM25-indexed
    n_tokens: int
    order: int = 0               # position in reading order (for context ordering)

    @property
    def section(self) -> str:
        return " > ".join(self.section_path) or "(front matter)"

    @property
    def pages(self) -> str:
        return f"{self.page_start}" if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    rank: int
    score: float                 # final score (rerank + route boost, or RRF when rerank is off)
    rrf_score: float = 0.0
    rerank_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    routed: bool = False


class ContextBlock(BaseModel):
    """One numbered source passed to the LLM after parent expansion and merging."""

    n: int
    chunk_ids: list[str]         # child hits that led here
    node_id: str
    section: str
    label: str | None
    page_start: int
    page_end: int
    printed_page: str | None
    kind: str
    text: str
    n_tokens: int


class Citation(BaseModel):
    n: int
    section: str
    page: int
    page_end: int
    printed_page: str | None
    label: str | None
    quote: str


class QueryPlan(BaseModel):
    question: str
    regions: list[str] = []
    chapters: list[str] = []
    labels: list[str] = []       # figure/table/box references named in the query
    wants_references: bool = False
    broad: bool = False          # summary / "what are the main..." questions
    sub_queries: list[str] = []


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    context: list[ContextBlock]
    retrieved: list[RetrievedChunk]
    plan: QueryPlan
    stop_reason: str | None = None
    model: str | None = None
    usage: dict = {}
    latency_ms: int = 0
