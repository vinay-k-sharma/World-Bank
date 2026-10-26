"""Context management: turn ranked hits into a compact, numbered, budgeted evidence set.

- Small-to-big: each child hit is replaced by its parent window (the surrounding section
  passage) when that fits the remaining token budget; otherwise the child itself is used.
- Hits that share a parent are merged into one source, so the LLM never reads the same
  passage twice and the source numbering stays short.
- Sources keep relevance order (best first) and carry a breadcrumb title with chapter /
  section / box / label and both physical and printed page numbers.
"""

from wb_rag.config import get_settings
from wb_rag.index import Index, load_index
from wb_rag.models import Chunk, ContextBlock, QueryPlan, RetrievedChunk


def _title(c: Chunk) -> str:
    parts = [c.section]
    if c.box and c.box not in c.section:
        parts.append(c.box)
    title = " > ".join(parts)
    if c.label and c.kind in ("figure", "table"):
        title += f" | {c.label}"
    if c.kind == "summary":
        title += " (section summary)"
    return title


def build_context(retrieved: list[RetrievedChunk], plan: QueryPlan, index: Index | None = None,
                  budget: int | None = None, expand: bool | None = None) -> list[ContextBlock]:
    s = get_settings()
    index = index or load_index()
    budget = budget or s.context_budget
    expand = s.expand_to_parent if expand is None else expand
    if plan.broad:
        budget = int(budget * 1.3)

    blocks: list[ContextBlock] = []
    by_target: dict[str, ContextBlock] = {}
    used = 0
    for r in retrieved:
        c = r.chunk
        parent = index.parents.get(c.parent_id) if (expand and c.parent_id) else None
        target = parent or c
        if target.chunk_id in by_target:
            by_target[target.chunk_id].chunk_ids.append(c.chunk_id)
            continue
        if parent and used + parent.n_tokens > budget:
            target = c                              # parent too big for what is left: use the child
            if target.chunk_id in by_target:
                continue
        if used + target.n_tokens > budget or len(blocks) >= s.max_sources:
            continue
        block = ContextBlock(
            n=len(blocks) + 1, chunk_ids=[c.chunk_id], node_id=target.node_id, section=_title(target),
            label=target.label, page_start=target.page_start, page_end=target.page_end,
            printed_page=target.printed_page, kind=target.kind, text=target.text, n_tokens=target.n_tokens)
        blocks.append(block)
        by_target[target.chunk_id] = block
        used += target.n_tokens
    return blocks
