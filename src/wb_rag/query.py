"""Query analysis: turn a question into a retrieval plan using the document's own metadata.

- Figure / table / box / annex references ("Table 2.1.1", "Box 1.2") route to that element.
- Region names, abbreviations and economy names route to the regional section. The economy
  -> region map is read from the report's regional forecast tables at ingest time, so "Nigeria"
  routes to Sub-Saharan Africa without a hand-written list.
- "Chapter 3" routes to the chapter.
- Bibliographies are excluded unless the question asks about references.
- Broad questions (summaries, "main risks", ...) are flagged so section summaries and more
  parent context are used.
- Optional LLM decomposition into sub-queries for multi-part / comparison questions.
"""

import logging
import re

from wb_rag.llm import LLMError, complete
from wb_rag.models import QueryPlan

log = logging.getLogger(__name__)

_LABEL = re.compile(r"\b(figure|fig\.?|table|box|annex)\s+([A-Z]?\d+(?:\.\d+){0,3}(?:\.[A-F])?)", re.I)
_CHAPTER = re.compile(r"\bchapter\s+(\d)\b", re.I)
_REFS = re.compile(r"\b(references?|cited|citations?|bibliograph\w*|literature cited|which (paper|study|studies))\b", re.I)
_BROAD = re.compile(r"\b(summar\w*|overview|main|key|overall|major|broad(ly)?|in general|what are the|"
                    r"policy priorities|recommend\w*|lessons|how does the report|themes?)\b", re.I)

REGION_ALIASES = {
    "EAP": [r"east asia(?: and (?:the )?pacific)?", r"\bEAP\b"],
    "ECA": [r"europe and central asia", r"\bECA\b", r"central asia"],
    "LAC": [r"latin america(?: and the caribbean)?", r"\bLAC\b", r"caribbean"],
    "MNA": [r"middle east(?: and north africa)?", r"\bMNA\b", r"\bMENA\b", r"north africa"],
    "SAR": [r"south asia(?:n)?", r"\bSAR\b"],
    "SSA": [r"sub-saharan africa(?:n)?", r"\bSSA\b", r"sub saharan"],
}


def plan_query(question: str, countries: dict[str, str] | None = None) -> QueryPlan:
    labels = []
    for kind, num in _LABEL.findall(question):
        kind = {"fig": "FIGURE", "fig.": "FIGURE"}.get(kind.lower(), kind.upper())
        labels.append(f"{kind} {num.upper()}")
    regions: list[str] = []
    for code, pats in REGION_ALIASES.items():
        if any(re.search(p, question, re.I if not p.startswith(r"\b") else 0) for p in pats):
            regions.append(code)
    for name, code in (countries or {}).items():
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", question) and code not in regions:
            regions.append(code)
    chapters = [f"Chapter {n}" for n in _CHAPTER.findall(question)]
    return QueryPlan(question=question, regions=regions, chapters=chapters, labels=labels,
                     wants_references=bool(_REFS.search(question)), broad=bool(_BROAD.search(question)))


_DECOMPOSE_SYSTEM = """You turn a question about the World Bank's Global Economic Prospects (June 2025) report into search queries.
If the question has several parts or compares several things (regions, years, topics), return one short self-contained search query per part (at most 4). Otherwise return the question itself, rephrased as a search query. Return JSON only."""

_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
           "required": ["queries"]}


def decompose(question: str) -> list[str]:
    try:
        res = complete(_DECOMPOSE_SYSTEM, question, json_schema=_SCHEMA, temperature=0.0, max_tokens=512,
                       model="gemini-3.5-flash-lite", fallbacks=["gemini-3.1-flash-lite"])
        queries = [q.strip() for q in res.json().get("queries", []) if q.strip()]  # type: ignore[union-attr]
        return queries[:4]
    except (LLMError, ValueError, AttributeError) as e:
        log.warning("Query decomposition failed (%s); using the question as is", e)
        return []
