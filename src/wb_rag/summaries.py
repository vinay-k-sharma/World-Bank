"""Section summaries: a coarse retrieval layer above the chunks (RAPTOR-style, one level).

Broad questions ("What are the main risks for South Asia?") match a section summary better
than any 200-token fragment. One summary is written per chapter, level-1 section, regional
subsection and box. Summaries are cached by a hash of the section text, so re-ingesting an
unchanged report costs nothing; without an API key an extractive fallback is used (the
section's italic opener if it has one, otherwise its first sentences).
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wb_rag.chunking import split_sentences
from wb_rag.llm import LLMError, api_key, complete
from wb_rag.models import Chunk, Element, Section

log = logging.getLogger(__name__)

SUMMARY_SYSTEM = """You write retrieval summaries of sections of the World Bank's Global Economic Prospects report.
Write 120-180 words of plain prose. Name the section's main claims, the key figures (growth rates, years, percentages) exactly as written, the economies or regions covered, and the policy messages. Do not add anything that is not in the text. No preamble, no bullet points."""

_MAX_INPUT_CHARS = 60_000


def _groups(elements: list[Element], sections: list[Section]) -> list[tuple[str, list[str], list[Element]]]:
    """(node key, section path, elements) for chapters, level-1 sections, level-2 regional sections, boxes."""
    by_node: dict[str, list[Element]] = {}
    for e in elements:
        if e.kind in ("references",):
            continue
        by_node.setdefault(e.node_id, []).append(e)
    out = []
    for s in sections:
        if s.title.lower() == "references":
            continue
        if s.level <= 1 or (s.level == 2 and s.region):
            els = [e for nid, es in by_node.items() if nid == s.node_id or nid.startswith(s.node_id + "/")
                   for e in es if not e.box]
            if sum(len(e.text) for e in els) > 400:
                out.append((s.node_id, s.path, els))
    boxes: dict[str, list[Element]] = {}
    for e in elements:
        if e.box:
            boxes.setdefault(e.node_id, []).append(e)
    for nid, els in boxes.items():
        out.append((nid, els[0].section_path, els))
    return out


def _extractive(els: list[Element]) -> str:
    abstract = next((e.text for e in els if e.kind == "abstract"), None)
    if abstract:
        return abstract
    sents = [s for e in els if e.kind == "text" for s in split_sentences(e.text)]
    return " ".join(sents[:6])


def build_summaries(elements: list[Element], sections: list[Section], cache_file: Path,
                    count_tokens, doc_title: str, model: str, use_llm: bool = True) -> list[Chunk]:
    cache: dict[str, str] = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    groups = _groups(elements, sections)
    jobs = []
    for nid, path, els in groups:
        body = "\n\n".join(e.text for e in els)[:_MAX_INPUT_CHARS]
        key = hashlib.sha256(f"{model}|{body}".encode()).hexdigest()[:20]
        jobs.append((nid, path, els, body, key))

    def run(job) -> tuple[str, str]:
        nid, path, els, body, key = job
        if key in cache:
            return key, cache[key]
        if use_llm and api_key():
            try:
                res = complete(SUMMARY_SYSTEM, f"Section: {' > '.join(path)}\n\n{body}", model=model,
                               temperature=0.0, max_tokens=1024)
                if res.text.strip():
                    return key, res.text.strip()
            except LLMError as e:
                log.warning("Summary failed for %s (%s); using extractive fallback", nid, e)
        return key, _extractive(els)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = dict(pool.map(run, jobs))
    cache.update(results)
    cache_file.write_text(json.dumps(cache, indent=1))

    chunks = []
    for i, (nid, path, els, _, key) in enumerate(jobs):
        text = results[key]
        crumb = " > ".join([doc_title, *path])
        pages = [e.page for e in els]
        chunks.append(Chunk(
            chunk_id=f"S-{i:03d}", kind="summary", parent_id=None, node_id=nid, section_path=path,
            chapter=els[0].chapter, region=els[0].region, box=els[0].box, page_start=min(pages),
            page_end=max(pages), printed_page=els[0].printed_page, text=text,
            embed_text=f"{crumb} (section summary)\n{text}", n_tokens=count_tokens(text), order=10_000 + i))
    return chunks
