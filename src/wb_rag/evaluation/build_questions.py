"""Draft evaluation questions from the parsed report, then verify every evidence quote.

The report came without a question set, so one is drafted per section with Gemini and then
**checked mechanically**: each question must carry a verbatim evidence quote, and the quote
must be found (after whitespace/punctuation normalisation) on the gold page it names.
Questions that fail the check are dropped. The output is reviewed and edited by hand before
it becomes data/eval/sample_questions.jsonl.

Usage: python -m wb_rag.evaluation.build_questions  (writes data/eval/drafts.jsonl)
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from wb_rag.config import get_settings
from wb_rag.llm import LLMError, complete
from wb_rag.models import Element

SYSTEM = """You write evaluation questions for a retrieval system over the World Bank's Global Economic Prospects (June 2025) report.
You receive one section of the report; every paragraph is prefixed with its page tag like <p57>.
Write {n} questions that a reader of the full report might ask and that this section answers. Requirements:
- Each question must be answerable from a single short passage, unless the category is "multi_part".
- Questions must be self-contained: name the region, economy, chapter topic, or box/figure/table explicitly where needed, so they are unambiguous across the whole 250-page report (e.g. "In South Asia, ..." not "In the region, ...").
- Vary the categories: "numeric" (a specific figure), "factual", "explanatory" (why/how), "table" (only if a table is present), "figure" (only if a FIGURE is present), "multi_part".
- expected_answer: 1-3 sentences, with exact figures.
- evidence_quote: a VERBATIM span of 8-25 words copied exactly from the section text that supports the answer.
- gold_page: the page tag number where the evidence_quote appears."""

SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "category": {"type": "string", "enum": ["numeric", "factual", "explanatory", "table", "figure", "multi_part"]},
            "expected_answer": {"type": "string"},
            "evidence_quote": {"type": "string"},
            "gold_page": {"type": "integer"},
        },
        "required": ["question", "category", "expected_answer", "evidence_quote", "gold_page"],
    }}},
    "required": ["questions"],
}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9%.]+", " ", t.lower()).strip()


def quote_on_page(quote: str, page_text: str) -> bool:
    q, p = _norm(quote), _norm(page_text)
    if q in p:
        return True
    words = q.split()  # tolerate a dropped/hyphenated word at the edges
    return len(words) >= 8 and (" ".join(words[1:]) in p or " ".join(words[:-1]) in p)


def main(per_section: int = 2, seed: int = 7) -> None:
    s = get_settings()
    els = [Element.model_validate_json(l) for l in open(s.processed_dir / "elements.jsonl")]
    page_text: dict[int, str] = {}
    for e in els:
        page_text[e.page] = page_text.get(e.page, "") + "\n" + e.text
    groups: dict[str, list[Element]] = {}
    for e in els:
        if e.kind == "references" or e.chapter in ("Front matter", "Acknowledgments", "Abbreviations", "Selected Topics"):
            continue
        groups.setdefault(e.node_id, []).append(e)
    nodes = [k for k, v in groups.items() if sum(len(e.text) for e in v) > 1500]
    random.Random(seed).shuffle(nodes)

    def run(nid: str) -> list[dict]:
        g = groups[nid]
        body = "\n".join(f"<p{e.page}> {('[' + e.label + '] ') if e.label else ''}{e.text}" for e in g)[:40_000]
        head = " > ".join(g[0].section_path) + (f" > {g[0].box}" if g[0].box and g[0].box not in g[0].section_path else "")
        try:
            res = complete(SYSTEM.format(n=per_section), f"Section: {head}\n\n{body}", model="gemini-3.5-flash",
                           json_schema=SCHEMA, temperature=0.4, max_tokens=4096)
            qs = res.json()["questions"]  # type: ignore[index]
        except (LLMError, ValueError, KeyError) as e:
            print("failed", nid, e)
            return []
        out = []
        for q in qs:
            ok = quote_on_page(q["evidence_quote"], page_text.get(q["gold_page"], ""))
            if not ok:  # the model may name an adjacent page
                for p in (q["gold_page"] - 1, q["gold_page"] + 1):
                    if quote_on_page(q["evidence_quote"], page_text.get(p, "")):
                        q["gold_page"], ok = p, True
                        break
            if ok:
                out.append({**q, "node_id": nid, "section": head})
        return out

    with ThreadPoolExecutor(max_workers=6) as pool:
        drafts = [q for qs in pool.map(run, nodes) for q in qs]
    path = s.eval_file.parent / "drafts.jsonl"
    with open(path, "w") as f:
        for q in drafts:
            f.write(json.dumps(q) + "\n")
    print(f"{len(drafts)} verified drafts from {len(nodes)} sections -> {path}")


if __name__ == "__main__":
    main()
