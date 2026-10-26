"""Answer generation with Gemini. Sources are numbered in the prompt and cited with [n] markers,
which are validated and renumbered (in order of first use) against the context blocks."""

import logging
import re

from wb_rag.config import get_settings
from wb_rag.llm import LLMError, complete
from wb_rag.models import Citation, ContextBlock
from wb_rag.prompts import NOT_FOUND, SYSTEM_PROMPT

log = logging.getLogger(__name__)

MARKER_RE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")

GenerationError = LLMError


def _page_label(b: ContextBlock) -> str:
    pages = f"p.{b.page_start}" if b.page_start == b.page_end else f"pp.{b.page_start}-{b.page_end}"
    return pages + (f", printed p.{b.printed_page}" if b.printed_page else "")


def build_prompt(question: str, blocks: list[ContextBlock]) -> str:
    parts = [f"[{b.n}] {b.section} ({_page_label(b)})\n{b.text}" for b in blocks]
    return "Excerpts:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


def _expand(raw: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"\s*,\s*", raw):
        if re.search(r"[\-–]", part):
            a, b = (int(x) for x in re.split(r"\s*[\-–]\s*", part))
            out += list(range(a, b + 1)) if 0 < b - a < 10 else [a, b]
        else:
            out.append(int(part))
    return out


def parse_answer(text: str, blocks: list[ContextBlock]) -> tuple[str, list[Citation]]:
    """Renumber [n] markers to 1..k in order of first use and drop markers that point nowhere."""
    if text.strip().startswith(NOT_FOUND):
        return MARKER_RE.sub("", text).strip(), []
    by_n = {b.n: b for b in blocks}
    citations: list[Citation] = []
    remap: dict[int, int] = {}

    def sub(m: re.Match) -> str:
        out = []
        for n in _expand(m.group(1)):
            b = by_n.get(n)
            if b is None:
                continue
            if n not in remap:
                remap[n] = len(citations) + 1
                citations.append(Citation(n=remap[n], section=b.section, page=b.page_start, page_end=b.page_end,
                                          printed_page=b.printed_page, label=b.label, quote=b.text.strip()[:600]))
            marker = f"[{remap[n]}]"
            if marker not in out:
                out.append(marker)
        return "".join(out)

    return re.sub(r"\s+([.,;])", r"\1", MARKER_RE.sub(sub, text)).strip(), citations


def generate(question: str, blocks: list[ContextBlock]) -> dict:
    s = get_settings()
    if not blocks:
        return {"answer": NOT_FOUND, "citations": [], "stop_reason": "no_context", "model": None, "usage": {}}
    res = complete(SYSTEM_PROMPT, build_prompt(question, blocks), model=s.gemini_model)
    if res.finish == "blocked" or (not res.text and res.finish in ("SAFETY", "PROHIBITED_CONTENT", "RECITATION")):
        return {"answer": "The model declined to answer this question.", "citations": [],
                "stop_reason": "refusal", "model": res.model, "usage": res.usage}
    if res.finish == "MAX_TOKENS":
        log.warning("Answer truncated at max_tokens=%d", s.max_tokens)
    answer, citations = parse_answer(res.text, blocks)
    return {"answer": answer, "citations": citations,
            "stop_reason": {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}.get(res.finish, res.finish.lower()),
            "model": res.model, "usage": res.usage}
