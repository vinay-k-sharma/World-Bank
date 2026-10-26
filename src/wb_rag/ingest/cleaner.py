"""Text normalisation and running header/footer removal (shared with the GPO baseline)."""

import re
import unicodedata
from collections import Counter

_PAGE_NUMBER = re.compile(r"^[*_\s]*\d{1,3}[*_\s]*$")
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "—", " ": " "}


def _norm_key(line: str) -> str:
    """Key used to detect repeated headers/footers: ignore spacing, markup and punctuation.

    Digits are kept so ordinary lines that differ only by a figure are never merged; bare page
    numbers are caught separately by _PAGE_NUMBER.
    """
    return re.sub(r"[\W_]+", "", line).lower()


class BoilerplateFilter:
    """Drops short lines that repeat on many pages (running headers/footers) and bare page numbers."""

    def __init__(self, pages_md: list[str], min_share: float = 0.2, max_len: int = 140):
        counts: Counter[str] = Counter()
        for md in pages_md:
            keys = {_norm_key(ln) for ln in md.splitlines() if 0 < len(ln.strip()) <= max_len}
            counts.update(k for k in keys if k)
        threshold = max(3, int(len(pages_md) * min_share))
        self.boilerplate = {k for k, c in counts.items() if c >= threshold}
        self.max_len = max_len

    def is_boilerplate(self, line: str) -> bool:
        s = line.strip()
        if not s or s.startswith("|"):
            return False
        if _PAGE_NUMBER.match(s):
            return True
        return len(s) <= self.max_len and _norm_key(s) in self.boilerplate

    def strip(self, md: str) -> str:
        return "\n".join(ln for ln in md.splitlines() if not self.is_boilerplate(ln))


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for a, b in _QUOTES.items():
        text = text.replace(a, b)
    return text


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_heading(text: str) -> str:
    return clean_text(text.replace("#", "")).strip(" :-")[:120]


def clean_table(table_md: str) -> str:
    rows = []
    for line in table_md.splitlines():
        line = re.sub(r"<sup>[^<]{1,4}</sup>", "", normalize_unicode(line))   # footnote markers
        line = re.sub(r"</?sup>", "", line).replace("<br>", " ")
        line = _MD_EMPHASIS.sub(r"\2", line)
        # "1,193,036 $" -> "$1,193,036" (dollar sign rendered after the value by the layout engine)
        line = re.sub(r"\|\s*([\d,().-]+)\s+\$\s*(?=\|)", r"|$\1", line)
        line = re.sub(r"[ \t]+", " ", line)
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
