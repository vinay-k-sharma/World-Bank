"""Hierarchical, section-aware chunking ("small-to-big").

For every outline node (and every box) the text is packed into **parent windows** of about
`parent_size` tokens that never cross a section boundary. Each parent is split into small
**child chunks** (~`child_size` tokens, sentence-aligned, with overlap, never crossing a
sub-heading). Children are what gets embedded and matched; at answer time a hit is expanded
to its parent so the LLM reads a coherent passage.

Figures and tables are atomic: a figure keeps its title, subtitle, panel titles and notes in
one chunk; a table is split into row groups that repeat the title and header row, with the
whole table as their parent. Italic section openers become `abstract` chunks.

Every chunk's `embed_text` starts with a breadcrumb (report > chapter > section > box >
sub-heading | label) so that "Outlook" in the EAP section and "Outlook" in the SSA section
embed differently.

`flat_chunks` is the naive baseline used in the ablation: fixed-size windows over the page
text with only page metadata.
"""

import re
from collections.abc import Callable, Iterable

from wb_rag.models import Chunk, Element

TokenCounter = Callable[[str], int]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


def _hard_split(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Last resort for a single 'sentence' longer than max_tokens: split on words."""
    out, cur = [], []
    for w in text.split():
        cur.append(w)
        if count(" ".join(cur)) > max_tokens:
            cur.pop()
            out.append(" ".join(cur))
            cur = [w]
    if cur:
        out.append(" ".join(cur))
    return out


def _pieces(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    out = []
    for s in split_sentences(text):
        out += [s] if count(s) <= max_tokens else _hard_split(s, max_tokens, count)
    return out


def split_table(table_md: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Row groups that each repeat the caption lines and header row."""
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    caption = [ln for ln in lines if not ln.startswith("|")]
    rows = [ln for ln in lines if ln.startswith("|")]
    notes_at = next((i for i, ln in enumerate(lines) if not ln.startswith("|") and i > 0
                     and any(l.startswith("|") for l in lines[:i])), None)
    head_caption = caption if notes_at is None else [ln for ln in lines[:notes_at] if not ln.startswith("|")]
    notes = [] if notes_at is None else [ln for ln in lines[notes_at:] if not ln.startswith("|")]
    if count(table_md) <= max_tokens or len(rows) < 3:
        return [table_md]
    n_header = 2 if len(rows) > 1 and re.fullmatch(r"\|?[\s:\-|]+\|?", rows[1]) else 1
    # Keep one extra header row when the first data row is just the year header (GEP tables)
    if len(rows) > n_header and not re.search(r"[A-Za-z]{3}", rows[n_header].split("|")[1] if "|" in rows[n_header][1:] else ""):
        n_header += 1
    header, body = rows[:n_header], rows[n_header:]
    prefix = head_caption + header
    parts, cur = [], []
    for row in body:
        if cur and count("\n".join(prefix + cur + [row])) > max_tokens:
            parts.append("\n".join(prefix + cur))
            cur = []
        cur.append(row)
    if cur:
        parts.append("\n".join(prefix + cur + notes[:2]))
    return parts


class _Builder:
    def __init__(self, doc_title: str, count: TokenCounter, child_size: int, child_max: int,
                 child_overlap: int, parent_size: int, parent_max: int):
        self.doc_title, self.count = doc_title, count
        self.child_size, self.child_max, self.child_overlap = child_size, child_max, child_overlap
        self.parent_size, self.parent_max = parent_size, parent_max
        self.search: list[Chunk] = []     # embedded + BM25-indexed
        self.parents: list[Chunk] = []    # expansion targets only
        self._ids: dict[str, int] = {}
        self._order = 0

    def _id(self, prefix: str, page: int) -> str:
        key = f"{prefix}-p{page:03d}"
        n = self._ids.get(key, 0)
        self._ids[key] = n + 1
        return f"{key}-{n:02d}"

    def breadcrumb(self, e: Element, heading: str | None = None) -> str:
        parts = [self.doc_title, *e.section_path]
        if e.box and e.box not in parts:
            parts.append(e.box)
        if heading:
            parts.append(heading)
        crumb = " > ".join(parts)
        if e.label and e.kind in ("figure", "table"):
            crumb += f" | {e.label}"
        return crumb

    def make(self, kind: str, els: list[Element], text: str, parent_id: str | None,
             heading: str | None = None, store: str = "search", prefix: str = "c") -> Chunk:
        first = els[0]
        self._order += 1
        chunk = Chunk(
            chunk_id=self._id(prefix, first.page),
            kind=kind,  # type: ignore[arg-type]
            parent_id=parent_id,
            node_id=first.node_id,
            section_path=first.section_path,
            chapter=first.chapter,
            region=first.region,
            label=first.label if kind in ("figure", "table") else None,
            box=first.box,
            page_start=min(e.page for e in els),
            page_end=max(e.page for e in els),
            printed_page=first.printed_page,
            text=text.strip(),
            embed_text=f"{self.breadcrumb(first, heading)}\n{text.strip()}",
            n_tokens=self.count(text),
            order=self._order,
        )
        (self.search if store == "search" else self.parents).append(chunk)
        return chunk

    # -- text sections -----------------------------------------------------------------------
    def add_text_group(self, els: list[Element]) -> None:
        """Pack one node's paragraphs into parent windows, then each window into children."""
        kind = "references" if els[0].kind == "references" else "text"
        # (heading, piece, element); paragraphs over parent_max are pre-split at sentences
        units: list[tuple[str | None, str, Element]] = []
        for e in els:
            if self.count(e.text) > self.parent_max:
                units += [(e.heading, p, e) for p in _pieces(e.text, self.child_max, self.count)]
            else:
                units.append((e.heading, e.text, e))

        windows: list[list[tuple[str | None, str, Element]]] = [[]]
        size = 0
        for u in units:
            n = self.count(u[1])
            if windows[-1] and (size + n > self.parent_max or size >= self.parent_size):
                windows.append([])
                size = 0
            windows[-1].append(u)
            size += n

        for win in windows:
            if not win:
                continue
            if kind == "references":
                # Bibliographies are only searchable on request; no parent expansion needed
                self._children(win, kind, parent_id=None)
                continue
            text, last_h = [], None
            for h, piece, _ in win:
                if h and h != last_h:
                    text.append(f"## {h}")
                    last_h = h
                text.append(piece)
            parent_text = "\n\n".join(text)
            if self.count(parent_text) <= self.child_max:
                # Small section: the parent is itself the child
                self.make(kind, [e for _, _, e in win], parent_text, None, heading=win[0][0])
                continue
            parent = self.make(kind, [e for _, _, e in win], parent_text, None,
                               heading=None, store="parent", prefix="P")
            self._children(win, kind, parent.chunk_id)

    def _children(self, win: list[tuple[str | None, str, Element]], kind: str, parent_id: str | None) -> None:
        buf: list[tuple[str, Element]] = []
        heading = win[0][0]
        fresh = False

        def flush(keep_overlap: bool) -> None:
            nonlocal buf, fresh
            if buf and fresh:
                self.make(kind, [e for _, e in buf], " ".join(s for s, _ in buf), parent_id, heading=heading)
            fresh = False
            tail: list[tuple[str, Element]] = []
            if keep_overlap:
                for item in reversed(buf):
                    if self.count(" ".join(s for s, _ in [item, *tail])) > self.child_overlap:
                        break
                    tail.insert(0, item)
            buf = tail

        for h, piece, e in win:
            if h != heading:                      # never cross a sub-heading
                flush(keep_overlap=False)
                heading = h
            for sent in _pieces(piece, self.child_max, self.count):
                buf.append((sent, e))
                fresh = True
                if self.count(" ".join(s for s, _ in buf)) >= self.child_size:
                    flush(keep_overlap=True)
        flush(keep_overlap=False)

    # -- atomic elements ---------------------------------------------------------------------
    def add_figure(self, e: Element) -> None:
        if self.count(e.text) <= self.child_max:
            self.make("figure", [e], e.text, None, heading=e.heading)
            return
        parent = self.make("figure", [e], e.text, None, heading=e.heading, store="parent", prefix="P")
        m = re.search(r"(?m)^Sources?\s*:", e.text)
        head, notes = (e.text[:m.start()], e.text[m.start():]) if m else (e.text, "")
        for part in _pieces_keep_title(head, self.child_max, self.count):
            self.make("figure", [e], part, parent.chunk_id, heading=e.heading)
        if notes:
            title = e.text.splitlines()[0]
            for part in _pieces_keep_title(f"{title} (sources and notes)\n{notes}", self.child_max, self.count):
                self.make("figure", [e], part, parent.chunk_id, heading=e.heading)

    def add_table(self, e: Element) -> None:
        parts = split_table(e.text, self.child_max, self.count)
        if len(parts) == 1:
            self.make("table", [e], e.text, None, heading=e.heading)
            return
        parent_id = None
        if self.count(e.text) <= 2 * self.parent_max:
            parent_id = self.make("table", [e], e.text, None, heading=e.heading, store="parent", prefix="P").chunk_id
        for part in parts:
            self.make("table", [e], part, parent_id, heading=e.heading)

    def add_abstract(self, e: Element) -> None:
        for part in _pieces_keep_title(e.text, self.child_max, self.count):
            self.make("abstract", [e], part, None, heading=None)


def _pieces_keep_title(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    if count(text) <= max_tokens:
        return [text]
    title = text.splitlines()[0] if "\n" in text else ""
    body = text[len(title):] if title else text
    out, cur = [], []
    for s in _pieces(body.replace("\n", " "), max_tokens - count(title) - 2, count):
        if cur and count(" ".join(cur + [s])) + count(title) > max_tokens:
            out.append(f"{title}\n{' '.join(cur)}".strip())
            cur = []
        cur.append(s)
    if cur:
        out.append(f"{title}\n{' '.join(cur)}".strip())
    return out


def hierarchical_chunks(elements: Iterable[Element], count: TokenCounter, doc_title: str,
                        child_size: int = 220, child_max: int = 320, child_overlap: int = 40,
                        parent_size: int = 900, parent_max: int = 1200) -> tuple[list[Chunk], list[Chunk]]:
    """Return (searchable chunks, parent chunks)."""
    b = _Builder(doc_title, count, child_size, child_max, child_overlap, parent_size, parent_max)
    group: list[Element] = []
    key = None

    def flush():
        nonlocal group
        if group:
            b.add_text_group(group)
        group = []

    for e in elements:
        if e.kind == "figure":
            flush(); b.add_figure(e); key = None
            continue
        if e.kind == "table":
            flush(); b.add_table(e); key = None
            continue
        if e.kind == "abstract":
            flush(); b.add_abstract(e); key = None
            continue
        k = (e.node_id, e.box, e.kind)
        if key is not None and k != key:
            flush()
        key = k
        group.append(e)
    flush()
    return b.search, b.parents


def flat_chunks(elements: Iterable[Element], count: TokenCounter, doc_title: str,
                size: int = 400, overlap: int = 60) -> list[Chunk]:
    """Baseline: one running stream of text, cut every ~size tokens; only page metadata."""
    sents: list[tuple[str, Element]] = []
    for e in elements:
        for line in e.text.splitlines():
            sents += [(s, e) for s in _pieces(line, size, count)]
    chunks: list[Chunk] = []
    buf: list[tuple[str, Element]] = []
    n_buf = 0

    def emit():
        first = buf[0][1]
        text = " ".join(s for s, _ in buf)
        chunks.append(Chunk(
            chunk_id=f"f-p{first.page:03d}-{len(chunks):04d}", kind="text", parent_id=None,
            node_id="", section_path=[], chapter="", region=None, page_start=first.page,
            page_end=max(e.page for _, e in buf), printed_page=first.printed_page, text=text,
            embed_text=f"{doc_title} (p.{first.page})\n{text}", n_tokens=count(text), order=len(chunks)))

    for s, e in sents:
        buf.append((s, e))
        n_buf += count(s)
        if n_buf >= size:
            emit()
            tail: list[tuple[str, Element]] = []
            for item in reversed(buf):
                if count(" ".join(x for x, _ in [item, *tail])) > overlap:
                    break
                tail.insert(0, item)
            buf, n_buf = tail, count(" ".join(x for x, _ in tail))
    if buf:
        emit()
    return chunks
