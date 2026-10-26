"""Document outline from the report's own Contents pages.

The heading levels that a layout parser infers from font sizes are unreliable on a 250-page
flagship report (most headings come out as `######`, and the regional sections of chapter 2
have no text heading at all). The Contents pages, however, list every chapter, section and
subsection with its printed page, and their indentation encodes the hierarchy. We read those
lines with their x-positions, rebuild the tree, and map printed pages to physical pages using
the running headers. Boxes, figures, tables and annexes are read from the lists that follow.
"""

import re
from dataclasses import dataclass, field

import pymupdf

from wb_rag.ingest.cleaner import normalize_unicode
from wb_rag.models import Section

_LEADER = re.compile(r"\s*\.{3,}[\s.…]*([0-9]{1,3}|[ivxlc]{1,7})\s*$", re.I)
_CHAPTER_LABEL = re.compile(r"^Chapter\s+(\d+)$", re.I)
_LIST_LABELS = {"annexes": "ANNEX", "boxes": "BOX", "figures": "FIGURE", "tables": "TABLE"}
_ITEM_NO = re.compile(r"^[A-Z]?\d+(?:\.\d+){1,3}$")

REGIONS = {
    "east asia and pacific": "EAP",
    "europe and central asia": "ECA",
    "latin america and the caribbean": "LAC",
    "middle east and north africa": "MNA",
    "south asia": "SAR",
    "sub-saharan africa": "SSA",
}


@dataclass
class _Line:
    page: int
    x: float
    y: float
    text: str
    bold: bool
    italic: bool


@dataclass
class ListItem:
    """An entry of the Boxes / Figures / Tables / Annexes lists."""

    kind: str                    # BOX | FIGURE | TABLE | ANNEX
    number: str                  # "1.1", "B1.2.1", "A3.2.1"
    title: str
    printed_page: str

    @property
    def label(self) -> str:
        return f"{self.kind} {self.number}"


@dataclass
class Outline:
    sections: list[Section]
    items: list[ListItem] = field(default_factory=list)
    printed: dict[int, str] = field(default_factory=dict)   # physical page -> printed page label
    toc_pages: set[int] = field(default_factory=set)        # Contents / list pages (not indexed)

    def physical(self, printed: str) -> int | None:
        for phys, lab in self.printed.items():
            if lab == printed:
                return phys
        return None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def _page_lines(doc: pymupdf.Document, pno: int) -> list[_Line]:
    out = []
    for b in doc[pno].get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            text = normalize_unicode("".join(s["text"] for s in ln["spans"])).strip()
            if not text:
                continue
            font = ln["spans"][0]["font"]
            out.append(_Line(pno + 1, ln["bbox"][0], ln["bbox"][1], text,
                             "Bold" in font, "Ital" in font))
    out.sort(key=lambda l: (round(l.y), l.x))
    return out


def printed_page_map(doc: pymupdf.Document) -> dict[int, str]:
    """Physical page -> printed page number, read from the running header/footer.

    Body pages carry an arabic number next to the running header; front matter carries a roman
    numeral in the footer. Pages without one (chapter openers, figure-only pages) are
    interpolated from their neighbours.
    """
    found: dict[int, int] = {}
    roman: dict[int, str] = {}
    for pno in range(doc.page_count):
        page = doc[pno]
        h = page.rect.height
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                t = "".join(s["text"] for s in ln["spans"]).strip()
                y = ln["bbox"][1]
                if not t:
                    continue
                if y < 70 and re.fullmatch(r"\d{1,3}", t):
                    found.setdefault(pno + 1, int(t))
                elif y > h - 70 and re.fullmatch(r"[ivxlc]{1,7}", t):
                    roman.setdefault(pno + 1, t)
    # Most common offset between physical and printed numbering
    offsets = [p - n for p, n in found.items()]
    offset = max(set(offsets), key=offsets.count) if offsets else 0
    first_body = min(found) if found else doc.page_count + 1
    out: dict[int, str] = {}
    for p in range(1, doc.page_count + 1):
        if p in found:
            out[p] = str(found[p])
        elif p in roman:
            out[p] = roman[p]
        elif p >= first_body - 3 and p - offset >= 1:
            out[p] = str(p - offset)
    return out


def _contents_pages(doc: pymupdf.Document, max_scan: int = 40) -> list[int]:
    """0-based indices of the detailed Contents pages (not the 'Summary of Contents')."""
    start = None
    for pno in range(min(max_scan, doc.page_count)):
        lines = _page_lines(doc, pno)
        heads = [l.text.lower() for l in lines[:4]]
        if start is None and "contents" in heads:
            start = pno
            continue
        if start is not None:
            n_leaders = sum(bool(_LEADER.search(l.text)) for l in lines)
            if n_leaders < 3:
                return list(range(start, pno))
    return [start] if start is not None else []


def _merge_rows(lines: list[_Line]) -> list[list[_Line]]:
    """Group lines that share a baseline (label column + title column)."""
    rows: list[list[_Line]] = []
    for ln in lines:
        if rows and abs(rows[-1][0].y - ln.y) < 3 and rows[-1][0].page == ln.page:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [sorted(r, key=lambda l: l.x) for r in rows]


def parse_outline(doc: pymupdf.Document) -> Outline:
    printed = printed_page_map(doc)
    phys_of = {v: k for k, v in sorted(printed.items(), reverse=True)}

    entries: list[dict] = []          # flat TOC entries in reading order
    items: list[ListItem] = []
    chapter_no: str | None = None
    chapter_title_x: float | None = None
    list_kind: str | None = None
    pending: tuple[_Line, re.Match | None] | None = None   # a title that wraps onto the next line

    for pno in _contents_pages(doc):
        lines = [l for l in _page_lines(doc, pno)
                 if l.text.lower() != "contents" and not re.fullmatch(r"[ivxlc]{1,7}|\d{1,3}", l.text, re.I)
                 or (l.text and _ITEM_NO.match(l.text))]
        for row in _merge_rows(lines):
            left = row[0]
            label = _CHAPTER_LABEL.match(left.text)
            low = left.text.lower()
            body = row
            if label or low in _LIST_LABELS:
                body = row[1:]
                if low in _LIST_LABELS:
                    list_kind = _LIST_LABELS[low]
            if not body:
                continue

            # Boxes / figures / tables / annexes lists: "<number> <title> .... <page>"
            if list_kind and len(body) >= 2 and _ITEM_NO.match(body[0].text):
                m = _LEADER.search(body[-1].text)
                title = " ".join(l.text for l in body[1:])
                if m:
                    items.append(ListItem(list_kind, body[0].text, _LEADER.sub("", title).strip(), m.group(1)))
                continue
            if list_kind:
                continue

            line = body[0]
            text = " ".join(l.text for l in body)
            if pending is not None:
                text = f"{pending[0].text} {text}"
                line, label = pending[0], label or pending[1]
                pending = None
            m = _LEADER.search(text)
            if not m:
                pending = (line, label)          # wrapped title; page number is on the next line
                continue
            title = _LEADER.sub("", text).strip()
            page_label = m.group(1)

            if label and label.group(1) != chapter_no and line.bold:
                chapter_no = label.group(1)
                chapter_title_x = line.x
                entries.append({"title": title, "level": 0, "page": page_label,
                                "chapter": f"Chapter {chapter_no}: {title}"})
            elif line.bold and not line.italic and line.x < 120:
                # Front/back matter at the left margin: Foreword, Statistical Appendix, ...
                chapter_no, chapter_title_x = None, None
                entries.append({"title": title, "level": 0, "page": page_label, "chapter": title})
            elif chapter_title_x is not None:
                level = 1 if line.x < chapter_title_x + 6 else 2
                entries.append({"title": title, "level": level, "page": page_label,
                                "chapter": entries[-1]["chapter"] if entries else title})

    # Annexes are only in the Annexes list; slot them into their chapter as level-1 sections
    for it in (i for i in items if i.kind == "ANNEX"):
        page = phys_of.get(it.printed_page)
        if page is None:
            continue
        pos = next((k for k, e in enumerate(entries) if (phys_of.get(e["page"]) or 0) > page), len(entries))
        chapter = entries[pos - 1]["chapter"] if pos else it.title
        entries.insert(pos, {"title": f"Annex {it.number} {it.title}", "level": 1,
                             "page": it.printed_page, "chapter": chapter})
    sections = _build_sections(entries, phys_of, doc.page_count, printed)
    toc_pages = {p + 1 for p in range(min(40, doc.page_count))
                 if sum(bool(_LEADER.search(l.text)) for l in _page_lines(doc, p)) >= 5}
    return Outline(sections=sections, items=items, printed=printed, toc_pages=toc_pages)


def _build_sections(entries: list[dict], phys_of: dict[str, int], n_pages: int,
                    printed: dict[int, str]) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str, str]] = []           # (level, title, node_id)
    region: str | None = None
    for e in entries:
        phys = phys_of.get(e["page"])
        if phys is None:
            continue
        level = e["level"]
        stack = [s for s in stack if s[0] < level]
        if level == 0:
            region = None
        elif level == 1:
            region = REGIONS.get(e["title"].lower())
        parent_id = stack[-1][2] if stack else ""
        node_id = f"{parent_id}/{slug(e['title'])}" if parent_id else slug(e["chapter"] if level == 0 else e["title"])
        stack.append((level, e["title"], node_id))
        path = [e["chapter"]] + [t for lv, t, _ in stack if lv > 0]
        sections.append(Section(node_id=node_id, title=e["title"], level=level, path=path,
                                chapter=e["chapter"], region=region, page_start=phys,
                                page_end=phys, printed_page=e["page"]))
    # A node ends where the next node at the same or a higher level begins
    for i, s in enumerate(sections):
        nxt = next((t for t in sections[i + 1:] if t.level <= s.level), None)
        end = (nxt.page_start if nxt else n_pages)
        s.page_end = max(s.page_start, end if nxt and nxt.page_start > s.page_start else end)
    # Pages before the first entry (cover, copyright, contents) form a front-matter node
    first = min((s.page_start for s in sections), default=1)
    if first > 1:
        sections.insert(0, Section(node_id="front-matter", title="Front matter", level=0,
                                   path=["Front matter"], chapter="Front matter", page_start=1,
                                   page_end=first - 1, printed_page=printed.get(1)))
    return sections
