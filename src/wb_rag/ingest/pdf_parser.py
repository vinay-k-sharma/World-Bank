"""PDF -> structural elements, each assigned to a node of the document outline.

pymupdf4llm gives per-page Markdown with `#` headings and `|` tables. This module walks it in
reading order and keeps a cursor into the outline (structure.py):

- A TOC node starts at its matching heading on its start page, or at the top of the page when
  the heading is not a text heading (e.g. the regional sections of chapter 2).
- `BOX x.y` headings open a box; the rest of the page belongs to it, and every page of a box
  repeats the heading ("(continued)"), so the main-text cursor is untouched.
- `FIGURE x.y` / `TABLE x.y` headings open an atomic figure/table element that collects the
  subtitle, panel titles, the table body and the Sources/Note lines.
- Headings that are not in the TOC become `heading` metadata (sub-subsections).
- Chart-axis "picture text" is dropped; italic chapter/section openers are tagged `abstract`.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from wb_rag.ingest.cleaner import BoilerplateFilter, clean_heading, clean_table, clean_text
from wb_rag.ingest.structure import Outline, slug
from wb_rag.models import Element, Section

log = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_PICTURE = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
_LABELLED = re.compile(r"^(FIGURE|TABLE|BOX|ANNEX)\s+([A-Z]?\d+(?:\.\d+){0,3})\.?\s*(.*)$", re.I)
_PANEL = re.compile(r"^[A-H]\.\s")
_NOTE = re.compile(r"^(Sources?|Notes?)\s*:|^[A-H](?:\.-?[A-H])*\.\s|^\d{1,2}\.\s|^[a-h]\.\s|^\*", re.I)
_SOURCE_NOTE = re.compile(r"^(Sources?|Notes?)\s*:|^[A-H](?:\.-?[A-H])*\.\s", re.I)
_RUNNING = [
    re.compile(r"^\**\d{1,3}\**\s+[A-Z][A-Z0-9 .,&'-]+$"),     # "**4** CHAPTER 1"
    re.compile(r"^[A-Z][A-Z0-9 .,&'-]+\s+\**\d{1,3}\**$"),     # "EAST ASIA AND PACIFIC **57**"
    re.compile(r"^\**\d{1,3}\**$"),
    re.compile(r"^(CHAPTER \d(\.\d)?|STATISTICAL APPENDIX|SELECTED TOPICS)$"),
]
_SKIP_HEADINGS = {"contents", "summary of contents"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_pages(pdf_path: Path, cache_dir: Path) -> list[tuple[int, str]]:
    """[(1-based page, markdown)], cached by file hash because layout parsing takes ~1.5 min."""
    cache = cache_dir / f"pages_{file_sha256(pdf_path)[:16]}.json"
    if cache.exists():
        return [tuple(p) for p in json.loads(cache.read_text())]  # type: ignore[misc]
    with pymupdf.open(pdf_path) as doc:
        pages = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    out = [(p["metadata"].get("page_number") or i + 1, p["text"]) for i, p in enumerate(pages)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def _strip_running(md: str) -> str:
    lines = md.splitlines()
    head = [i for i, ln in enumerate(lines) if ln.strip()][:3]
    keep = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if i in head and any(p.match(s) for p in _RUNNING):
            continue
        keep.append(ln)
    return "\n".join(keep)


def _blocks(markdown: str) -> list[tuple[str, str]]:
    """Split one page into ("heading"|"table"|"text", raw) blocks; picture text is dropped."""
    markdown = _PICTURE.sub("\n", markdown)
    out: list[tuple[str, str]] = []
    para: list[str] = []
    table: list[str] = []

    def end_para():
        if para:
            out.append(("text", " ".join(para)))
            para.clear()

    def end_table():
        if table:
            out.append(("table", "\n".join(table)))
            table.clear()

    for line in markdown.splitlines():
        s = line.strip()
        if s.startswith("|"):
            end_para()
            table.append(s)
            continue
        end_table()
        if not s:
            end_para()
        elif _HEADING.match(s):
            end_para()
            out.append(("heading", s))
        else:
            para.append(s)
    end_para()
    end_table()
    return out


def _heading_text(raw: str) -> str:
    m = _HEADING.match(raw)
    text = re.sub(r"</?sup>", " ", m.group(2) if m else raw)
    return clean_heading(text)


def _is_italic_para(raw: str) -> bool:
    s = raw.strip()
    return s.startswith("_") and s.rstrip().endswith("_") and len(s.split()) > 40


class _Cursor:
    """Tracks the current outline node, box, sub-heading and open figure/table while walking pages."""

    def __init__(self, outline: Outline):
        self.outline = outline
        self.node: Section = outline.sections[0]
        self.heading: str | None = None
        self.box: str | None = None
        self.box_heading: str | None = None
        self.open: dict | None = None         # open figure/table element being collected
        self.elements: list[Element] = []

    # -- emit -------------------------------------------------------------------------------
    def _base(self, page: int) -> dict:
        n = self.node
        if self.box:  # a box is a self-contained unit of its chapter, not of the section it interrupts
            path = [n.chapter, self.box]
            node_id = f"{slug(n.chapter)}/{slug(self.box.split(' ')[0] + '-' + self.box.split(' ')[1])}"
            region, heading = None, self.box_heading
        else:
            path, node_id, region, heading = list(n.path), n.node_id, n.region, self.heading
        return {
            "page": page,
            "printed_page": self.outline.printed.get(page),
            "node_id": node_id,
            "section_path": path,
            "chapter": n.chapter,
            "region": region,
            "box": self.box,
            "heading": heading,
        }

    def emit(self, page: int, kind: str, text: str, label: str | None = None) -> None:
        if not text.strip():
            return
        if self.node.title.lower() == "references" and kind == "text":
            kind = "references"
        self.elements.append(Element(**self._base(page), kind=kind, label=label, text=text))  # type: ignore[arg-type]

    def close_open(self) -> None:
        o = self.open
        if o:
            body = "\n".join(o["parts"]).strip()
            title = f"{o['label']} {o['title']}".strip()
            self.elements.append(Element(**{**o["base"], "kind": o["kind"], "label": o["label"],
                                            "text": f"{title}\n{body}" if body else title}))
        self.open = None

    # -- navigation -------------------------------------------------------------------------
    def enter(self, node: Section) -> None:
        self.close_open()
        self.node, self.heading = node, None


def elements_from_pages(pages: list[tuple[int, str]], outline: Outline) -> list[Element]:
    bp = BoilerplateFilter([md for _, md in pages])
    cur = _Cursor(outline)
    starts: dict[int, list[Section]] = {}
    for s in outline.sections:
        starts.setdefault(s.page_start, []).append(s)

    for page, md in pages:
        if page in outline.toc_pages:
            continue                      # Contents pages only add keyword noise; the outline uses them
        blocks = _blocks(bp.strip(_strip_running(md)))
        headings = {i: _heading_text(raw) for i, (k, raw) in enumerate(blocks) if k == "heading"}

        # Which TOC nodes start on this page, and at which block?
        pending = list(starts.get(page, []))
        at_block: dict[int, Section] = {}
        for node in pending:
            want = norm(node.title)
            for i, h in headings.items():
                hn = norm(h)
                if i not in at_block and hn and (hn == want or (len(want) > 12 and hn.startswith(want[:40]))):
                    at_block[i] = node
                    break
        # Unmatched nodes start at the top of the page, in outline order
        for node in pending:
            if node not in at_block.values():
                cur.enter(node)
        cur.box, cur.box_heading = None, None
        cur.close_open()

        for i, (kind, raw) in enumerate(blocks):
            if i in at_block and not cur.box:
                cur.enter(at_block[i])
                continue
            if kind == "heading":
                title = headings[i]
                if not title or title.lower() in _SKIP_HEADINGS:
                    continue
                m = _LABELLED.match(title)
                if m and m.group(1).upper() == "BOX":
                    cur.close_open()
                    name = re.sub(r"\s*\(continued\)\s*$", "", m.group(3), flags=re.I)
                    cur.box, cur.box_heading = f"BOX {m.group(2)} {name}".strip(), None
                    continue
                if m and m.group(1).upper() in ("FIGURE", "TABLE"):
                    cur.close_open()
                    kind_ = "figure" if m.group(1).upper() == "FIGURE" else "table"
                    cur.open = {"kind": kind_, "label": f"{m.group(1).upper()} {m.group(2)}",
                                "title": m.group(3), "parts": [], "base": cur._base(page),
                                "notes": False, "long": 0}
                    continue
                if _PANEL.match(title) and cur.open and cur.open["kind"] == "figure":
                    cur.open["parts"].append(title)
                    continue
                cur.close_open()
                if cur.box:
                    cur.box_heading = title
                else:
                    cur.heading = title
                continue

            if kind == "table":
                text = clean_table(raw)
                if not text:
                    continue
                # Caption merged into the first table row: "| TABLE 2.6.2Sub-Sahar | an Africa co | ..."
                first_row = re.sub(r"\s*\|\s*", "", text.splitlines()[0])
                cap = re.match(r"^(TABLE|FIGURE)\s*([A-Z]?\d+(?:\.\d+){1,3})(.*)$", first_row)
                if cap and not (cur.open and cur.open["label"].endswith(cap.group(2))):
                    cur.close_open()
                    title = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cap.group(3)).strip()
                    cur.open = {"kind": cap.group(1).lower(), "label": f"{cap.group(1)} {cap.group(2)}",
                                "title": title, "parts": [], "base": cur._base(page), "notes": False, "long": 0}
                    text = "\n".join(text.splitlines()[1:])
                if cur.open and cur.open["kind"] == "table" and not cur.open["notes"]:
                    cur.open["parts"].append(text)
                    continue
                cur.close_open()
                cur.emit(page, "table", text, label=None)
                continue

            # text paragraph
            if raw.count("<br>") >= 2:
                continue                                  # chart axis labels that escaped picture text
            text = clean_text(re.sub(r"</?sup>", "", raw).lstrip("> "))
            if not text:
                continue
            last = cur.elements[-1] if cur.elements else None
            if (not cur.open and last and last.page == page and last.kind in ("figure", "table")
                    and _SOURCE_NOTE.match(text)):
                last.text += "\n" + text                  # Sources/Note lines laid out after the figure
                continue
            cap = _LABELLED.match(text)
            if (cap and cap.group(1).isupper() and cap.group(1) in ("FIGURE", "TABLE")
                    and len(text.split()) <= 30):
                # Caption set as a bold paragraph rather than a heading
                cur.close_open()
                cur.open = {"kind": cap.group(1).lower(), "label": f"{cap.group(1)} {cap.group(2)}",
                            "title": re.sub(r"\d$", "", cap.group(3)).strip(), "parts": [],
                            "base": cur._base(page), "notes": False, "long": 0}
                continue
            o = cur.open
            if o:
                is_note = bool(_NOTE.match(text))
                words = len(text.split())
                if is_note:
                    o["notes"] = o["notes"] or text.lower().startswith(("source", "note"))
                    o["parts"].append(text)
                    continue
                if not o["notes"] and o["kind"] == "figure" and o["long"] == 0:
                    o["long"] += 1                      # the figure's descriptive subtitle
                    o["parts"].append(text)
                    continue
                if not o["notes"] and words <= 25:
                    o["parts"].append(text)              # table caption / units line
                    continue
                cur.close_open()
            if _is_italic_para(raw):
                cur.emit(page, "abstract", text)
            else:
                cur.emit(page, "text", text)
        cur.close_open()
    return cur.elements


def sanity_report(pages: list[tuple[int, str]], elements: list[Element]) -> dict:
    chars = {p: 0 for p, _ in pages}
    for e in elements:
        chars[e.page] += len(e.text)
    kinds: dict[str, int] = {}
    for e in elements:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    report = {
        "pages": len(pages),
        "elements": len(elements),
        "by_kind": kinds,
        "boxes": sorted({e.box.split(" ")[1] for e in elements if e.box}),
        "figures": len({e.label for e in elements if e.kind == "figure"}),
        "tables_labelled": len({e.label for e in elements if e.kind == "table" and e.label}),
        "low_text_pages": [p for p, c in chars.items() if c < 50],
    }
    log.info("Parsed: %s", report)
    return report
