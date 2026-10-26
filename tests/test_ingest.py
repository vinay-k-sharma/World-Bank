from wb_rag.chunking import hierarchical_chunks, split_table
from wb_rag.ingest.pdf_parser import elements_from_pages
from wb_rag.ingest.structure import Outline, _build_sections
from wb_rag.models import Element, Section


def count(text: str) -> int:
    return len(text.split())


def outline() -> Outline:
    entries = [
        {"title": "Regional Outlooks", "level": 0, "page": "51", "chapter": "Chapter 2: Regional Outlooks"},
        {"title": "East Asia and Pacific", "level": 1, "page": "53", "chapter": "Chapter 2: Regional Outlooks"},
        {"title": "Recent developments", "level": 2, "page": "53", "chapter": "Chapter 2: Regional Outlooks"},
        {"title": "Outlook", "level": 2, "page": "54", "chapter": "Chapter 2: Regional Outlooks"},
    ]
    printed = {p: str(p - 22) for p in range(73, 80)}
    phys = {v: k for k, v in printed.items()}
    return Outline(sections=_build_sections(entries, phys, 80, printed), printed=printed)


def test_outline_hierarchy_and_regions():
    o = outline()
    by_title = {s.title: s for s in o.sections}
    eap = by_title["East Asia and Pacific"]
    assert eap.region == "EAP" and eap.level == 1
    assert by_title["Outlook"].path == ["Chapter 2: Regional Outlooks", "East Asia and Pacific", "Outlook"]
    assert by_title["Outlook"].node_id.endswith("east-asia-and-pacific/outlook")
    assert by_title["Recent developments"].page_start == 75 and by_title["Outlook"].page_start == 76


def test_parser_assigns_sections_boxes_figures():
    pages = [
        (75, "Growth in EAP is set to slow.\n\n### **Recent developments**\n\nActivity slowed in early 2025."),
        (76, "More recent developments text here.\n\n### **Outlook**\n\nGrowth is projected at 4.5 percent.\n\n"
             "###### **FIGURE 2.1.3 EAP: Outlook**\n\nGrowth will slow.\n\n###### **A. GDP growth**\n\n"
             "Sources: World Bank.\n\nA. Panel shows growth.\n\nThe main text resumes after the figure with a long paragraph "
             "that clearly does not belong to the figure notes at all and continues for a while."),
        (77, "###### **BOX 1.1 Regional perspectives (continued)**\n\nBox text about regions."),
    ]
    els = elements_from_pages(pages, outline())
    find = lambda prefix: next(e for e in els if e.text.startswith(prefix))  # noqa: E731
    assert find("Activity slowed").section_path[-1] == "Recent developments"
    assert find("More recent developments").section_path[-1] == "Recent developments"  # before the next heading
    assert find("Growth is projected").section_path[-1] == "Outlook"
    assert find("The main text resumes").kind == "text"                               # figure closed
    fig = next(e for e in els if e.kind == "figure")
    assert fig.label == "FIGURE 2.1.3" and "A. GDP growth" in fig.text and "Sources: World Bank." in fig.text
    assert fig.region == "EAP"
    box = next(e for e in els if e.box)
    assert box.box == "BOX 1.1 Regional perspectives" and box.section_path[-1] == "BOX 1.1 Regional perspectives"
    assert box.region is None


def _el(text: str, node: str = "n1", heading: str | None = None, kind: str = "text", label: str | None = None) -> Element:
    return Element(page=10, printed_page="1", node_id=node, section_path=["Chapter 1: X", node], chapter="Chapter 1: X",
                   region=None, kind=kind, label=label, heading=heading, text=text)  # type: ignore[arg-type]


def test_children_link_to_parents_and_respect_sections():
    para = " ".join(f"Sentence number {i} is here." for i in range(60))
    els = [_el(para, "n1", "H1"), _el(para, "n1", "H2"), _el(para, "n2")]
    search, parents = hierarchical_chunks(els, count, "GEP", child_size=60, child_max=90, child_overlap=10,
                                          parent_size=250, parent_max=320)
    pid = {p.chunk_id: p for p in parents}
    assert parents and all(c.parent_id in pid for c in search if c.parent_id)
    for c in search:
        if c.parent_id:
            assert c.node_id == pid[c.parent_id].node_id          # never crosses a section
            assert c.text[:40] in pid[c.parent_id].text.replace("\n\n", " ")
        assert c.embed_text.startswith("GEP > Chapter 1: X > ")
    assert any("> H2\n" in c.embed_text for c in search)         # sub-heading in the breadcrumb


def test_table_split_repeats_header():
    rows = "\n".join(f"| Country{i} | {i}.0 | {i}.5 |" for i in range(40))
    table = "TABLE 2.1.2 Country forecasts\n| | 2024e | 2025f |\n| --- | --- | --- |\n" + rows
    parts = split_table(table, 80, count)
    assert len(parts) > 2
    assert all(p.startswith("TABLE 2.1.2 Country forecasts\n| | 2024e | 2025f |") for p in parts)
