from wb_rag.evaluation.run import retrieval_metrics
from wb_rag.generate import parse_answer
from wb_rag.models import Chunk, ContextBlock, RetrievedChunk
from wb_rag.prompts import NOT_FOUND
from wb_rag.query import plan_query


def test_plan_routes_labels_regions_countries():
    p = plan_query("What does Table 2.4.2 show for Nigeria and South Asia?", {"Nigeria": "SSA"})
    assert p.labels == ["TABLE 2.4.2"]
    assert set(p.regions) == {"SSA", "SAR"}
    assert not p.wants_references
    assert plan_query("Which studies are cited on FDI spillovers?").wants_references
    assert plan_query("What are the main risks to the outlook?").broad
    assert plan_query("Explain Box 1.2 and figure 3.5").labels == ["BOX 1.2", "FIGURE 3.5"]
    # Abbreviations are case-sensitive: "sar" in a word must not route
    assert plan_query("necessary reforms").regions == []


def _block(n: int, page: int = 10) -> ContextBlock:
    return ContextBlock(n=n, chunk_ids=[f"c{n}"], node_id="x", section=f"Sec {n}", label=None, page_start=page,
                        page_end=page, printed_page=str(page - 5), kind="text", text=f"text {n}", n_tokens=3)


def test_parse_answer_renumbers_and_drops_invalid():
    blocks = [_block(1), _block(2), _block(3)]
    text, cites = parse_answer("Growth slows [3]. Inflation eases [1, 3][9].", blocks)
    assert text == "Growth slows [1]. Inflation eases [2][1]."
    assert [c.section for c in cites] == ["Sec 3", "Sec 1"]
    text, cites = parse_answer("See [1-2].", blocks)
    assert text == "See [1][2]." and len(cites) == 2
    text, cites = parse_answer(NOT_FOUND + " [2]", blocks)
    assert cites == [] and "[" not in text


def _hit(rank: int, page: int) -> RetrievedChunk:
    c = Chunk(chunk_id=f"c{rank}", kind="text", parent_id=None, node_id="x", section_path=[], chapter="",
              region=None, page_start=page, page_end=page, printed_page=None, text="the quoted evidence here",
              embed_text="", n_tokens=4)
    return RetrievedChunk(chunk=c, rank=rank, score=1.0)


def test_retrieval_metrics_groups_and_evidence():
    q = {"expected_sources": [{"pages": [20]}, {"pages": [30]}], "evidence": [{"page": 20, "quote": "quoted evidence"}]}
    hits = [_hit(1, 5), _hit(2, 20), _hit(3, 30)]
    m = retrieval_metrics(q, hits, [_block(1, 20), _block(2, 30)])
    assert m["first_hit_rank"] == 2 and m["hit@1"] is False and m["hit@3"] is True
    assert m["groups_covered"] and m["context_recall"]
    assert m["evidence_in_context"] is False  # the blocks' text does not contain the quote
