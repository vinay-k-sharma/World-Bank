"""Offline ingestion: PDF -> outline + elements -> chunks (+ summaries) -> embeddings + manifest."""

import hashlib
import json
import logging
import re
import time

import numpy as np

import pymupdf

from wb_rag.chunking import flat_chunks, hierarchical_chunks
from wb_rag.config import get_settings
from wb_rag.embeddings import count_tokens, embed_passages
from wb_rag.index import save_index
from wb_rag.ingest.pdf_parser import elements_from_pages, file_sha256, parse_pages, sanity_report
from wb_rag.ingest.structure import parse_outline
from wb_rag.models import Element

log = logging.getLogger(__name__)

_NOT_COUNTRY = re.compile(r"^(memorandum|real gdp|world|region|gdp|of which|percentage|calendar|fiscal|"
                          r"[a-z ]*(average|excluding|exporters|importers|aggregate|economies|countries))", re.I)


def country_regions(elements: list[Element]) -> dict[str, str]:
    """Economy name -> region code, read from the regional 'country forecasts' tables of chapter 2."""
    out: dict[str, str] = {}
    for e in elements:
        if e.kind != "table" or not e.region or not e.label:
            continue
        first = e.text.splitlines()[0].lower()
        if "country forecasts" not in first and "economy forecasts" not in first:
            continue
        for line in e.text.splitlines():
            if not line.startswith("|"):
                continue
            name = line.strip("|").split("|")[0].strip()
            name = re.sub(r"\s*\d+(,\s*\d+)*$", "", name).strip()
            if (re.fullmatch(r"[A-Z][A-Za-z.,' ()&-]{2,40}", name) and not _NOT_COUNTRY.match(name)
                    and "GDP" not in name):
                out.setdefault(name, e.region)
    return out


def _embed_cached(texts: list[str], cache_file) -> np.ndarray:
    """Embed only texts not seen before; re-ingesting after a parser tweak then takes seconds."""
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    cache: dict[str, np.ndarray] = {}
    if cache_file.exists():
        with np.load(cache_file) as z:
            cache = dict(zip(z["keys"].tolist(), z["vectors"]))
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if todo:
        log.info("Embedding %d new chunks (%d cached)", len(todo), len(texts) - len(todo))
        for (k, _), v in zip(todo, embed_passages([t for _, t in todo])):
            cache[k] = v
        np.savez(cache_file, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
    return np.stack([cache[k] for k in keys]).astype(np.float32)


def is_up_to_date() -> bool:
    s = get_settings()
    manifest = s.index_dir / "manifest.json"
    if not manifest.exists():
        return False
    m = json.loads(manifest.read_text())
    params = {"embed_model": s.embed_model, "chunk_strategy": s.chunk_strategy,
              "pdf_sha256": file_sha256(s.raw_dir / f"{s.doc_id}.pdf"), "chunk_params": _chunk_params()}
    return all(m.get(k) == v for k, v in params.items())


def _chunk_params() -> dict:
    s = get_settings()
    if s.chunk_strategy == "flat":
        return {"size": s.flat_size, "overlap": s.flat_overlap}
    return {"child": [s.child_size, s.child_max, s.child_overlap], "parent": [s.parent_size, s.parent_max],
            "summaries": s.section_summaries}


def run_ingest(rebuild: bool = False) -> dict:
    s = get_settings()
    if not rebuild and is_up_to_date():
        log.info("Index is up to date — skipping (use --rebuild to force).")
        return {"skipped": True, "index": str(s.index_dir)}

    t0 = time.time()
    pdf = s.raw_dir / f"{s.doc_id}.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"Missing {pdf}")
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as doc:
        outline = parse_outline(doc)
    pages = parse_pages(pdf, s.processed_dir)
    elements = elements_from_pages(pages, outline)
    report = sanity_report(pages, elements)

    (s.processed_dir / "outline.json").write_text(json.dumps(
        {"sections": [x.model_dump() for x in outline.sections],
         "items": [vars(i) for i in outline.items], "printed": outline.printed}, indent=1))
    with open(s.processed_dir / "elements.jsonl", "w") as f:
        for e in elements:
            f.write(e.model_dump_json() + "\n")

    countries = country_regions(elements)
    if s.chunk_strategy == "flat":
        chunks, parents = flat_chunks(elements, count_tokens, s.doc_title, s.flat_size, s.flat_overlap), []
    else:
        chunks, parents = hierarchical_chunks(elements, count_tokens, s.doc_title, s.child_size, s.child_max,
                                              s.child_overlap, s.parent_size, s.parent_max)
        if s.section_summaries:
            from wb_rag.summaries import build_summaries

            chunks += build_summaries(elements, outline.sections, s.processed_dir / "summaries.json",
                                      count_tokens, s.doc_title, s.summary_model)
    log.info("Built %d chunks (+%d parents); embedding ...", len(chunks), len(parents))
    vectors = _embed_cached([c.embed_text for c in chunks], s.processed_dir / f"emb_cache_{s.embed_model.replace('/', '_')}.npz")
    save_index(chunks, parents, vectors, {
        "pdf_sha256": file_sha256(pdf), "chunk_params": _chunk_params(), "countries": countries,
        "doc_id": s.doc_id, "pages": len(pages)})

    sizes = sorted(c.n_tokens for c in chunks)
    kinds: dict[str, int] = {}
    for c in chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    stats = {
        "strategy": s.chunk_strategy,
        "parse": report,
        "outline_nodes": len(outline.sections),
        "chunks": len(chunks),
        "parents": len(parents),
        "chunks_by_kind": kinds,
        "countries_mapped": len(countries),
        "tokens_p50": sizes[len(sizes) // 2],
        "tokens_max": sizes[-1],
        "seconds": round(time.time() - t0, 1),
    }
    (s.processed_dir / f"ingest_report_{s.chunk_strategy}.json").write_text(json.dumps(stats, indent=2))
    return stats
