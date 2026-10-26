# World Bank Economic Report: Advanced Document RAG

Question answering over the World Bank's **Global Economic Prospects, June 2025** (254 pages, 4 chapters,
6 regional sections, 5 boxes, 74 figures, 22 tables, a statistical appendix). Every answer cites the
section and page it came from.

This builds on the earlier GPO baseline RAG (parse → clean → chunk → embed → hybrid retrieval → LLM with
citations) and adds what a long, structured document needs: **outline-driven hierarchical chunking,
rich metadata, metadata routing, cross-encoder reranking and small-to-big context management**. Every
technique is measured against the baseline on the sample questions (see [Results](#results)).

```
PDF ─► Contents pages ─► outline tree (chapter › section › subsection, regions, boxes, annexes, printed↔PDF pages)
    └► pymupdf4llm pages ─► elements assigned to outline nodes (text / table / figure / box / abstract / references)
         ─► hierarchical chunks: child (~220 tok, embedded) ⊂ parent window (~900 tok, read by the LLM)
            + atomic figure/table chunks + section summaries (Gemini) + breadcrumb in every embedding
         ─► bge-base embeddings + BM25 + metadata arrays
query ─► plan (region / economy / chapter / figure / table / box / broad / references)
      ─► hybrid RRF ─► soft metadata routing ─► bge-reranker cross-encoder ─► parent-diverse top-8
      ─► small-to-big expansion + dedupe within a 6k-token budget ─► Gemini 3.5 Flash, [n] citations validated
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
cp .env.example .env            # set GEMINI_API_KEY
uv run wb-rag ingest            # builds both the advanced (hierarchical) and the baseline (flat) index
```

The report is in `data/raw/GEP-June-2025.pdf` (downloaded from
[thedocs.worldbank.org](https://thedocs.worldbank.org/en/doc/8bf0b62ec6bcb886d97295ad930059e9-0050012025/global-economic-prospects-june-2025)).
The first ingest downloads the embedding model (~210 MB) and the reranker (~1.1 GB) into `.cache/fastembed`,
parses the PDF (~1.5 min, cached by file hash), writes ~66 section summaries with Gemini (cached), and embeds
the chunks (cached by text hash, so re-ingesting after a parser change only embeds what changed).

## Usage

```bash
uv run wb-rag serve                                      # chat UI at http://127.0.0.1:8000
uv run wb-rag ask "What is the growth forecast for Nigeria in 2026?"
uv run wb-rag ask "..." --show-context                   # list the context blocks the LLM saw
uv run wb-rag ask "..." --retrieval-only                 # no LLM call: plan + ranked hits with scores
uv run wb-rag ask "..." --rewrite                        # + LLM query decomposition
uv run wb-rag ask "..." --strategy flat                  # the baseline index
uv run wb-rag outline                                    # the recovered document outline
```

API: `POST /ask {question, top_k?, mode?, preset?: advanced|baseline, rewrite?}` returns the answer,
citations (section, PDF page, printed page, label, quote), the context blocks, the ranked hits and the
query plan. Also `GET /outline`, `GET /health`, `POST /ingest`.

## What makes it "advanced"

### 1. Structure from the document itself (`ingest/structure.py`)

Layout parsers infer heading levels from font size, which is unreliable over 250 pages: most GEP headings
come out as `######`, and the six regional sections of chapter 2 have no text heading at all (their titles
are images). The report's **Contents pages**, though, list every chapter, section and subsection with its
printed page, and the **indentation (x-position) encodes the level**. The parser reads those lines with
PyMuPDF, rebuilds a 113-node tree, tags regional sections (EAP, ECA, LAC, MNA, SAR, SSA), adds the annexes
from the Annexes list, and maps printed pages to PDF pages from the running headers (offset 22). The
Boxes / Figures / Tables lists are parsed too.

### 2. Elements assigned to outline nodes (`ingest/pdf_parser.py`)

Walking the pages with a cursor into the outline:

- a node starts at its matching heading on its start page, or at the top of the page when there is none;
- `BOX x.y` headings open a box (a self-contained unit of its chapter, with its own path), and each box page
  repeats the heading, so the main-text cursor is not disturbed;
- `FIGURE` / `TABLE` captions (as headings, bold paragraphs, or merged into a table's first row) open an atomic
  element that collects the subtitle, panel titles, table body, and Sources/Notes;
- headings not in the TOC become sub-heading metadata; chart-axis "picture text" and running headers are
  dropped; italic section openers are tagged `abstract`; bibliographies are tagged `references`; Contents
  pages are not indexed;
- economy → region map read from the regional "country forecasts" tables (146 economies), so "Nigeria" can
  route to Sub-Saharan Africa without a hand-written list.

### 3. Hierarchical chunking (`chunking.py`, `summaries.py`)

| Level | What | Size | Used for |
|---|---|---|---|
| section summary | Gemini summary per chapter / section / regional subsection / box (66) | ~200 tok | matching broad questions |
| parent window | consecutive paragraphs of one outline node, never crossing it | ≤ 1200 tok | what the LLM reads |
| child | sentence-aligned, 40-token overlap, never crossing a sub-heading | ~220 tok | embedding + BM25 matching |
| figure / table / abstract | atomic; tables split into row groups that repeat caption + header | ≤ 320 tok | exact numbers |

Every chunk's embedding text starts with a breadcrumb, e.g.
`Global Economic Prospects, June 2025 > Chapter 2: Regional Outlooks > South Asia > Risks | FIGURE 2.5.3`,
so "Outlook" in the EAP section and "Outlook" in the SSA section embed differently. Chunk metadata:
chapter, section path, node id, region, box, figure/table label, kind, PDF page range, printed page,
reading order, parent id.

### 4. Retrieval (`query.py`, `retrieve.py`)

1. **Query plan**: figure/table/box/annex references, region names and abbreviations, economy names (via the
   ingested map), chapter numbers, "broad" questions, and requests for references.
2. **Hybrid search**: bge-base-en-v1.5 dense + BM25, fused with Reciprocal Rank Fusion; bibliographies
   masked unless asked for.
3. **Soft metadata routing**: the best chunks of the routed subset (e.g. region = SSA, label = TABLE 2.6.2) are
   added to the candidate pool and boosted, rather than hard-filtered: an SSA figure is often quoted in chapter 1.
4. **Cross-encoder reranking** (`BAAI/bge-reranker-base`, local ONNX) of ~30 pooled candidates.
5. **Parent diversity**: keep hits until 8 distinct parents (≤ 2 children each); each named route keeps at
   least one hit, so comparisons see both regions.
6. Optional **query decomposition** with Gemini for multi-part questions (`--rewrite`; measured below).

### 5. Context management (`context.py`)

Small-to-big: each child hit is replaced by its parent window when it fits the remaining budget (6k tokens,
+30% for broad questions), otherwise the child is used. Hits sharing a parent become one source. Sources stay
in relevance order with breadcrumb titles and both PDF and printed pages. Gemini cites `[n]`; markers are
validated, ranges expanded and renumbered in order of first use. If the context doesn't answer, the model
says "The report does not contain this information."

## Evaluation

`data/eval/sample_questions.jsonl`: **52 questions**: numeric, factual, explanatory, table, figure,
comparison, multi-hop, broad and 4 unanswerable. No question set came with the report, so 42 were drafted
per section by Gemini (`evaluation/build_questions.py`) and kept only if their **verbatim evidence quote was
found on the stated gold page**, then reviewed and edited (structure hints like "according to Chapter 3"
removed from most). 10 (comparisons, multi-hop, broad, unanswerable) were written by hand and checked against
the text.

Metrics (`evaluation/run.py`):

- **hit@k / MRR**: a retrieved chunk overlaps a gold page.
- **context_recall**: a gold page is in the context the LLM reads.
- **evidence_in_context**: the verbatim evidence quote is in that context. This is the strictest check that
  chunking and context management kept the passage that actually answers.
- **Answers**: a Gemini judge scores correctness and faithfulness against the excerpts; also citation on a gold
  page and refusal accuracy on unanswerable questions.

```bash
uv run wb-rag ablate              # retrieval-only comparison of all presets (no LLM calls except rewrite)
uv run wb-rag ablate --full --presets baseline_flat,advanced   # + answers and judge
uv run wb-rag eval --preset advanced          # one full run → eval/reports/*.md|json
uv run wb-rag eval --retrieval-only
uv run pytest -q
```

## Results

RESULTS_PLACEHOLDER

## Layout

```
src/wb_rag/
  ingest/structure.py      Contents pages → outline tree, regions, annexes, printed↔PDF page map
  ingest/pdf_parser.py     pages → elements on outline nodes (boxes, figures, tables, abstracts, references)
  ingest/cleaner.py        normalisation, running header/footer removal
  ingest/pipeline.py       ingest orchestration, economy→region map, embedding cache
  chunking.py              hierarchical (child/parent/atomic) and flat baseline chunking
  summaries.py             section summaries (cached), extractive fallback
  embeddings.py            bge embeddings + cross-encoder reranker (fastembed/ONNX)
  index.py                 NumPy vectors + BM25 + metadata arrays + parent store
  query.py                 query plan, LLM decomposition
  retrieve.py              hybrid RRF, routing, rerank, parent diversity
  context.py               small-to-big expansion, dedupe, token budget
  prompts.py generate.py   Gemini answer with validated [n] citations
  llm.py                   Gemini REST client with model fallback
  evaluation/              question builder, harness + presets, judge
  cli.py api.py web/       Typer CLI, FastAPI, chat UI
```

## Notes

- **Page numbers**: citations give the PDF page (1-based) and the printed page; the eval uses PDF pages.
- **Intel Mac**: embeddings and reranking run on ONNX via fastembed; `onnxruntime` is pinned below 1.24.
- **Gemini rate limits**: on 429/503 the client retries, then falls back to `gemini-3.5-flash-lite` /
  `gemini-3.1-flash-lite`; the model actually used is recorded with each answer and judgement.
