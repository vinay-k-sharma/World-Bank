"""Retrieval: hybrid search -> metadata routing -> cross-encoder rerank -> parent-diverse top-n.

1. Dense (bge) and BM25 rankings over the searchable chunks, fused with Reciprocal Rank Fusion.
   Bibliography chunks are masked out unless the question asks for references.
2. Routing: when the plan names a region / chapter / figure / table / box, the best chunks of
   that subset are added to the candidate pool (a *soft* filter: an answer can still come from
   elsewhere, e.g. a regional figure quoted in chapter 1) and get a score boost.
3. A cross-encoder rescores the pooled candidates against the question (and sub-queries).
4. Hits are kept until `top_n` distinct parents are covered, so the context is not eight
   fragments of one passage; each named route keeps at least one hit for comparisons.
"""

import math

import numpy as np

from wb_rag.config import get_settings
from wb_rag.embeddings import embed_query, rerank_scores
from wb_rag.index import Index, bm25_tokenize, load_index
from wb_rag.models import QueryPlan, RetrievedChunk
from wb_rag.query import decompose, plan_query


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _ranked(scores: np.ndarray, mask: np.ndarray, k: int) -> list[int]:
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(-scores[idx], kind="stable")]
    return order[:k].tolist()


def _route_masks(index: Index, plan: QueryPlan) -> list[np.ndarray]:
    masks = []
    for lab in plan.labels:
        m = (index.label == lab) | np.array([bool(b) and b.upper().startswith(lab + " ") for b in index.box])
        if m.any():
            masks.append(m)
    for r in plan.regions:
        m = index.region == r
        if m.any():
            masks.append(m)
    for ch in plan.chapters:
        m = np.array([c.startswith(ch + ":") for c in index.chapter])
        if m.any():
            masks.append(m)
    return masks


def search_candidates(index: Index, query: str, base: np.ndarray, mode: str, k: int,
                      qvec: np.ndarray | None = None) -> tuple[list[int], list[int], dict[int, float]]:
    s = get_settings()
    dense: list[int] = []
    sparse: list[int] = []
    if mode in ("dense", "hybrid"):
        q = qvec if qvec is not None else embed_query(query)
        dense = _ranked(index.vectors @ q, base, k)
    if mode in ("bm25", "hybrid"):
        bm = index.bm25.get_scores(bm25_tokenize(query))
        sparse = [i for i in _ranked(bm, base, k) if bm[i] > 0]
    return dense, sparse, rrf_fuse([r for r in (dense, sparse) if r], k=s.rrf_k)


def retrieve(question: str, top_n: int | None = None, mode: str | None = None,
             plan: QueryPlan | None = None, strategy: str | None = None,
             rerank: bool | None = None, routing: bool | None = None,
             rewrite: bool | None = None) -> tuple[list[RetrievedChunk], QueryPlan]:
    s = get_settings()
    index = load_index(strategy)
    top_n = top_n or s.top_n
    mode = mode or s.retrieval_mode
    rerank = s.rerank if rerank is None else rerank
    routing = s.metadata_routing if routing is None else routing
    rewrite = s.query_rewrite if rewrite is None else rewrite
    plan = plan or plan_query(question, index.countries)
    if rewrite and not plan.sub_queries:
        plan.sub_queries = [q for q in decompose(question) if q.lower() != question.lower()]

    base = np.ones(len(index.chunks), bool)
    if s.exclude_references and not plan.wants_references:
        base &= index.kind != "references"

    queries = [question, *plan.sub_queries]
    pool: dict[int, float] = {}
    dense_rank: dict[int, int] = {}
    bm25_rank: dict[int, int] = {}
    for qi, q in enumerate(queries):
        dense, sparse, fused = search_candidates(index, q, base, mode, s.candidate_k)
        for i, sc in fused.items():
            pool[i] = max(pool.get(i, 0.0), sc)
        if qi == 0:
            dense_rank = {i: r + 1 for r, i in enumerate(dense)}
            bm25_rank = {i: r + 1 for r, i in enumerate(sparse)}

    routes = _route_masks(index, plan) if routing else []
    routed = np.zeros(len(index.chunks), bool)
    for m in routes:
        routed |= m
        _, _, fused = search_candidates(index, question, base & m, mode, 12)
        for i, sc in fused.items():
            pool[i] = max(pool.get(i, 0.0), sc)

    cand = sorted(pool, key=lambda i: -pool[i])
    if not routes:
        cand = cand[: s.rerank_top]
    else:  # keep the routed candidates even if their fused rank is low
        cand = list(dict.fromkeys(cand[: s.rerank_top] + [i for i in cand if routed[i]][:24]))

    scores: dict[int, float] = {}
    rr: dict[int, float] = {}
    if rerank and cand:
        texts = [index.chunks[i].embed_text for i in cand]
        best = np.full(len(cand), -1e9)
        for q in queries:
            best = np.maximum(best, np.array(rerank_scores(q, texts)))
        for i, sc in zip(cand, best):
            rr[i] = float(sc)
            scores[i] = 1 / (1 + math.exp(-sc)) + (s.route_boost if routed[i] else 0.0)
    else:
        top = max(pool.values()) if pool else 1.0
        for i in cand:
            scores[i] = pool[i] / top + (s.route_boost if routed[i] else 0.0)

    order = sorted(cand, key=lambda i: -scores[i])
    chosen = _diverse(index, order, top_n, routes)
    results = [
        RetrievedChunk(chunk=index.chunks[i], rank=r + 1, score=round(scores[i], 5),
                       rrf_score=round(pool.get(i, 0.0), 6), rerank_score=rr.get(i),
                       dense_rank=dense_rank.get(i), bm25_rank=bm25_rank.get(i), routed=bool(routed[i]))
        for r, i in enumerate(chosen)
    ]
    return results, plan


def _diverse(index: Index, order: list[int], top_n: int, routes: list[np.ndarray]) -> list[int]:
    """Take hits in score order until top_n distinct parents; make sure every route is represented."""
    chosen: list[int] = []
    parents: set[str] = set()
    for i in order:
        c = index.chunks[i]
        key = c.parent_id or c.chunk_id
        if key in parents:
            if sum(1 for j in chosen if (index.chunks[j].parent_id or index.chunks[j].chunk_id) == key) >= 2:
                continue  # at most two children of the same parent
        else:
            if len(parents) >= top_n:
                continue
            parents.add(key)
        chosen.append(i)
    for m in routes if len(routes) > 1 else []:
        if not any(m[j] for j in chosen):
            best = next((i for i in order if m[i]), None)
            if best is not None:
                chosen.append(best)
    return chosen
