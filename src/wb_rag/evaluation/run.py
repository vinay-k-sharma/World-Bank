"""Evaluation harness over data/eval/sample_questions.jsonl, with presets for the ablation.

Retrieval metrics (no LLM calls):
- hit@k / MRR: a retrieved chunk overlaps a gold page (first source group).
- groups_covered: every source group (e.g. both regions of a comparison) is hit in the top-n.
- context_recall: a gold page is inside the final context blocks the LLM reads.
- evidence_in_context: the verbatim evidence quote appears in the context text — the
  strictest check that context management kept the passage that actually answers.
- context_tokens: evidence tokens sent to the LLM.

Answer metrics (LLM): judge correctness / faithfulness, citation on a gold page, refusals.
"""

import json
import logging
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from wb_rag.config import get_settings
from wb_rag.context import build_context
from wb_rag.index import load_index
from wb_rag.models import Answer, ContextBlock, RetrievedChunk
from wb_rag.prompts import NOT_FOUND
from wb_rag.retrieve import retrieve

log = logging.getLogger(__name__)

KS = (1, 3, 5, 8)

# Each preset adds one technique on top of the previous one.
PRESETS: dict[str, dict] = {
    "baseline_flat": {"strategy": "flat", "rerank": False, "routing": False, "expand": False, "rewrite": False},
    "structure": {"strategy": "hierarchical", "rerank": False, "routing": False, "expand": False, "rewrite": False},
    "structure+rerank": {"strategy": "hierarchical", "rerank": True, "routing": False, "expand": False, "rewrite": False},
    "+routing": {"strategy": "hierarchical", "rerank": True, "routing": True, "expand": False, "rewrite": False},
    "advanced": {"strategy": "hierarchical", "rerank": True, "routing": True, "expand": True, "rewrite": False},
    "advanced+rewrite": {"strategy": "hierarchical", "rerank": True, "routing": True, "expand": True, "rewrite": True},
}


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or get_settings().eval_file
    return [json.loads(line) for line in open(path) if line.strip()]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9%.]+", " ", t.lower()).strip()


def _overlaps(start: int, end: int, pages: list[int]) -> bool:
    return any(start <= p <= end for p in pages)


def retrieval_metrics(q: dict, retrieved: list[RetrievedChunk], blocks: list[ContextBlock]) -> dict:
    groups = [g["pages"] for g in q["expected_sources"]]
    out: dict = {"context_tokens": sum(b.n_tokens for b in blocks)}
    if not groups:
        return out
    gold = groups[0] if len(groups) == 1 else sorted({p for g in groups for p in g})
    first = next((r.rank for r in retrieved if _overlaps(r.chunk.page_start, r.chunk.page_end, gold)), None)
    for k in KS:
        out[f"hit@{k}"] = bool(first and first <= k)
    out["rr"] = 1.0 / first if first else 0.0
    out["first_hit_rank"] = first
    out["groups_covered"] = all(any(_overlaps(r.chunk.page_start, r.chunk.page_end, g) for r in retrieved)
                                for g in groups)
    out["context_recall"] = all(any(_overlaps(b.page_start, b.page_end, g) for b in blocks) for g in groups)
    quotes = [e["quote"] for e in q.get("evidence", []) if e.get("quote")]
    if quotes:
        ctx = _norm(" ".join(b.text for b in blocks))
        # Tolerate a clipped first/last word, but only for quotes long enough to stay specific
        out["evidence_in_context"] = all(_norm(qt) in ctx or (len(qt.split()) >= 6 and _norm(" ".join(qt.split()[1:-1])) in ctx)
                                         for qt in quotes)
    return out


def declined(answer_text: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in answer_text.lower()


def evaluate_one(q: dict, preset: dict, retrieval_only: bool, use_judge: bool) -> dict:
    row = {"id": q["id"], "category": q["category"], "question": q["question"]}
    opts = {k: preset[k] for k in ("rerank", "routing", "rewrite")}
    strategy = preset["strategy"]
    if retrieval_only:
        hits, plan = retrieve(q["question"], strategy=strategy, **opts)
        blocks = build_context(hits, plan, index=load_index(strategy), expand=preset["expand"])
        row["retrieved"] = [f"p{r.chunk.pages}:{r.chunk.kind}" for r in hits]
        row.update(retrieval_metrics(q, hits, blocks))
        return row

    from wb_rag.generate import generate

    from wb_rag.llm import LLMError

    t0 = time.perf_counter()
    hits, plan = retrieve(q["question"], strategy=strategy, **opts)
    blocks = build_context(hits, plan, index=load_index(strategy), expand=preset["expand"])
    out = None
    for attempt in range(3):
        try:
            out = generate(q["question"], blocks)
            break
        except LLMError as e:
            log.warning("Generation failed for %s (attempt %d): %s", q["id"], attempt + 1, e)
            time.sleep(30 * (attempt + 1))
    if out is None:  # API unavailable: record and leave this question out of the rates
        row.update(retrieval_metrics(q, hits, blocks), error="generation_failed",
                   latency_ms=int((time.perf_counter() - t0) * 1000))
        return row
    ans = Answer(question=q["question"], retrieved=hits, context=blocks, plan=plan,
                 latency_ms=int((time.perf_counter() - t0) * 1000), **out)
    row["retrieved"] = [f"p{r.chunk.pages}:{r.chunk.kind}" for r in hits]
    row.update(retrieval_metrics(q, hits, blocks))
    row.update(answer=ans.answer, latency_ms=ans.latency_ms, model=ans.model, usage=ans.usage,
               citations=[{"n": c.n, "page": c.page, "page_end": c.page_end, "section": c.section} for c in ans.citations])
    row["declined"] = declined(ans.answer)
    if q["category"] == "unanswerable":
        row["correct"] = row["declined"]
        row["faithful"] = True
        return row
    gold = {p for g in q["expected_sources"] for p in g["pages"]}
    row["has_citation"] = bool(ans.citations)
    row["citation_on_gold_page"] = any(_overlaps(c.page, c.page_end, sorted(gold)) for c in ans.citations)
    if use_judge:
        from wb_rag.evaluation.judge import judge

        try:
            v = judge(ans, q["expected_answer"])
            row.update(judgement=v.correctness, judge_score=v.score, faithful=v.faithful,
                       unsupported_claims=v.unsupported_claims, judge_reasoning=v.reasoning, judge_model=v.model)
            row["correct"] = v.correctness == "correct"
            row["correct_or_partial"] = v.correctness != "incorrect"
        except Exception as e:  # noqa: BLE001 - a failed judgement should not sink the run
            log.warning("Judge failed for %s: %s", q["id"], e)
    return row


def failure_tag(row: dict) -> str | None:
    if row["category"] == "unanswerable":
        return None if row.get("correct") else "bad_refusal"
    if row.get("correct"):
        return None
    if row.get("context_recall") is False:
        return "retrieval_miss"
    if row.get("declined"):
        return "false_refusal"
    if row.get("faithful") is False:
        return "unfaithful"
    if "correct" in row:
        return "wrong_or_incomplete"
    return None


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(map(bool, vals)) / len(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    sm: dict = {f"hit@{k}": _rate(answerable, f"hit@{k}") for k in KS}
    rr = [r["rr"] for r in answerable if "rr" in r]
    sm["mrr"] = round(statistics.mean(rr), 3) if rr else None
    multi = [r for r in answerable if r["category"] in ("comparison", "multi_hop", "broad")]
    sm["groups_covered_multi"] = _rate(multi, "groups_covered")
    sm["context_recall"] = _rate(answerable, "context_recall")
    sm["evidence_in_context"] = _rate(answerable, "evidence_in_context")
    sm["avg_context_tokens"] = round(statistics.mean(r["context_tokens"] for r in rows))
    if any("answer" in r for r in rows):
        answerable = [r for r in answerable if not r.get("error")]
        lat = sorted(r["latency_ms"] for r in rows)
        sm.update(
            correct=_rate(answerable, "correct"),
            correct_or_partial=_rate(answerable, "correct_or_partial"),
            faithful=_rate(answerable, "faithful"),
            citation_on_gold_page=_rate(answerable, "citation_on_gold_page"),
            refusal_accuracy=_rate([r for r in rows if r["category"] == "unanswerable" and not r.get("error")], "correct"),
            false_refusals=sum(bool(r.get("declined")) for r in answerable),
            latency_p50_s=round(lat[len(lat) // 2] / 1000, 1),
            errors=sum(bool(r.get("error")) for r in rows),
        )
        cats: dict[str, dict] = {}
        for cat in sorted({r["category"] for r in rows}):
            rs = [r for r in rows if r["category"] == cat]
            cats[cat] = {"n": len(rs), "correct": _rate(rs, "correct"), "context_recall": _rate(rs, "context_recall")}
        sm["by_category"] = cats
        tags: dict[str, int] = {}
        for r in rows:
            if t := r.get("failure"):
                tags[t] = tags.get(t, 0) + 1
        sm["failures"] = tags
    return sm


def run_eval(preset: str = "advanced", retrieval_only: bool = False, use_judge: bool = True,
             ids: list[str] | None = None, label: str = "", write: bool = True) -> dict:
    s = get_settings()
    cfg = PRESETS[preset]
    questions = [q for q in load_questions() if not ids or q["id"] in ids]
    retrieve("warm-up", top_n=1, strategy=cfg["strategy"], rerank=cfg["rerank"], rewrite=False)
    workers = 1 if retrieval_only else s.eval_workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda q: evaluate_one(q, cfg, retrieval_only, use_judge), questions))
    for r in rows:
        r["failure"] = None if retrieval_only or r.get("error") else failure_tag(r)
    config = {"preset": preset, **cfg, "retrieval_only": retrieval_only, "top_n": s.top_n,
              "context_budget": s.context_budget, "llm_model": None if retrieval_only else s.gemini_model,
              "judge_model": s.judge_model if use_judge and not retrieval_only else None,
              "embed_model": s.embed_model, "rerank_model": s.rerank_model if cfg["rerank"] else None,
              "n_questions": len(rows)}
    result = {"config": config, "summary": summarize(rows), "rows": rows}
    if write:
        write_report(result, label)
    return result


def write_report(result: dict, label: str = "") -> Path:
    s = get_settings()
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"eval_{stamp}_{cfg['preset']}" + ("_retrieval" if cfg["retrieval_only"] else "") + (f"_{label}" if label else "")
    (s.reports_dir / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    sm = result["summary"]
    lines = [f"# Eval report — {name}", "", "## Config", "", "```json", json.dumps(cfg, indent=2), "```", "",
             "## Summary", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sm.items() if not isinstance(v, dict)]
    if "by_category" in sm:
        lines += ["", "## By category", "", "| Category | n | Correct | Context recall |", "|---|---|---|---|"]
        lines += [f"| {c} | {v['n']} | {v['correct']} | {v['context_recall']} |" for c, v in sm["by_category"].items()]
        lines += ["", f"Failures: {sm.get('failures')}"]
    lines += ["", "## Per question", ""]
    if cfg["retrieval_only"]:
        lines += ["| id | category | first hit | context recall | evidence in ctx | top retrieved |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['category']} | {r.get('first_hit_rank')} | {r.get('context_recall')} | "
                  f"{r.get('evidence_in_context')} | {', '.join(r['retrieved'][:3])} |" for r in result["rows"]]
    else:
        lines += ["| id | category | ctx recall | judgement | faithful | failure | latency s |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['category']} | {r.get('context_recall')} | {r.get('error') or r.get('judgement', r.get('correct'))} | "
                  f"{r.get('faithful')} | {r.get('failure') or ''} | {r['latency_ms'] / 1000:.1f} |" for r in result["rows"]]
    path = s.reports_dir / f"{name}.md"
    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", path)
    return path


def write_ablation(presets: list[str], full: bool = False) -> str:
    """Run each preset and write a single comparison table (eval/reports/ablation_*.md)."""
    s = get_settings()
    rows = []
    for p in presets:
        res = run_eval(preset=p, retrieval_only=not full, use_judge=full, label="ablation")
        rows.append((p, res["summary"]))
    cols = ["hit@1", "hit@3", "hit@5", "mrr", "groups_covered_multi", "context_recall", "evidence_in_context",
            "avg_context_tokens"] + (["correct", "correct_or_partial", "faithful", "citation_on_gold_page",
                                       "refusal_accuracy"] if full else [])
    lines = ["| preset | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {p} | " + " | ".join(str(sm.get(c)) for c in cols) + " |" for p, sm in rows]
    table = "\n".join(lines)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (s.reports_dir / f"ablation_{stamp}{'_full' if full else ''}.md").write_text(
        f"# Ablation ({'answers + judge' if full else 'retrieval only'})\n\n{table}\n")
    return table
