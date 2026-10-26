"""Command line: wb-rag ingest | outline | ask | eval | ablate | serve."""

import json
import logging
from typing import Optional

import typer

from wb_rag.config import get_settings

app = typer.Typer(add_completion=False, help="Advanced RAG over the World Bank Global Economic Prospects (June 2025).")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def ingest(rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild even if the index is current"),
           strategy: str = typer.Option("all", help="hierarchical | flat | all")) -> None:
    """Parse the PDF, build the outline, chunk, summarise, embed, and write the index(es)."""
    from wb_rag.ingest.pipeline import run_ingest

    s = get_settings()
    for strat in (["hierarchical", "flat"] if strategy == "all" else [strategy]):
        s.chunk_strategy = strat  # type: ignore[assignment]
        typer.echo(json.dumps(run_ingest(rebuild=rebuild), indent=2))
    s.chunk_strategy = "hierarchical"


@app.command()
def outline() -> None:
    """Print the document outline recovered from the Contents pages."""
    s = get_settings()
    data = json.loads((s.processed_dir / "outline.json").read_text())
    for sec in data["sections"]:
        region = f" [{sec['region']}]" if sec["region"] else ""
        typer.echo(f"{'  ' * sec['level']}{sec['title']}{region}  (pp.{sec['page_start']}-{sec['page_end']})")


@app.command()
def ask(
    question: str,
    top_n: Optional[int] = typer.Option(None, "--k", help="Hits kept after reranking"),
    mode: Optional[str] = typer.Option(None, help="dense | bm25 | hybrid"),
    strategy: Optional[str] = typer.Option(None, help="hierarchical | flat"),
    rewrite: bool = typer.Option(False, "--rewrite", help="LLM multi-query decomposition"),
    show_context: bool = typer.Option(False, "--show-context", help="Print the context blocks"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Skip the LLM; show retrieval"),
) -> None:
    """Answer a question with citations."""
    if retrieval_only:
        from wb_rag.retrieve import retrieve

        hits, plan = retrieve(question, top_n=top_n, mode=mode, strategy=strategy, rewrite=rewrite or None)
        typer.secho(f"plan: {plan.model_dump(exclude={'question'})}", dim=True)
        for r in hits:
            c = r.chunk
            typer.secho(f"#{r.rank} p.{c.pages} [{c.kind}{' ' + c.label if c.label else ''}] {c.section}"
                        f"{' | ' + c.box if c.box else ''}  (score={r.score}, rerank={r.rerank_score and round(r.rerank_score, 2)}, "
                        f"dense={r.dense_rank}, bm25={r.bm25_rank}{', routed' if r.routed else ''})", bold=True)
            typer.echo(c.text[:500] + ("..." if len(c.text) > 500 else "") + "\n")
        return

    from wb_rag.llm import LLMError
    from wb_rag.pipeline import ask as run_ask

    try:
        ans = run_ask(question, top_n=top_n, mode=mode, strategy=strategy, rewrite=rewrite or None)
    except LLMError as e:
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(1)
    typer.echo("\n" + ans.answer + "\n")
    if ans.citations:
        typer.secho("Sources", bold=True)
        for c in ans.citations:
            pp = f"p.{c.page}" if c.page == c.page_end else f"pp.{c.page}-{c.page_end}"
            typer.echo(f"  [{c.n}] {pp} (printed {c.printed_page}) — {c.section}")
    if show_context:
        typer.secho("\nContext", bold=True)
        for b in ans.context:
            typer.echo(f"  [{b.n}] p.{b.page_start}-{b.page_end} {b.kind} {b.n_tokens} tok — {b.section}")
    u = ans.usage
    typer.secho(f"\n{ans.latency_ms / 1000:.1f}s · in {u.get('input_tokens')} / out {u.get('output_tokens')} tokens · "
                f"{ans.model}", dim=True)


@app.command("eval")
def eval_cmd(
    preset: str = typer.Option("advanced", help="Pipeline preset (see evaluation/run.py PRESETS)"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Retrieval metrics only (no LLM calls)"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the LLM judge"),
    ids: Optional[str] = typer.Option(None, help="Comma-separated question ids"),
    label: str = typer.Option("", help="Suffix for the report filename"),
) -> None:
    """Run the sample-question evaluation and write a report to eval/reports/."""
    from wb_rag.evaluation.run import run_eval

    result = run_eval(preset=preset, retrieval_only=retrieval_only, use_judge=not no_judge,
                      ids=ids.split(",") if ids else None, label=label)
    typer.echo(json.dumps({"config": result["config"], "summary": result["summary"]}, indent=2))


@app.command()
def ablate(presets: str = typer.Option("", help="Comma-separated presets (default: all)"),
           full: bool = typer.Option(False, "--full", help="Also generate + judge answers (LLM calls)")) -> None:
    """Compare pipeline presets: baseline -> + structure -> + rerank -> + routing -> + small-to-big."""
    from wb_rag.evaluation.run import PRESETS, write_ablation

    names = presets.split(",") if presets else list(PRESETS)
    typer.echo(write_ablation(names, full=full))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server with the chat UI."""
    import uvicorn

    uvicorn.run("wb_rag.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
