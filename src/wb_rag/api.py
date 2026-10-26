"""FastAPI service: chat UI at /, POST /ask, GET /outline, POST /ingest, GET /health, GET /files/report.pdf."""

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from wb_rag.config import get_settings
from wb_rag.index import load_index
from wb_rag.llm import LLMError, api_key
from wb_rag.models import Answer
from wb_rag.pipeline import ask

app = FastAPI(title="World Bank GEP RAG", version="0.1.0")

WEB_DIR = Path(__file__).parent / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["dense", "bm25", "hybrid"] | None = None
    preset: Literal["advanced", "baseline"] = "advanced"
    rewrite: bool | None = None


@app.get("/health")
def health() -> dict:
    s = get_settings()
    try:
        manifest = load_index().manifest
        manifest = {k: v for k, v in manifest.items() if k != "countries"}
    except FileNotFoundError:
        manifest = None
    outline = s.processed_dir / "outline.json"
    nodes = len(json.loads(outline.read_text())["sections"]) if outline.exists() else 0
    return {"status": "ok" if manifest else "no_index", "index": manifest, "outline_nodes": nodes,
            "llm_model": s.gemini_model, "embed_model": s.embed_model, "rerank_model": s.rerank_model,
            "credentials": bool(api_key())}


@app.get("/", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/files/report.pdf", include_in_schema=False)
def report_pdf() -> FileResponse:
    s = get_settings()
    return FileResponse(s.raw_dir / f"{s.doc_id}.pdf", media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{s.doc_id}.pdf"'})


@app.get("/outline")
def outline() -> dict:
    path = get_settings().processed_dir / "outline.json"
    if not path.exists():
        raise HTTPException(503, "Outline not built — run `wb-rag ingest`.")
    return json.loads(path.read_text())


@app.post("/ask", response_model=Answer)
def ask_endpoint(req: AskRequest) -> Answer:
    opts: dict = {"rewrite": req.rewrite}
    if req.preset == "baseline":
        opts.update(strategy="flat", rerank=False, routing=False)
    try:
        ans = ask(req.question, top_n=req.top_k, mode=req.mode, **opts)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except LLMError as e:
        raise HTTPException(502, str(e))
    return ans


@app.post("/ingest")
def ingest_endpoint(rebuild: bool = False) -> dict:
    from wb_rag.ingest.pipeline import run_ingest

    return run_ingest(rebuild=rebuild)
