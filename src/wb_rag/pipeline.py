"""Online path: question -> plan -> retrieve -> build context -> generate -> Answer."""

import time

from wb_rag.context import build_context
from wb_rag.generate import generate
from wb_rag.index import load_index
from wb_rag.models import Answer
from wb_rag.retrieve import retrieve


def ask(question: str, top_n: int | None = None, mode: str | None = None, strategy: str | None = None,
        **retrieval_opts) -> Answer:
    t0 = time.perf_counter()
    retrieved, plan = retrieve(question, top_n=top_n, mode=mode, strategy=strategy, **retrieval_opts)
    blocks = build_context(retrieved, plan, index=load_index(strategy))
    out = generate(question, blocks)
    return Answer(question=question, retrieved=retrieved, context=blocks, plan=plan,
                  latency_ms=int((time.perf_counter() - t0) * 1000), **out)
