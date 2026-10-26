"""Search index: searchable chunks + parent store + embeddings + BM25 + metadata arrays.

At ~1.5k chunks an exact cosine search over a NumPy matrix takes about a millisecond, so no
vector DB is needed. The metadata arrays (kind, region, chapter, label) make filtering and
routing a boolean mask.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from wb_rag.config import Settings, get_settings
from wb_rag.models import Chunk

log = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who how did does do according report gep".split()
)
_TOKEN = re.compile(r"\$?\d[\d,.]*%?|[a-z][a-z0-9'&.-]*", re.I)


def bm25_tokenize(text: str) -> list[str]:
    tokens = []
    for t in _TOKEN.findall(text.lower()):
        t = t.strip(".,'-")
        if not t or t in _STOPWORDS:
            continue
        tokens.append(t)
        bare = t.replace("$", "").replace(",", "").rstrip("%")
        if bare != t and bare:
            tokens.append(bare)
    return tokens


def save_index(chunks: list[Chunk], parents: list[Chunk], vectors: np.ndarray, extra: dict,
               s: Settings | None = None) -> None:
    s = s or get_settings()
    d = s.index_dir
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embeddings.npy", vectors)
    for name, rows in (("chunks.jsonl", chunks), ("parents.jsonl", parents)):
        with open(d / name, "w") as f:
            for c in rows:
                f.write(c.model_dump_json() + "\n")
    manifest = {
        "embed_model": s.embed_model,
        "chunk_strategy": s.chunk_strategy,
        "n_chunks": len(chunks),
        "n_parents": len(parents),
        "dim": int(vectors.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    load_index.cache_clear()


@dataclass
class Index:
    chunks: list[Chunk]
    parents: dict[str, Chunk]
    vectors: np.ndarray
    bm25: BM25Okapi
    manifest: dict
    kind: np.ndarray
    region: np.ndarray
    chapter: np.ndarray
    label: np.ndarray
    box: np.ndarray
    by_id: dict[str, int]

    @property
    def countries(self) -> dict[str, str]:
        return self.manifest.get("countries", {})


@lru_cache
def load_index(strategy: str | None = None) -> Index:
    s = get_settings()
    d = s.index_root / f"index_{strategy or s.chunk_strategy}"
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index not found at {d} — run `wb-rag ingest` first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["embed_model"] != s.embed_model:
        raise RuntimeError(f"Index was built with {manifest['embed_model']} but config uses {s.embed_model}; "
                           "re-run ingest.")
    chunks = [Chunk.model_validate_json(line) for line in open(d / "chunks.jsonl")]
    parents = {c.chunk_id: c for c in (Chunk.model_validate_json(l) for l in open(d / "parents.jsonl"))}
    vectors = np.load(d / "embeddings.npy")
    bm25 = BM25Okapi([bm25_tokenize(c.embed_text) for c in chunks])
    arr = lambda f: np.array([f(c) or "" for c in chunks], dtype=object)  # noqa: E731
    return Index(
        chunks=chunks, parents=parents, vectors=vectors, bm25=bm25, manifest=manifest,
        kind=arr(lambda c: c.kind), region=arr(lambda c: c.region), chapter=arr(lambda c: c.chapter),
        label=arr(lambda c: c.label), box=arr(lambda c: c.box),
        by_id={c.chunk_id: i for i, c in enumerate(chunks)},
    )
