"""Local bge embeddings and cross-encoder reranking via fastembed (ONNX, no PyTorch), plus the tokenizer."""

import logging
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer

from wb_rag.config import get_settings

log = logging.getLogger(__name__)


@lru_cache
def _model(name: str) -> TextEmbedding:
    log.info("Loading embedding model %s", name)
    return TextEmbedding(model_name=name, cache_dir=str(get_settings().embed_cache_dir))


@lru_cache
def _tokenizer(name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(name)


def count_tokens(text: str) -> int:
    return len(_tokenizer(get_settings().embed_model).encode(text, add_special_tokens=False).ids)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    s = get_settings()
    vecs = np.array(list(_model(s.embed_model).embed(texts, batch_size=s.embed_batch_size)))
    return _normalize(vecs)


def embed_query(text: str) -> np.ndarray:
    s = get_settings()
    vec = np.array(list(_model(s.embed_model).embed([s.query_instruction + text])))
    return _normalize(vec)[0]


@lru_cache
def _reranker(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    log.info("Loading reranker %s", name)
    return TextCrossEncoder(model_name=name, cache_dir=str(get_settings().embed_cache_dir))


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance logits for (query, text) pairs."""
    return list(_reranker(get_settings().rerank_model).rerank(query, texts, batch_size=16))
