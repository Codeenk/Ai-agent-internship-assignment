"""Embedding client with disk cache and a deterministic offline fallback.

Embeddings use the OpenAI-compatible ``/embeddings`` endpoint. To keep the
project dependency-free and honest about cost, vectors are cached on disk
(keyed by model + normalized text). When no API key is configured — or the
network fails — the index switches to a deterministic local hashing-based
vectorizer so retrieval, and therefore the whole agent, still works offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

from .bm25 import tokenize
from .config import CONFIG

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIM = 512


class EmbeddingError(RuntimeError):
    pass


def _cache_path(model: str, text: str) -> Path:
    key = hashlib.sha256(f"{model}::{text}".encode()).hexdigest()[:32]
    return _CACHE_DIR / "embeddings" / model / f"{key}.json"


def _embed_via_api(texts: list[str]) -> list[list[float]]:
    cfg = CONFIG.embeddings
    if not cfg.api_key:
        raise EmbeddingError("no embedding API key configured")
    body = json.dumps({"model": cfg.model, "input": texts}).encode()
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
        payload = json.loads(resp.read().decode())
    data = sorted(payload["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


# ---------------------------------------------------------------------------
# Offline fallback: feature hashing over tokens (+ bigrams), L2-normalized.
# Deterministic across runs and machines; purely local.
# ---------------------------------------------------------------------------

def _hashed_vector(text: str) -> list[float]:
    tokens = tokenize(text)
    feats: list[str] = list(tokens)
    feats += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    vec = [0.0] * _DIM
    for feat in feats:
        h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
        idx = h % _DIM
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    return dot


class EmbeddingIndex:
    """Dense index over chunks with an offline fallback vectorizer."""

    def __init__(self) -> None:
        self.chunks: list = []
        self.vectors: list[list[float]] = []
        self.mode: str = "hashing"  # or "api"
        self._api_ready = bool(CONFIG.embeddings.api_key)

    def build(self, chunks: list) -> None:
        self.chunks = chunks
        texts = [f"{c.heading}\n{c.text}" for c in chunks]
        if self._api_ready:
            try:
                self.vectors = self._embed_all(texts)
                self.mode = "api"
                return
            except Exception:
                self.mode = "hashing"  # graceful degradation
        self.vectors = [_hashed_vector(t) for t in texts]

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        missing: list[int] = []
        for i, t in enumerate(texts):
            p = _cache_path(CONFIG.embeddings.model, t)
            if p.exists():
                out.append(json.loads(p.read_text()))
            else:
                out.append([])
                missing.append(i)
        if missing:
            # Batch in groups of 64 to stay well under request limits.
            for start in range(0, len(missing), 64):
                batch_idx = missing[start:start + 64]
                batch_texts = [texts[i] for i in batch_idx]
                vectors = _embed_via_api(batch_texts)
                for i, vec in zip(batch_idx, vectors):
                    out[i] = vec
                    p = _cache_path(CONFIG.embeddings.model, texts[i])
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(vec))
        return out

    def search(self, query: str, k: int = 8) -> list[tuple[object, float]]:
        if not self.chunks:
            return []
        qvec = (
            self._embed_all([query])[0]
            if self.mode == "api"
            else _hashed_vector(query)
        )
        scored = [(_cosine(qvec, v), i) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[object, float]]], k: int = 60
) -> list[tuple[object, float]]:
    """Fuse multiple rankings with Reciprocal Rank Fusion (score in [0,1])."""
    scores: dict[int, float] = {}
    order: dict[int, object] = {}
    for ranking in rankings:
        for rank, (chunk, _s) in enumerate(ranking):
            key = id(chunk)
            order[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(order[key], score) for key, score in fused]
