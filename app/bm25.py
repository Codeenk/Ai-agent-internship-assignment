"""A compact BM25 (Okapi) index — no external dependencies.

Tokens are lowercased words with simple suffix folding so that "returns",
"return" and "returning" share weight without pulling in a stemmer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Lightweight suffix folding: cheap, deterministic, and good enough to bridge
# the vocabulary gap between customer questions and policy prose.
_FOLDS = (
    ("ization", "ize"),
    ("ations", "ate"),
    ("ation", "ate"),
    ("ities", "ity"),
    ("ements", "ement"),
    ("ement", "e"),
    ("ingly", ""),
    ("edly", ""),
    ("ies", "y"),
    ("ifest", "est"),  # placeholder no-op guard, harmless
    ("sses", "ss"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("xes", "x"),
    ("zes", "z"),
    ("ders", "d"),
    ("ings", "ing"),
    ("ss", "ss"),
)


# Small canonicalization map: bridges the vocabulary gap between customer
# wording and document wording without a real stemmer. Chosen deliberately:
# each entry fixed a retrieval miss observed in the bug diary (README).
_SYNONYMS = {
    "canadian": "canada",
    "shipped": "ship",
    "shipping": "ship",
    "shipment": "ship",
    "shipments": "ship",
    "delivered": "deliver",
    "delivery": "deliver",
    "deliveries": "deliver",
    "returned": "return",
    "returns": "return",
    "returning": "return",
    "cancelled": "cancel",
    "canceled": "cancel",
    "refunded": "refund",
    "refunds": "refund",
    "warranties": "warranty",
    "dishwashers": "dishwasher",
}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in tokens:
        if len(tok) <= 3:
            out.append(tok)
            continue
        for suffix, repl in _FOLDS:
            if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
                tok = tok[: len(tok) - len(suffix)] + repl
                break
        else:
            # Generic plural fold for longer tokens (ships -> ship, days -> day).
            if tok.endswith("s") and not tok.endswith(("ss", "us", "is")) and len(tok) >= 4:
                tok = tok[:-1]
        out.append(_SYNONYMS.get(tok, tok))
    return out


@dataclass
class BM25Index:
    """Okapi BM25 over pre-chunked passages."""

    k1: float = 1.2
    b: float = 0.75
    _docs: list = field(default_factory=list, init=False)
    _doc_lens: list[int] = field(default_factory=list, init=False)
    _tf: list[Counter] = field(default_factory=list, init=False)
    _df: Counter = field(default_factory=Counter, init=False)
    _N: int = field(default=0, init=False)
    _avgdl: float = field(default=1.0, init=False)

    def add(self, chunk) -> None:
        tokens = tokenize(f"{chunk.heading} {chunk.text} {chunk.source}")
        self._docs.append(chunk)
        self._doc_lens.append(max(len(tokens), 1))
        tf = Counter(tokens)
        self._tf.append(tf)
        for term in tf:
            self._df[term] += 1
        self._N += 1
        total = sum(self._doc_lens)
        self._avgdl = total / max(self._N, 1)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        n_over = (self._N - df + 0.5) / (df + 0.5)
        return math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def score(self, query: str, index: int) -> float:
        q_tokens = tokenize(query)
        tf = self._tf[index]
        dl = self._doc_lens[index]
        s = 0.0
        for term in q_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def search(self, query: str, k: int = 10) -> list[tuple[object, float]]:
        scores = [(self.score(query, i), i) for i in range(self._N)]
        scores = [s for s in scores if s[0] > 0]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [(self._docs[i], s) for s, i in scores[:k]]


def build_index(chunks: Iterable) -> BM25Index:
    index = BM25Index()
    for chunk in chunks:
        index.add(chunk)
    return index
