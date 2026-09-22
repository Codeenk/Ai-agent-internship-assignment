"""Retrieval pipeline with document precedence and conflict detection.

Precedence rules (deterministic, metadata-driven — see
knowledge-base/13-support-escalation.md):

1. Only ``status: active`` + ``policy_authority: official`` documents may be
   used as the *authority* for an answer.
2. ``superseded`` documents are excluded unless the customer explicitly asks
   about the old policy (historical questions).
3. ``draft``/``policy_authority: none`` documents (the migration scratchpad)
   are never authoritative, and their instruction-like lines are treated as
   untrusted data.
4. If two active official documents genuinely conflict (e.g. the Product Care
   Guide says hand-wash the Breeze Tumbler body while the product card says
   all components are dishwasher-safe), the conflict is surfaced to the
   customer rather than silently resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math
import re

from .bm25 import BM25Index, build_index, tokenize
from .config import CONFIG
from .docmeta import DocMeta
from .embeddings import EmbeddingIndex
from .kb import Chunk, load_document

# Query terms that should contribute to topic-sufficiency checks.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "did", "can", "i", "my", "me",
    "you", "your", "we", "us", "our", "it", "to", "of", "for", "with", "on",
    "and", "or", "how", "what", "when", "where", "why", "will", "would", "should",
    "about", "have", "has", "if", "in", "at", "be", "get", "got", "am",
}


@dataclass
class RetrievedPassage:
    """A retrieved chunk prepared for the prompt (and the trace)."""

    chunk: Chunk
    score: float

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def heading(self) -> str:
        return self.chunk.heading

    @property
    def ref(self) -> str:
        return f"{self.chunk.source} — {self.chunk.meta.title} § {self.chunk.heading}"

    def prompt_block(self) -> str:
        """Render this passage for the LLM prompt with metadata framing."""
        meta = self.chunk.meta
        return (
            f"[{self.ref}] (status: {meta.status}, authority: {meta.policy_authority})\n"
            f"{self.chunk.text}"
        )


@dataclass
class RetrievalResult:
    passages: list[RetrievedPassage] = field(default_factory=list)
    topic_supported: bool = True
    conflicts: list[dict] = field(default_factory=list)
    mode: str = "hybrid"  # hybrid | bm25 | historical

    def best(self) -> RetrievedPassage | None:
        return self.passages[0] if self.passages else None


def _dedupe(passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
    """Keep the highest-scored passage per (source, heading)."""
    seen: dict[tuple[str, str], RetrievedPassage] = {}
    for p in passages:
        key = (p.source, p.heading)
        if key not in seen or p.score > seen[key].score:
            seen[key] = p
    return sorted(seen.values(), key=lambda p: p.score, reverse=True)


def _weighted_rrf(
    primary: list[tuple[object, float]],
    secondary: list[tuple[object, float]],
    bm25_weight: float = 2.0,
    k: int = 60,
) -> list[tuple[object, float]]:
    """RRF where the lexical (BM25) ranking carries more weight.

    With a dependency-free hashing vectorizer as the dense channel, the
    lexical signal is far more reliable for exact policy vocabulary ("30
    calendar days", "hand-wash"). Weighting BM25 keeps policy sections on
    top while the dense channel still helps with paraphrase recall.
    """
    scores: dict[int, float] = {}
    order: dict[int, object] = {}
    for ranking, weight in ((primary, bm25_weight), (secondary, 0.5)):
        for rank, (chunk, _s) in enumerate(ranking):
            key = id(chunk)
            order[key] = chunk
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(order[key], score) for key, score in fused]


class Retriever:
    """Hybrid BM25 + dense retrieval with metadata-driven precedence."""

    def __init__(self, kb_dir=None) -> None:
        from .config import KB_DIR

        self.docs: dict[str, DocMeta] = {}
        self.chunks: list[Chunk] = []
        paths = sorted((kb_dir or KB_DIR).glob("*.md"))
        for path in paths:
            meta, doc_chunks = load_document(path)
            self.docs[meta.path] = meta
            self.chunks.extend(doc_chunks)
        self.bm25: BM25Index = build_index(self.chunks)
        self.dense: EmbeddingIndex = EmbeddingIndex()
        self.dense.build(self.chunks)
        self._doc_term_cache: dict[tuple[str, str], bool] = {}

    # -- public API ---------------------------------------------------------

    def retrieve(self, query: str, allow_superseded: bool = False) -> RetrievalResult:
        """Retrieve passages for *query* applying precedence rules.

        ``allow_superseded`` permits legacy documents (used when the customer
        explicitly asks about a past policy). They are still marked so the
        generator can label them as historical.
        """
        rc = CONFIG.retrieval
        bm25_hits = self.bm25.search(query, k=rc.candidate_k)
        dense_hits = self.dense.search(query, k=rc.candidate_k)
        fused = _weighted_rrf(bm25_hits, dense_hits, bm25_weight=2.5)

        selected: list[RetrievedPassage] = []
        mode = "hybrid"
        if not bm25_hits:
            mode = "dense-only"
        for chunk, score in fused:
            if chunk.meta.status == "superseded" and not allow_superseded:
                continue
            if chunk.meta.status == "superseded":
                mode = "historical"
            # Draft/internal docs may enter the context ONLY as untrusted
            # data; they can never be cited as authority. Authority is
            # enforced by the authority boost below + generator prompt.
            selected.append(RetrievedPassage(chunk=chunk, score=round(score, 6)))
            if len(selected) >= rc.candidate_k:
                break

        if not selected:
            return RetrievalResult(topic_supported=False, mode=mode)

        # Authority-aware re-ranking: active official docs first, then
        # superseded (explicit queries only), then draft/internal — which
        # also keeps instruction-like content away from the prompt window.
        def rank_key(p: RetrievedPassage):
            if p.chunk.meta.is_active_official:
                tier = 0
            elif p.chunk.meta.status == "superseded":
                # Explicitly-historical queries let legacy docs compete on score.
                tier = 0 if allow_superseded else 1
            else:
                tier = 2
            return (tier, -p.score)

        selected.sort(key=rank_key)
        selected = _dedupe(selected)
        selected.sort(key=rank_key)
        selected = selected[: rc.final_k]
        if selected:
            floor = rc.min_score_ratio * selected[0].score
            selected = [p for p in selected if p.score >= floor]

        # Document-context padding: when a document is clearly relevant, its
        # best sibling section often holds the supporting detail (e.g.
        # "Supported destinations" + "Canada delivery estimate"). Append the
        # best active-official sibling for the top two documents.
        top_docs: list[str] = []
        for p in selected:
            if p.chunk.meta.is_active_official and p.source not in top_docs:
                top_docs.append(p.source)
            if len(top_docs) == 2:
                break
        chosen_ids = {p.chunk.chunk_id for p in selected}
        padded: list[RetrievedPassage] = []
        for src in top_docs:
            best: tuple[Chunk, float] | None = None
            for cand, s in fused:
                if cand.source == src and cand.meta.is_active_official \
                        and cand.chunk_id not in chosen_ids:
                    if best is None or s > best[1]:
                        best = (cand, s)
            if best is not None:
                padded.append(RetrievedPassage(chunk=best[0], score=round(best[1], 6)))
                chosen_ids.add(best[0].chunk_id)
        selected = (selected + padded)[: rc.final_k + 2]

        # Historical queries: guarantee the best superseded passage is
        # available so the model can answer "what did the policy USED TO be?"
        if allow_superseded:
            best_legacy: tuple[Chunk, float] | None = None
            legacy_ids = {p.chunk.chunk_id for p in selected if p.chunk.meta.status == "superseded"}
            if not legacy_ids:
                for cand, s in fused:
                    if cand.meta.status == "superseded":
                        if best_legacy is None or s > best_legacy[1]:
                            best_legacy = (cand, s)
                if best_legacy is not None:
                    selected.append(
                        RetrievedPassage(chunk=best_legacy[0], score=round(best_legacy[1], 6))
                    )

        # Topic-support gate: a query is supported when enough of its
        # distinctive terms appear in strong authoritative passages.
        topic_supported = self._topic_supported(query, selected)

        conflicts = self._detect_conflicts(selected)
        return RetrievalResult(
            passages=selected, topic_supported=topic_supported, conflicts=conflicts, mode=mode
        )

    def _topic_supported(self, query: str, passages: list[RetrievedPassage]) -> bool:
        """Conservative sufficiency gate for abstention.

        Abstain at the retrieval layer only when there is essentially no
        lexical support: no selected passage or best BM25 below a small
        floor. Questions whose answer is a NEGATIVE ("do you ship to
        Germany?" -> "only Canada") legitimately lack the entity's own
        vocabulary, so absence of nouns must NOT trigger abstention.

        Subtle insufficiency (right document, missing detail, e.g. vegan
        material certification) is the generator's job: its grounding rules
        require it to say the supplied information is insufficient and
        recommend a human. That behavior is verified by the eval suite with
        the live model and by unit tests with the extractive fallback.
        """
        if not passages:
            return False
        best_active = next((p for p in passages if p.chunk.meta.is_active_official), None)
        if best_active is None:
            return False
        hits = self.bm25.search(query, k=1)
        if not hits or hits[0][1] < 2.5:
            return False
        return True

    def _detect_conflicts(self, passages: list[RetrievedPassage]) -> list[dict]:
        """Surface genuine conflicts between current active official docs.

        The corpus contains one planted conflict: Product Care Guide (11)
        says the Breeze Tumbler body must be hand-washed while the product
        card (12) says all components are dishwasher safe. We detect this
        class of conflict generically: two ACTIVE OFFICIAL docs retrieved for
        the same query whose guidance on the same topic is mutually
        exclusive per the conflict lexicon below.
        """
        rc = CONFIG.retrieval
        actives = [
            p
            for p in passages[: rc.conflict_overlap]
            if p.chunk.meta.is_active_official
        ]
        conflicts: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Conflict lexicon: mutually exclusive guidance markers. Both sides
        # must appear in the same retrieved set about the same product/topic.
        exclusive_pairs = [
            ("hand-wash", "dishwasher safe"),
            ("do not machine wash", "machine washable"),
            ("never", "always"),
        ]
        for i in range(len(actives)):
            for j in range(i + 1, len(actives)):
                a, b = actives[i], actives[j]
                if a.source == b.source:
                    continue
                text_a, text_b = a.chunk.text.lower(), b.chunk.text.lower()
                for x, y in exclusive_pairs:
                    if (x in text_a and y in text_b) or (y in text_a and x in text_b):
                        pair = tuple(sorted((a.source, b.source)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        conflicts.append(
                            {
                                "sources": [a.source, b.source],
                                "topic": f"{a.heading} / {b.heading}",
                                "statement_a": a.ref,
                                "statement_b": b.ref,
                            }
                        )
        return conflicts
