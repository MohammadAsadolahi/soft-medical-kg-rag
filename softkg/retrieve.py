"""Retrieval: resolve soft patterns against the graph, then rank documents.

Three steps per query.

**1. Rewrite.** An LLM turns the question into soft patterns (``softkg/prompts.py``). This is the only
LLM call at query time.

**2. Resolve.** Each pattern's *shape* selects a matching strategy -- which subspaces to score
against and which type filters to apply. That dispatch is the substance of this module and is laid out
in ``_STRATEGIES`` below. A pattern yields scored triplets.

**3. Aggregate.** Triplet scores become document scores. Default is max-pooling: a document is as
relevant as its single best-matching fact.

Two deliberate departures from the original research prototype, both because the prototype's choices
were inert rather than merely different:

*Scores are cosine similarities, combined by arithmetic mean.* The prototype multiplied raw L2
distances as ``(d1+1)*(d2+1)`` and sorted ascending. With normalised vectors that product is not
monotone in either component -- it can rank a pair of mediocre matches above one excellent and one
fair match -- so a mean of cosines is used instead. It is bounded, monotone in every component, and
directly interpretable.

*Candidate sets are bounded by top-k, not by an absolute distance threshold.* The prototype filtered
with thresholds of 100 and 200 on L2 distance. Between L2-normalised vectors the maximum possible
distance is 2, so every threshold admitted the entire index and filtered nothing. Explicit top-k
makes the intended cutoff real and tunable.

Aggregation is max-pooling by default. ``"sum{n}"`` (sum of a document's n best facts) is also
provided, since "many weakly relevant facts" and "one strongly relevant fact" are genuinely different
notions of document relevance and which one suits a task is an empirical question, not a given.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .embedder import Embedder
from .graph import SoftKnowledgeGraph, Triplet
from .llm import LLMClient
from .patterns import Pattern, SlotKind, parse_patterns
from .prompts import SEARCH_PATTERNS_TYPED

logger = logging.getLogger(__name__)


@dataclass
class TripletHit:
    """One matched fact and why it matched."""

    index: int
    score: float
    triplet: Triplet
    shape: str
    pattern: str

    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.triplet}   <- {self.pattern}"


@dataclass
class DocumentHit:
    """A retrieved document with the facts that support it."""

    doc_id: str
    score: float
    supporting: list[TripletHit] = field(default_factory=list)

    def explain(self, limit: int = 3) -> str:
        """Human-readable provenance.

        Worth noting as a property of the approach rather than a convenience: the retrieval unit is a
        typed fact, so every ranked document arrives with the specific extracted statements that
        caused it to rank. A dense whole-document retriever can only report a similarity score.
        """
        lines = [f"{self.doc_id}  score={self.score:.4f}"]
        for hit in self.supporting[:limit]:
            lines.append(f"    {hit}")
        if len(self.supporting) > limit:
            lines.append(f"    ... and {len(self.supporting) - limit} more facts")
        return "\n".join(lines)


class SoftRetriever:
    """Pattern-based retrieval over a :class:`SoftKnowledgeGraph`."""

    def __init__(
        self,
        graph: SoftKnowledgeGraph,
        embedder: Embedder,
        *,
        llm: LLMClient | None = None,
        search_prompt: str = SEARCH_PATTERNS_TYPED,
        slot_top_k: int = 2000,
        per_pattern: int = 100,
        include_query_channel: bool = True,
        query_channel_k: int = 100,
    ) -> None:
        """
        ``slot_top_k``           candidates kept from a single-slot similarity scan before any
                                 intersection or rescoring. Generous, because it is a recall funnel:
                                 a fact discarded here cannot be recovered by a later stage.
        ``per_pattern``          facts each pattern may contribute to the final pool. Bounds the
                                 influence of any one pattern so a broad pattern cannot swamp the
                                 precise ones.
        ``include_query_channel`` also match the raw question against the ``full`` subspace. Retained
                                 from the original design: it is the safety net when pattern
                                 generation misfires, and it is what a no-LLM ablation reduces to.
        """
        if not graph.is_built:
            raise RuntimeError("graph vectors are not built -- call graph.build_vectors() first")
        self.graph = graph
        self.embedder = embedder
        self.llm = llm
        self.search_prompt = search_prompt
        self.slot_top_k = slot_top_k
        self.per_pattern = per_pattern
        self.include_query_channel = include_query_channel
        self.query_channel_k = query_channel_k

    # -- step 1: rewrite ---------------------------------------------------
    def generate_patterns(self, query: str, *, model: str | None = None) -> list[Pattern]:
        """Ask the LLM for soft patterns. Returns [] when no LLM is configured."""
        if self.llm is None:
            return []
        reply = self.llm.chat(self.search_prompt, query, model=model)
        patterns = parse_patterns(reply or "")
        if not patterns:
            logger.warning("no parseable pattern for %r; falling back to the query channel", query)
        return patterns

    # -- step 2: resolve ---------------------------------------------------
    def _vectors_for(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Batch-embed every query-side string a set of patterns needs."""
        wanted = sorted({t for t in texts if t})
        if not wanted:
            return {}
        matrix = self.embedder.encode_queries(wanted)
        return dict(zip(wanted, matrix))

    def _combine(self, *components: dict[int, float]) -> dict[int, float]:
        """Mean of the similarity components available for each candidate.

        Averaging over *present* components only, rather than treating a missing component as zero,
        keeps candidates found by one channel comparable to those found by several. Penalising a fact
        for not appearing in a channel it was never eligible for would be an artefact of the funnel,
        not evidence about relevance.
        """
        totals: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for component in components:
            for idx, score in component.items():
                totals[idx] += score
                counts[idx] += 1
        return {idx: totals[idx] / counts[idx] for idx in totals}

    def _scan(self, subspace: str, text: str, vectors: dict[str, np.ndarray],
              candidates: np.ndarray | None = None, k: int | None = None) -> dict[int, float]:
        """Top-k similarity scan of one subspace, as {row index: score}."""
        vector = vectors.get(text)
        if vector is None:
            return {}
        idx, scores = self.graph.topk(subspace, vector, k or self.slot_top_k, candidates)
        return dict(zip(idx.tolist(), scores.tolist()))

    def _rescan(self, subspace: str, text: str, vectors: dict[str, np.ndarray],
                rows: Sequence[int]) -> dict[int, float]:
        """Score an existing candidate set against another subspace, without re-filtering.

        Distinct from ``_scan`` in that nothing is dropped: this adds a second opinion about
        candidates already selected, which is how a verb constrains a set chosen by its endpoints.
        """
        vector = vectors.get(text)
        if vector is None or not len(rows):
            return {}
        rows_arr = np.asarray(list(rows), dtype=np.int64)
        scores = self.graph.vectors[subspace][rows_arr] @ vector
        return dict(zip(rows_arr.tolist(), scores.tolist()))

    def match_pattern(self, pattern: Pattern, vectors: dict[str, np.ndarray],
                      query_vector: np.ndarray | None = None) -> dict[int, float]:
        """Resolve one pattern to scored triplet rows.

        Dispatch is on ``pattern.shape``. Every branch answers the same two questions: which rows are
        *eligible* (type filters), and what should they be *scored against* (subspaces).
        """
        g = self.graph
        h, r, t = pattern.head_kind, pattern.relation_kind, pattern.tail_kind
        head, verb, tail = pattern.head, pattern.relation, pattern.tail
        C, T, W = SlotKind.CONCRETE, SlotKind.TYPE, SlotKind.WILDCARD

        # Eligibility: a TYPE slot restricts that endpoint by exact type match.
        head_pool = g.candidates_for_type(head, endpoint="head") if h is T else None
        tail_pool = g.candidates_for_type(tail, endpoint="tail") if t is T else None
        if h is T and t is T:
            # Both endpoints typed: intersect. Cheap, and often a very small set.
            pool = np.intersect1d(head_pool, tail_pool, assume_unique=False)
        else:
            pool = head_pool if head_pool is not None else tail_pool

        # -- both endpoints typed ------------------------------------------
        if h is T and t is T:
            if pool.size == 0:
                return {}
            if r is C:
                # The verb is the only text available to rank by.
                scored = self._scan("verb", verb, vectors, pool, self.per_pattern)
            elif query_vector is not None:
                # No text in the pattern at all (T?T). Rank the typed pool by the original question
                # against whole facts -- the types have already done the structural filtering, and
                # ranking by nothing would return an arbitrary slice of a large pool.
                idx, scores = g.topk("full", query_vector, self.per_pattern, pool)
                scored = dict(zip(idx.tolist(), scores.tolist()))
            else:
                return {}
            return scored

        # -- one endpoint typed, the other concrete ------------------------
        if h is T and t is C:
            base = self._scan("tail", tail, vectors, pool)
            if r is C:
                # The verb+tail composite is the query-side analogue of the stored fact's right half.
                return self._combine(base, self._rescan("full", f"{verb} {tail}", vectors, base))
            return base

        if h is C and t is T:
            base = self._scan("head", head, vectors, pool)
            if r is C:
                return self._combine(base, self._rescan("full", f"{head} {verb}", vectors, base))
            return base

        # -- one endpoint typed, the other wildcard ------------------------
        if h is T and t is W:
            if pool.size == 0 or r is not C:
                return {}
            return self._scan("verb", verb, vectors, pool, self.per_pattern)

        if h is W and t is T:
            if pool.size == 0 or r is not C:
                return {}
            return self._scan("verb", verb, vectors, pool, self.per_pattern)

        # -- both endpoints concrete ---------------------------------------
        if h is C and t is C:
            if r is C:
                # Fully specified: the whole fact is the right comparison unit, because the endpoints
                # and the verb are only jointly meaningful.
                return self._scan("full", f"{head} {verb} {tail}", vectors, None, self.per_pattern)
            # No verb given. Require a fact to match on BOTH endpoints (intersection), and separately
            # allow the joint pair subspace to propose facts the per-endpoint scans missed.
            head_scores = self._scan("head", head, vectors)
            tail_scores = self._scan("tail", tail, vectors)
            both = set(head_scores) & set(tail_scores)
            intersected = {i: (head_scores[i] + tail_scores[i]) / 2 for i in both}
            paired = self._scan("pair", f"{head} {tail}", vectors, None, self.per_pattern)
            return self._combine(intersected, paired)

        # -- one endpoint concrete, the other wildcard ---------------------
        if h is C and t is W:
            base = self._scan("head", head, vectors)
            if r is C:
                return self._combine(base, self._rescan("full", f"{head} {verb}", vectors, base))
            return base

        if h is W and t is C:
            base = self._scan("tail", tail, vectors)
            if r is C:
                return self._combine(base, self._rescan("full", f"{verb} {tail}", vectors, base))
            return base

        # -- verb only (?, verb, ?) ----------------------------------------
        if r is C:
            return self._scan("verb", verb, vectors, None, self.per_pattern)

        return {}

    # -- step 3: aggregate -------------------------------------------------
    @staticmethod
    def _aggregator(mode: str) -> Callable[[list[float]], float]:
        if mode == "max":
            return max
        if mode.startswith("sum"):
            n = int(mode[3:] or 0) or None
            def sum_top(scores: list[float]) -> float:
                ordered = sorted(scores, reverse=True)
                return float(sum(ordered[:n] if n else ordered))
            return sum_top
        raise ValueError(f"unknown aggregation {mode!r}; use 'max' or 'sum<N>' (e.g. 'sum3')")

    def search(
        self,
        query: str,
        *,
        patterns: Sequence[Pattern] | None = None,
        top_k: int = 10,
        aggregation: str = "max",
        model: str | None = None,
    ) -> list[DocumentHit]:
        """Retrieve documents for a question.

        ``patterns`` may be supplied to skip the LLM call -- useful for evaluation over a cached set
        of generated patterns, and for the no-LLM ablation (pass ``[]`` to use only the raw-query
        channel).
        """
        if patterns is None:
            patterns = self.generate_patterns(query, model=model)

        needed: set[str] = {query}
        for pattern in patterns:
            needed |= pattern.texts()
        vectors = self._vectors_for(sorted(needed))
        query_vector = vectors.get(query)

        # Keep the best score per fact, and remember which pattern produced it, so provenance
        # reported to the user is the reason it actually ranked rather than a coincidental match.
        best: dict[int, tuple[float, str, str]] = {}

        def offer(rows: dict[int, float], shape: str, label: str) -> None:
            # Bound each pattern's contribution before it enters the shared pool.
            top = sorted(rows.items(), key=lambda kv: -kv[1])[:self.per_pattern]
            for idx, score in top:
                current = best.get(idx)
                if current is None or score > current[0]:
                    best[idx] = (score, shape, label)

        for pattern in patterns:
            offer(self.match_pattern(pattern, vectors, query_vector), pattern.shape, str(pattern))

        if self.include_query_channel and query_vector is not None:
            idx, scores = self.graph.topk("full", query_vector, self.query_channel_k)
            offer(dict(zip(idx.tolist(), scores.tolist())), "query", f"raw query: {query}")

        if not best:
            return []

        # Fact scores -> document scores.
        by_doc: dict[str, list[TripletHit]] = defaultdict(list)
        for idx, (score, shape, label) in best.items():
            triplet = self.graph.triplets[idx]
            by_doc[triplet.doc_id].append(
                TripletHit(index=idx, score=score, triplet=triplet, shape=shape, pattern=label))

        combine = self._aggregator(aggregation)
        hits = []
        for doc_id, facts in by_doc.items():
            facts.sort(key=lambda f: -f.score)
            hits.append(DocumentHit(doc_id=doc_id,
                                    score=combine([f.score for f in facts]),
                                    supporting=facts))
        hits.sort(key=lambda d: -d.score)
        return hits[:top_k]


__all__ = ["DocumentHit", "SoftRetriever", "TripletHit"]
