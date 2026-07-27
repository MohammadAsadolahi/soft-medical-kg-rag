"""The soft knowledge graph: typed triplets indexed across several vector subspaces.

A conventional KG stores one canonical form per node and matches it exactly. This graph instead
projects each triplet into **five** separate vector subspaces and keeps its type labels as exact
metadata:

    head    "cholesterol"                          -- head entity alone
    tail    "cardiovascular disease"               -- tail entity alone
    pair    "cholesterol cardiovascular disease"   -- both endpoints, verb ignored
    verb    "increases risk of"                    -- relation alone
    full    "cholesterol increases risk of ..."    -- the whole fact as a sentence

Why five rather than one. A pattern constrains different parts of a triplet depending on its shape,
and the right thing to compare against changes accordingly. Asking "what FOOD lowers cholesterol"
should match on the *tail* entity while filtering heads by type -- scoring that against a whole-fact
embedding drags in the verb and the other endpoint as noise. Conversely a fully specified pattern is
best matched against the whole fact at once, because the endpoints are only jointly meaningful. One
undifferentiated index cannot serve both; it forces every query shape through the same comparison and
throws away the structure extraction just paid for.

The type labels stay exact. Two nodes typed DISEASE_DISORDER are *the same kind of thing* by
construction, with no threshold to tune, which is what lets a query say "any disease" and mean it.
Softness is confined to surface forms, where it is needed; the closed part of the schema stays hard.

Similarity is cosine, computed as a dot product over L2-normalised vectors (see ``embedder.py``).
Brute-force matrix multiplication is used rather than an ANN index: at corpus scale here (tens of
thousands of triplets, 1024 dimensions) a full scan is milliseconds, exact, and free of the recall
loss and tuning surface an approximate index adds. ``search`` accepts a ``top_k`` per subspace, so
swapping in FAISS later is a localised change to the two ``_topk`` helpers.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .embedder import Embedder
from .schema import QUERY_TYPES, normalize_type

logger = logging.getLogger(__name__)

# The five graph-side subspaces. Order is fixed because it is used as an array axis.
SUBSPACES: tuple[str, ...] = ("head", "tail", "pair", "verb", "full")


@dataclass(frozen=True)
class Triplet:
    """One typed fact, with the document it came from."""

    head: str
    head_type: str
    verb: str
    tail: str
    tail_type: str
    doc_id: str

    @property
    def sentence(self) -> str:
        """The fact as a single string -- what the ``full`` subspace indexes."""
        return " ".join(p for p in (self.head, self.verb, self.tail) if p)

    @property
    def pair(self) -> str:
        return f"{self.head} {self.tail}"

    def __str__(self) -> str:
        return (f"({self.head} : {self.head_type}) --[{self.verb}]--> "
                f"({self.tail} : {self.tail_type})")


class SoftKnowledgeGraph:
    """Typed triplets plus their five vector subspaces and type indices."""

    def __init__(self, triplets: Sequence[Triplet], embedder: Embedder) -> None:
        if not triplets:
            raise ValueError("cannot build a graph from zero triplets")
        self.triplets: list[Triplet] = list(triplets)
        self.embedder = embedder
        self.vectors: dict[str, np.ndarray] = {}

        # type -> row indices, for each endpoint. Precomputed as arrays because these are used as
        # candidate masks on every typed pattern and rebuilding them per query dominates the cost.
        self.head_type_index: dict[str, np.ndarray] = {}
        self.tail_type_index: dict[str, np.ndarray] = {}
        self._build_type_indices()

    # -- construction ------------------------------------------------------
    def _build_type_indices(self) -> None:
        heads: dict[str, list[int]] = defaultdict(list)
        tails: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.triplets):
            heads[t.head_type].append(i)
            tails[t.tail_type].append(i)
        self.head_type_index = {k: np.asarray(v, dtype=np.int64) for k, v in heads.items()}
        self.tail_type_index = {k: np.asarray(v, dtype=np.int64) for k, v in tails.items()}

    def build_vectors(self) -> None:
        """Embed all five subspaces. Idempotent, and cheap once the embedder cache is warm."""
        texts = {
            "head": [t.head for t in self.triplets],
            "tail": [t.tail for t in self.triplets],
            "pair": [t.pair for t in self.triplets],
            "verb": [t.verb for t in self.triplets],
            "full": [t.sentence for t in self.triplets],
        }
        for name in SUBSPACES:
            if name in self.vectors:
                continue
            logger.info("embedding subspace %-5s (%d strings)", name, len(texts[name]))
            self.vectors[name] = self.embedder.encode_nodes(texts[name])

    @property
    def is_built(self) -> bool:
        return all(name in self.vectors for name in SUBSPACES)

    def __len__(self) -> int:
        return len(self.triplets)

    @property
    def doc_ids(self) -> set[str]:
        return {t.doc_id for t in self.triplets}

    # -- similarity primitives --------------------------------------------
    def similarity(self, subspace: str, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query vector against every triplet in a subspace."""
        if subspace not in self.vectors:
            raise RuntimeError(f"subspace {subspace!r} not built -- call build_vectors() first")
        return self.vectors[subspace] @ query_vector

    def topk(self, subspace: str, query_vector: np.ndarray, k: int,
             candidates: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-k rows of a subspace, optionally restricted to ``candidates``.

        Returns (row indices into the full triplet list, their scores), highest first. Restricting
        *before* scoring is what makes typed patterns cheap: a DISEASE_DISORDER filter often cuts the
        candidate set by an order of magnitude, and the similarity is then only computed there.
        """
        if candidates is not None:
            if candidates.size == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            scores = self.vectors[subspace][candidates] @ query_vector
            k = min(k, candidates.size)
            local = np.argpartition(-scores, k - 1)[:k]
            local = local[np.argsort(-scores[local])]
            return candidates[local], scores[local]

        scores = self.similarity(subspace, query_vector)
        k = min(k, scores.size)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def candidates_for_type(self, type_name: str, *, endpoint: str) -> np.ndarray:
        """Row indices whose head or tail carries a given type."""
        index = self.head_type_index if endpoint == "head" else self.tail_type_index
        return index.get(normalize_type(type_name), np.empty(0, dtype=np.int64))

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        """Persist triplets and vectors so a graph can be reopened without re-embedding."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "triplets.jsonl").open("w", encoding="utf-8") as fh:
            for t in self.triplets:
                fh.write(json.dumps({
                    "head": t.head, "head_type": t.head_type, "verb": t.verb,
                    "tail": t.tail, "tail_type": t.tail_type, "doc_id": t.doc_id,
                }, ensure_ascii=False) + "\n")
        if self.vectors:
            np.savez_compressed(directory / "vectors.npz", **self.vectors)
        logger.info("graph saved to %s (%d triplets, vectors=%s)",
                    directory, len(self.triplets), bool(self.vectors))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> "SoftKnowledgeGraph":
        directory = Path(directory)
        triplets = []
        with (directory / "triplets.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                triplets.append(Triplet(**d))
        graph = cls(triplets, embedder)
        vectors_path = directory / "vectors.npz"
        if vectors_path.exists():
            with np.load(vectors_path) as data:
                graph.vectors = {k: data[k] for k in data.files}
            # A vector file from a different triplet list would misattribute every score, and the
            # symptom would be subtly wrong rankings rather than an error. Check the shape.
            for name, matrix in graph.vectors.items():
                if matrix.shape[0] != len(triplets):
                    raise RuntimeError(
                        f"{vectors_path}: subspace {name!r} has {matrix.shape[0]} rows for "
                        f"{len(triplets)} triplets -- the saved vectors do not match the triplets.")
        return graph

    # -- reporting ---------------------------------------------------------
    def describe(self) -> dict[str, object]:
        entities = {t.head.lower() for t in self.triplets} | {t.tail.lower() for t in self.triplets}
        type_counts: dict[str, int] = defaultdict(int)
        for t in self.triplets:
            type_counts[t.head_type] += 1
            type_counts[t.tail_type] += 1
        return {
            "triplets": len(self.triplets),
            "documents": len(self.doc_ids),
            "triplets_per_document": len(self.triplets) / max(1, len(self.doc_ids)),
            "unique_entities": len(entities),
            "unique_verbs": len({t.verb.lower() for t in self.triplets if t.verb}),
            "types": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
            "vectors_built": self.is_built,
        }


# ---------------------------------------------------------------------------
# Loading triplets
# ---------------------------------------------------------------------------
def triplets_from_records(records: Iterable[tuple[str, list[dict]]]) -> list[Triplet]:
    """Convert (doc_id, raw triplet dicts) into validated Triplets.

    Endpoint-less triplets are dropped: a triplet with no head or no tail cannot be placed in any
    subspace, so keeping it would only add an unreachable row. Types are normalised here rather than
    at extraction time so that a graph can be rebuilt with an updated alias table without re-running
    any LLM calls.
    """
    out: list[Triplet] = []
    for doc_id, raw in records:
        for t in raw:
            head = (t.get("e1") or t.get("head") or "").strip()
            tail = (t.get("e2") or t.get("tail") or "").strip()
            if not head or not tail:
                continue
            out.append(Triplet(
                head=head,
                head_type=normalize_type(t.get("e1_type") or t.get("head_type")),
                verb=(t.get("verb") or "").strip(),
                tail=tail,
                tail_type=normalize_type(t.get("e2_type") or t.get("tail_type")),
                doc_id=doc_id,
            ))
    return out


def triplets_from_checkpoint(db_path: str | Path, *, strip_chunk_suffix: bool = True
                             ) -> list[Triplet]:
    """Load every completed extraction from a pipeline checkpoint database.

    Long documents are chunked before extraction with ids like ``DOC__2``. ``strip_chunk_suffix``
    folds those back to the parent document id, because retrieval is evaluated at document level --
    without it, every chunk of a document competes as a separate result.
    """
    from .checkpoint import CheckpointStore

    store = CheckpointStore(db_path)
    try:
        records = []
        for doc_id, raw in store.iter_triplets():
            if strip_chunk_suffix and "__" in doc_id:
                doc_id = doc_id.rsplit("__", 1)[0]
            records.append((doc_id, raw))
    finally:
        store.close()
    return triplets_from_records(records)


def triplets_from_json(path: str | Path) -> list[Triplet]:
    """Load triplets from a JSON export.

    Accepts either ``{doc_id: [triplet, ...]}`` or ``{doc_id: {"triplets": [...]}}``, since both
    shapes show up in exports from different stages of the pipeline.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for doc_id, payload in data.items():
        raw = payload.get("triplets", []) if isinstance(payload, dict) else payload
        if isinstance(raw, list):
            records.append((doc_id, raw))
    return triplets_from_records(records)


def build_graph(triplets: Sequence[Triplet], embedder: Embedder, *,
                build_vectors: bool = True) -> SoftKnowledgeGraph:
    graph = SoftKnowledgeGraph(triplets, embedder)
    if build_vectors:
        graph.build_vectors()
    return graph


__all__ = ["QUERY_TYPES", "SUBSPACES", "SoftKnowledgeGraph", "Triplet", "build_graph",
           "triplets_from_checkpoint", "triplets_from_json", "triplets_from_records"]
