"""softkg -- soft knowledge graph construction and retrieval for biomedical documents.

Two halves that meet at a typed triplet store:

    extraction   decontextualize passages, then extract (entity:TYPE) --[verb]--> (entity:TYPE)
                 softkg.extract / softkg.llm / softkg.checkpoint / softkg.prompts
    retrieval    rewrite a question into soft graph patterns, resolve them against the graph's
                 vector subspaces and type index, rank documents by their best matching fact
                 softkg.graph / softkg.patterns / softkg.retrieve / softkg.embedder

Typical use::

    from softkg import Embedder, SoftRetriever, build_graph, triplets_from_checkpoint

    triplets = triplets_from_checkpoint("data/checkpoint.db")
    graph = build_graph(triplets, Embedder(cache_dir="data/emb_cache"))
    hits = SoftRetriever(graph, graph.embedder, llm=llm).search("what food lowers cholesterol")
    print(hits[0].explain())
"""
from __future__ import annotations

__version__ = "0.1.0"

from .checkpoint import CheckpointStore
from .embedder import Embedder
from .extract import ExtractionPipeline
from .graph import (SoftKnowledgeGraph, Triplet, build_graph, triplets_from_checkpoint,
                    triplets_from_json, triplets_from_records)
from .llm import LLMClient
from .patterns import Pattern, SlotKind, parse_patterns, shape_summary
from .retrieve import DocumentHit, SoftRetriever, TripletHit
from .schema import ENTITY_SCHEMA_PROSE, EXTRACTION_TYPES, QUERY_TYPES, normalize_type

__all__ = [
    "CheckpointStore",
    "DocumentHit",
    "ENTITY_SCHEMA_PROSE",
    "EXTRACTION_TYPES",
    "Embedder",
    "ExtractionPipeline",
    "LLMClient",
    "Pattern",
    "QUERY_TYPES",
    "SlotKind",
    "SoftKnowledgeGraph",
    "SoftRetriever",
    "Triplet",
    "TripletHit",
    "build_graph",
    "normalize_type",
    "parse_patterns",
    "shape_summary",
    "triplets_from_checkpoint",
    "triplets_from_json",
    "triplets_from_records",
]
