#!/usr/bin/env python
"""Benchmark soft-KG retrieval against lexical and dense baselines on a BEIR-format dataset.

    # 1. cache the LLM-generated soft patterns once (needs an API key)
    python evaluation/run_benchmark.py --data data/nfcorpus --graph data/graph \
        --generate-patterns --patterns data/patterns_typed.json --prompt typed

    # 2. evaluate -- no API access needed from here on
    python evaluation/run_benchmark.py --data data/nfcorpus --graph data/graph \
        --patterns data/patterns_typed.json --systems soft_kg,soft_kg_nollm,bm25,dense,hybrid

Systems
-------
  soft_kg        soft patterns + raw-query channel (the full method)
  soft_kg_nollm  raw-query channel only -- isolates what the LLM pattern layer contributes
  bm25           lexical baseline over the same documents
  dense          whole-document embeddings from the *same encoder* the graph uses
  hybrid         reciprocal-rank fusion of soft_kg + bm25 + dense

Two controls this harness insists on, because leaving either out makes the headline number
uninterpretable:

**The dense baseline uses the same encoder as the graph.** Otherwise a comparison conflates the
retrieval structure with the embedding model, and any gain may be nothing but a better encoder.

**Restricted vs full frame.** Extraction may not cover every document in the corpus. Scoring the graph
on the whole corpus penalises it for documents it was never given, while scoring the baselines only on
the extracted subset flatters the graph by deleting distractors. Both frames are reported:
``restricted`` (documents that have triplets; like-for-like) and ``full`` (the whole corpus;
deployment-honest). A conclusion that holds in only one frame is a conclusion about the frame.

An ``oracle`` pseudo-system is also available: a perfect ranking restricted to documents the graph can
reach. It measures the ceiling imposed by extraction coverage, which separates "the retrieval
mechanism is weak" from "the graph never saw the answer" -- two problems with completely different
fixes.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import evaluate_run, format_table  # noqa: E402
from significance import format_comparisons, paired_bootstrap  # noqa: E402
from softkg import Embedder, LLMClient, SoftRetriever, parse_patterns, shape_summary  # noqa: E402
from softkg.graph import SoftKnowledgeGraph  # noqa: E402
from softkg.prompts import SEARCH_PROMPTS  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s", level=logging.INFO)
logger = logging.getLogger("benchmark")

RRF_K = 60  # standard reciprocal-rank-fusion damping constant


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_beir(data_dir: Path, split: str) -> tuple[dict, dict, dict]:
    """Load a BEIR-layout dataset: corpus.jsonl, queries.jsonl, qrel_<split>.tsv."""
    corpus = {}
    for line in io.open(data_dir / "corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}

    queries = {}
    for line in io.open(data_dir / "queries.jsonl", encoding="utf-8"):
        d = json.loads(line)
        queries[d["_id"]] = d["text"]

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    qrel_path = data_dir / f"qrel_{split}.tsv"
    if not qrel_path.exists():
        qrel_path = data_dir / "qrels" / f"{split}.tsv"
    with io.open(qrel_path, encoding="utf-8") as fh:
        first = fh.readline()
        # BEIR ships a header row; a file without one must not lose its first judgement.
        if "query" not in first.lower():
            fh.seek(0)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qrels[parts[0]][parts[1]] = int(parts[2])
    return corpus, queries, dict(qrels)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
def run_soft_kg(retriever: SoftRetriever, queries: dict, qids: list[str],
                patterns_by_qid: dict[str, list], depth: int, aggregation: str) -> dict:
    run = {}
    t0 = time.time()
    for n, qid in enumerate(qids, 1):
        hits = retriever.search(queries[qid], patterns=patterns_by_qid.get(qid, []),
                                top_k=depth, aggregation=aggregation)
        run[qid] = {h.doc_id: h.score for h in hits}
        if n % 50 == 0:
            logger.info("  soft_kg %d/%d (%.1fs)", n, len(qids), time.time() - t0)
    return run


def run_bm25(corpus: dict, doc_ids: list[str], queries: dict, qids: list[str],
             depth: int) -> dict:
    from rank_bm25 import BM25Okapi

    def tokenize(text: str) -> list[str]:
        return [t for t in "".join(
            c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 1]

    logger.info("bm25: indexing %d documents", len(doc_ids))
    index = BM25Okapi([tokenize(f"{corpus[d]['title']} {corpus[d]['text']}") for d in doc_ids])
    ids = np.array(doc_ids)
    run = {}
    for qid in qids:
        scores = index.get_scores(tokenize(queries[qid]))
        top = np.argsort(-scores)[:depth]
        run[qid] = {ids[i]: float(scores[i]) for i in top if scores[i] > 0}
    return run


def run_dense(embedder: Embedder, corpus: dict, doc_ids: list[str], queries: dict,
              qids: list[str], depth: int) -> dict:
    logger.info("dense: embedding %d documents with the graph's encoder", len(doc_ids))
    doc_vectors = embedder.encode_documents(
        [f"{corpus[d]['title']} {corpus[d]['text']}".strip() for d in doc_ids])
    query_vectors = embedder.encode_queries([queries[q] for q in qids])
    ids = np.array(doc_ids)
    run = {}
    for qid, qv in zip(qids, query_vectors):
        scores = doc_vectors @ qv
        top = np.argsort(-scores)[:depth]
        run[qid] = {ids[i]: float(scores[i]) for i in top}
    return run


def run_oracle(qrels: dict, qids: list[str], reachable: set[str], depth: int) -> dict:
    """Perfect ranking of the relevant documents the graph can actually reach.

    This is a ceiling, not a system. It answers: if pattern matching were flawless, how well could
    this graph possibly score? A low oracle means extraction coverage is the binding constraint and no
    amount of retrieval tuning will help.
    """
    run = {}
    for qid in qids:
        rel = {d: g for d, g in qrels.get(qid, {}).items() if g > 0 and d in reachable}
        ordered = sorted(rel.items(), key=lambda kv: -kv[1])[:depth]
        run[qid] = {d: float(len(ordered) - i) for i, (d, _) in enumerate(ordered)}
    return run


def fuse_rrf(runs: list[dict], qids: list[str], depth: int) -> dict:
    """Reciprocal rank fusion. Rank-based, so it needs no score calibration between systems."""
    fused = {}
    for qid in qids:
        totals: dict[str, float] = defaultdict(float)
        for run in runs:
            ranked = sorted(run.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            for rank, (doc_id, _) in enumerate(ranked, 1):
                totals[doc_id] += 1.0 / (RRF_K + rank)
        fused[qid] = dict(sorted(totals.items(), key=lambda kv: -kv[1])[:depth])
    return fused


# ---------------------------------------------------------------------------
# Pattern generation
# ---------------------------------------------------------------------------
def generate_patterns(llm: LLMClient, prompt: str, queries: dict, qids: list[str],
                      cache_path: Path, model: str) -> dict[str, str]:
    """Generate and cache one pattern reply per query. Resumable."""
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("pattern cache: %d replies already present", len(cache))

    todo = [q for q in qids if not (cache.get(q) or "").strip()]
    logger.info("generating patterns for %d queries", len(todo))
    for n, qid in enumerate(todo, 1):
        try:
            cache[qid] = llm.chat(prompt, queries[qid], model=model)
        except Exception as exc:
            logger.warning("  %s failed: %s", qid, exc)
            continue
        # Write through on every reply: these calls cost money, and an interrupted run should never
        # have to pay for them twice.
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        if n % 10 == 0:
            logger.info("  %d/%d", n, len(todo))
    return cache


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="BEIR-format dataset directory")
    ap.add_argument("--split", default="test")
    ap.add_argument("--graph", default=ROOT / "data" / "graph", type=Path)
    ap.add_argument("--cache-dir", default=ROOT / "data" / "emb_cache", type=Path)
    ap.add_argument("--patterns", type=Path, default=None,
                    help="JSON cache of {qid: pattern reply}")
    ap.add_argument("--generate-patterns", action="store_true",
                    help="call the LLM to fill the pattern cache, then exit")
    ap.add_argument("--prompt", default="typed", choices=sorted(SEARCH_PROMPTS))
    ap.add_argument("--systems", default="soft_kg,soft_kg_nollm,bm25,dense",
                    help="comma-separated: soft_kg, soft_kg_nollm, bm25, dense, hybrid, oracle")
    ap.add_argument("--frames", default="restricted,full",
                    help="comma-separated: restricted, full")
    ap.add_argument("--depth", type=int, default=100, help="results kept per query")
    ap.add_argument("--aggregation", default="max", help="'max' or 'sum<N>'")
    ap.add_argument("--metric", default="nDCG@10", help="metric for significance testing")
    ap.add_argument("--out", default=ROOT / "evaluation" / "results", type=Path)
    ap.add_argument("--model", default=os.environ.get("SOFTKG_MODEL", "gpt-5-mini"))
    ap.add_argument("--endpoint", default=os.environ.get("SOFTKG_ENDPOINT",
                                                        "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("SOFTKG_API_KEY", ""))
    args = ap.parse_args()

    corpus, queries, qrels = load_beir(args.data, args.split)
    qids = sorted(qrels)
    logger.info("dataset: %d documents, %d %s queries", len(corpus), len(qids), args.split)

    # -- pattern generation mode ------------------------------------------
    if args.generate_patterns:
        if not args.patterns:
            logger.error("--generate-patterns requires --patterns")
            return 2
        if not args.api_key:
            logger.error("--generate-patterns requires an API key")
            return 2
        args.patterns.parent.mkdir(parents=True, exist_ok=True)
        llm = LLMClient(api_key=args.api_key, endpoint=args.endpoint, default_model=args.model)
        cache = generate_patterns(llm, SEARCH_PROMPTS[args.prompt], queries, qids,
                                  args.patterns, args.model)
        parsed = {q: parse_patterns(cache.get(q, "")) for q in qids}
        print("\nPattern generation summary")
        for key, value in shape_summary(parsed).items():
            print(f"  {key:26} {value}")
        return 0

    # -- load graph --------------------------------------------------------
    embedder = Embedder(cache_dir=args.cache_dir)
    graph = SoftKnowledgeGraph.load(args.graph, embedder)
    reachable = graph.doc_ids
    logger.info("graph: %d triplets over %d documents (%.1f%% of the corpus)",
                len(graph), len(reachable), 100 * len(reachable) / max(1, len(corpus)))

    patterns_by_qid: dict[str, list] = {}
    if args.patterns and args.patterns.exists():
        cache = json.loads(args.patterns.read_text(encoding="utf-8"))
        patterns_by_qid = {q: parse_patterns(cache.get(q, "")) for q in qids}
        summary = shape_summary(patterns_by_qid)
        logger.info("patterns: %.2f/query, %.1f%% typed slots, %.1f%% of queries all-degenerate",
                    summary["patterns_per_query"], 100 * summary["typed_slot_rate"],
                    100 * summary["degenerate_rate"])
    elif "soft_kg" in args.systems:
        logger.warning("no pattern cache -- soft_kg will fall back to the raw-query channel, "
                       "which makes it identical to soft_kg_nollm")

    wanted = [s.strip() for s in args.systems.split(",") if s.strip()]
    frames = [f.strip() for f in args.frames.split(",") if f.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    # -- build runs --------------------------------------------------------
    runs: dict[str, dict] = {}
    retriever = SoftRetriever(graph, embedder, llm=None)

    if "soft_kg" in wanted:
        runs["soft_kg"] = run_soft_kg(retriever, queries, qids, patterns_by_qid,
                                      args.depth, args.aggregation)
    if "soft_kg_nollm" in wanted:
        runs["soft_kg_nollm"] = run_soft_kg(retriever, queries, qids, {},
                                            args.depth, args.aggregation)

    # Baselines are built once per frame: a lexical or dense index over the extracted subset is a
    # different system from one over the whole corpus, and reusing one for both frames is the
    # like-for-like error this harness exists to avoid.
    for frame in frames:
        doc_ids = sorted(reachable) if frame == "restricted" else sorted(corpus)
        if "bm25" in wanted:
            runs[f"bm25__{frame}"] = run_bm25(corpus, doc_ids, queries, qids, args.depth)
        if "dense" in wanted:
            runs[f"dense__{frame}"] = run_dense(embedder, corpus, doc_ids, queries, qids,
                                                args.depth)
    if "oracle" in wanted:
        runs["oracle"] = run_oracle(qrels, qids, reachable, args.depth)

    if "hybrid" in wanted:
        for frame in frames:
            parts = [runs[k] for k in ("soft_kg", f"bm25__{frame}", f"dense__{frame}") if k in runs]
            if len(parts) > 1:
                runs[f"hybrid__{frame}"] = fuse_rrf(parts, qids, args.depth)

    # -- evaluate ----------------------------------------------------------
    summary: dict[str, dict] = {}
    per_query: dict[str, dict] = {}

    for frame in frames:
        # In the restricted frame, judgements pointing outside the extracted subset are removed for
        # every system alike, so nobody is credited or penalised for documents not in play.
        if frame == "restricted":
            frame_qrels = {q: {d: g for d, g in rel.items() if d in reachable}
                           for q, rel in qrels.items()}
            frame_qrels = {q: rel for q, rel in frame_qrels.items()
                           if any(g > 0 for g in rel.values())}
        else:
            frame_qrels = qrels

        rows = []
        for name, run in runs.items():
            # Skip a baseline built for the other frame.
            if "__" in name and not name.endswith(f"__{frame}"):
                continue
            label = name.split("__")[0]
            mean, detail = evaluate_run(run, frame_qrels)
            if not mean:
                continue
            rows.append((label, mean))
            summary[f"{frame}/{label}"] = mean
            per_query[f"{frame}/{label}"] = detail

        print(f"\n{'=' * 100}")
        print(f"FRAME: {frame}   ({len(frame_qrels)} queries with positives, "
              f"{len(reachable) if frame == 'restricted' else len(corpus)} documents)")
        print("=" * 100)
        print(format_table(rows))

        # -- significance --------------------------------------------------
        comparisons = []
        names = [label for label, _ in rows if label != "oracle"]
        if "soft_kg" in names:
            for other in ("soft_kg_nollm", "bm25", "dense"):
                if other in names:
                    comparisons.append(("soft_kg", other))
        if "hybrid" in names and "dense" in names:
            comparisons.append(("hybrid", "dense"))
        if "dense" in names and "bm25" in names:
            comparisons.append(("dense", "bm25"))

        stat_rows = []
        for a, b in comparisons:
            pa = {q: m[args.metric] for q, m in per_query[f"{frame}/{a}"].items()}
            pb = {q: m[args.metric] for q, m in per_query[f"{frame}/{b}"].items()}
            stat_rows.append((a, b, paired_bootstrap(pa, pb)))
        if stat_rows:
            print()
            print(format_comparisons(stat_rows, args.metric))

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, run in runs.items():
        (args.out / f"run_{name}.json").write_text(json.dumps(run), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
