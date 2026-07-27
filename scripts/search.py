#!/usr/bin/env python
"""Query the soft knowledge graph and show why each document was retrieved.

    # let the LLM write the patterns
    python scripts/search.py --graph data/graph -q "what food lowers cholesterol"

    # supply patterns yourself -- no LLM needed, useful for probing the graph directly
    python scripts/search.py --graph data/graph -q "..." \
        --pattern "MEM( FOOD ^ lowers ^ cholesterol )" \
        --pattern "MEM( cholesterol ^ lowered by ^ FOOD )"

    # ablation: no patterns at all, raw question against the fact index
    python scripts/search.py --graph data/graph -q "..." --no-patterns

Each result prints the extracted facts that caused it to rank, which is the practical argument for a
fact-level index: the provenance is a property of the retrieval, not a post-hoc explanation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from softkg import Embedder, LLMClient, SoftRetriever, parse_patterns  # noqa: E402
from softkg.graph import SoftKnowledgeGraph  # noqa: E402
from softkg.prompts import SEARCH_PROMPTS  # noqa: E402

logging.basicConfig(format="%(levelname)-7s %(message)s", level=logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=ROOT / "data" / "graph", type=Path)
    ap.add_argument("--cache-dir", default=ROOT / "data" / "emb_cache", type=Path)
    ap.add_argument("-q", "--query", required=True)
    ap.add_argument("--pattern", action="append", default=[],
                    help="supply a MEM(...) pattern explicitly; repeatable")
    ap.add_argument("--no-patterns", action="store_true",
                    help="raw-query channel only (the no-LLM ablation)")
    ap.add_argument("--prompt", default="typed", choices=sorted(SEARCH_PROMPTS),
                    help="which query-rewriting prompt to use (default: typed)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--aggregation", default="max",
                    help="'max' or 'sum<N>', e.g. sum3 (default: max)")
    ap.add_argument("--facts", type=int, default=3, help="supporting facts to show per document")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="optional JSONL corpus, to print retrieved titles")
    ap.add_argument("--model", default=os.environ.get("SOFTKG_MODEL", "gpt-5-mini"))
    ap.add_argument("--endpoint", default=os.environ.get("SOFTKG_ENDPOINT",
                                                        "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("SOFTKG_API_KEY", ""))
    ap.add_argument("--json", action="store_true", help="emit JSON instead of formatted text")
    args = ap.parse_args()

    if not (args.graph / "triplets.jsonl").exists():
        print(f"no graph at {args.graph} -- run scripts/build_graph.py first", file=sys.stderr)
        return 2

    embedder = Embedder(cache_dir=args.cache_dir)
    graph = SoftKnowledgeGraph.load(args.graph, embedder)
    if not graph.is_built:
        print(f"graph at {args.graph} has no saved vectors -- rebuild it", file=sys.stderr)
        return 2

    # An LLM is only constructed when patterns actually have to be generated.
    llm = None
    if not args.no_patterns and not args.pattern:
        if not args.api_key:
            print("pattern generation needs an API key (--api-key / SOFTKG_API_KEY), or pass "
                  "--pattern / --no-patterns", file=sys.stderr)
            return 2
        llm = LLMClient(api_key=args.api_key, endpoint=args.endpoint, default_model=args.model)

    retriever = SoftRetriever(graph, embedder, llm=llm,
                              search_prompt=SEARCH_PROMPTS[args.prompt])

    if args.no_patterns:
        patterns = []
    elif args.pattern:
        patterns = parse_patterns("\n".join(args.pattern))
        if not patterns:
            print("none of the supplied --pattern values parsed", file=sys.stderr)
            return 2
    else:
        patterns = retriever.generate_patterns(args.query, model=args.model)

    hits = retriever.search(args.query, patterns=patterns, top_k=args.top_k,
                            aggregation=args.aggregation)

    titles: dict[str, str] = {}
    if args.corpus and args.corpus.exists():
        wanted = {h.doc_id for h in hits}
        for line in args.corpus.open(encoding="utf-8"):
            record = json.loads(line)
            if str(record.get("_id")) in wanted:
                titles[str(record["_id"])] = (record.get("title") or "").strip()

    if args.json:
        print(json.dumps({
            "query": args.query,
            "patterns": [{"pattern": str(p), "shape": p.shape} for p in patterns],
            "results": [{
                "doc_id": h.doc_id, "score": h.score, "title": titles.get(h.doc_id),
                "facts": [{"score": f.score, "fact": str(f.triplet), "via": f.pattern}
                          for f in h.supporting[:args.facts]],
            } for h in hits],
        }, indent=2, ensure_ascii=False))
        return 0

    print(f"\nQuery: {args.query}")
    if patterns:
        print(f"\nSoft patterns ({len(patterns)}):")
        for p in patterns:
            print(f"  [{p.shape}] {p}")
    else:
        print("\nSoft patterns: none (raw-query channel only)")

    print(f"\nTop {len(hits)} documents (aggregation={args.aggregation}):")
    for rank, hit in enumerate(hits, 1):
        title = titles.get(hit.doc_id)
        header = f"{hit.doc_id}" + (f" -- {title}" if title else "")
        print(f"\n{rank:2}. {header}   score={hit.score:.4f}")
        for fact in hit.supporting[:args.facts]:
            print(f"      [{fact.score:.3f}] {fact.triplet}")
            print(f"              via [{fact.shape}] {fact.pattern}")
        remaining = len(hit.supporting) - args.facts
        if remaining > 0:
            print(f"      ... and {remaining} more supporting facts")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
