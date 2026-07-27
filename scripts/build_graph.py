#!/usr/bin/env python
"""Build and persist the soft knowledge graph from extracted triplets.

Embeds all five subspaces and saves them alongside the triplets, so retrieval can open the graph
without touching the encoder.

    python scripts/build_graph.py --db data/checkpoint.db --out data/graph
    python scripts/build_graph.py --triplets data/triplets.json --out data/graph

Re-runnable: the embedding cache means a rebuild after adding documents only encodes the new strings.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from softkg import Embedder, build_graph, triplets_from_checkpoint, triplets_from_json  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s", level=logging.INFO)
logger = logging.getLogger("build_graph")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", type=Path, help="extraction checkpoint database")
    source.add_argument("--triplets", type=Path, help="triplet JSON export")
    ap.add_argument("--out", default=ROOT / "data" / "graph", type=Path,
                    help="output directory (default: data/graph)")
    ap.add_argument("--cache-dir", default=ROOT / "data" / "emb_cache", type=Path,
                    help="embedding cache (default: data/emb_cache)")
    ap.add_argument("--model", default="jinaai/jina-embeddings-v3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="e.g. cuda -- defaults to CPU")
    ap.add_argument("--keep-chunk-ids", action="store_true",
                    help="do not fold DOC__n chunk ids back to DOC (retrieval becomes chunk-level)")
    ap.add_argument("--stats-only", action="store_true",
                    help="report graph statistics without embedding anything")
    args = ap.parse_args()

    if args.db:
        triplets = triplets_from_checkpoint(args.db, strip_chunk_suffix=not args.keep_chunk_ids)
    else:
        triplets = triplets_from_json(args.triplets)
    if not triplets:
        logger.error("no triplets found -- has extraction run?")
        return 2

    embedder = Embedder(args.model, cache_dir=args.cache_dir, device=args.device,
                        batch_size=args.batch_size)
    graph = build_graph(triplets, embedder, build_vectors=not args.stats_only)

    info = graph.describe()
    print("\nGraph")
    print("-----")
    for key, value in info.items():
        if key == "types":
            print(f"  {key}")
            total = sum(value.values())
            for name, count in value.items():
                print(f"      {name:20} {count:7}  {100 * count / total:5.1f}%")
        elif isinstance(value, float):
            print(f"  {key:24} {value:.2f}")
        else:
            print(f"  {key:24} {value}")

    if args.stats_only:
        return 0

    graph.save(args.out)
    (args.out / "stats.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nsaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
