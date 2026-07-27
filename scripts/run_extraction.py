#!/usr/bin/env python
"""Build a soft knowledge graph's triplet store from a document corpus.

Reads a JSONL corpus, runs every document through the four-stage extraction pipeline, and
checkpoints each stage to SQLite. Interrupt and re-run freely -- work already committed is skipped.

    python scripts/run_extraction.py --corpus data/corpus.jsonl --db data/checkpoint.db

Corpus format is one JSON object per line with an id and text::

    {"_id": "MED-123", "title": "...", "text": "..."}

Long documents are split into chunks before extraction, because a single call asked to extract from
a 4000-word document reliably truncates its output partway through -- the failure is silent and looks
like a sparse document rather than a truncated response. Chunk ids carry a ``__n`` suffix that the
graph loader folds back to the parent id.

Set the endpoint and key via environment variables, or pass them explicitly::

    export SOFTKG_API_KEY=...
    export SOFTKG_ENDPOINT=https://api.openai.com/v1
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from softkg import CheckpointStore, ExtractionPipeline, LLMClient  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s", level=logging.INFO)
logger = logging.getLogger("run_extraction")


def chunk_text(text: str, max_words: int) -> list[str]:
    """Split on paragraph boundaries, packing up to ``max_words`` per chunk.

    Paragraphs are kept whole wherever possible: the decontextualization stages resolve references
    against preceding text, so a chunk boundary mid-paragraph destroys exactly the context those
    stages need. A single paragraph longer than the budget is emitted oversized rather than cut.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()] or [text.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and count + words > max_words:
            chunks.append("\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words
    if current:
        chunks.append("\n".join(current))
    return chunks


def load_corpus(path: Path, *, id_field: str, text_field: str, title_field: str | None,
                max_words: int) -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        doc_id = str(record[id_field])
        body = (record.get(text_field) or "").strip()
        title = (record.get(title_field) or "").strip() if title_field else ""
        # Prepend the title: it often carries the topic that the body's first sentence refers back to
        # ("This review examines ..."), which is precisely what stage 2 needs in order to resolve it.
        text = f"{title}\n{body}".strip() if title else body
        if not text:
            continue
        chunks = chunk_text(text, max_words)
        if len(chunks) == 1:
            documents.append((doc_id, chunks[0]))
        else:
            documents.extend((f"{doc_id}__{i}", chunk) for i, chunk in enumerate(chunks))
    return documents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path, help="JSONL corpus")
    ap.add_argument("--db", default=ROOT / "data" / "checkpoint.db", type=Path,
                    help="checkpoint database (created if absent)")
    ap.add_argument("--id-field", default="_id")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--title-field", default="title")
    ap.add_argument("--max-words", type=int, default=350,
                    help="chunk size in words (default: 350)")
    ap.add_argument("--model", default=os.environ.get("SOFTKG_MODEL", "gpt-5-mini"))
    ap.add_argument("--endpoint", default=os.environ.get("SOFTKG_ENDPOINT",
                                                        "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("SOFTKG_API_KEY", ""))
    ap.add_argument("--azure", action="store_true", help="endpoint is Azure OpenAI")
    ap.add_argument("--no-reasoning-effort", action="store_true",
                    help="omit reasoning_effort (required for non-reasoning models)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N documents")
    ap.add_argument("--enqueue-only", action="store_true",
                    help="load the corpus into the database and exit without calling any LLM")
    ap.add_argument("--retry-failed", action="store_true",
                    help="requeue documents previously marked failed, then run")
    ap.add_argument("--stats", action="store_true", help="print progress and exit")
    args = ap.parse_args()

    store = CheckpointStore(args.db)

    if args.stats:
        for key, value in store.stats().items():
            print(f"  {key:28} {value}")
        return 0

    if args.corpus.exists():
        documents = load_corpus(args.corpus, id_field=args.id_field, text_field=args.text_field,
                                title_field=args.title_field, max_words=args.max_words)
        added = store.enqueue(documents)
        logger.info("corpus: %d units (%d newly enqueued)", len(documents), added)
    else:
        logger.error("corpus not found: %s", args.corpus)
        return 2

    if args.retry_failed:
        logger.info("requeued %d previously failed documents", store.reset_failed())

    if args.enqueue_only:
        for key, value in store.stats().items():
            logger.info("  %-28s %s", key, value)
        return 0

    if not args.api_key:
        logger.error("no API key: pass --api-key or set SOFTKG_API_KEY")
        return 2

    llm = LLMClient(
        api_key=args.api_key,
        endpoint=args.endpoint,
        default_model=args.model,
        use_azure=args.azure,
        reasoning_effort=None if args.no_reasoning_effort else "high",
    )
    pipeline = ExtractionPipeline(llm, store, model=args.model)
    stats = pipeline.run(limit=args.limit)

    logger.info("done: %d completed, %d failed, %d triplets (%d dropped, %d rewrites rejected)",
                stats.completed, stats.failed, stats.triplets,
                stats.dropped_triplets, stats.rejected_rewrites)
    logger.info("calls per stage: %s", stats.calls)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
