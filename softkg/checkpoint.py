"""Resumable per-document checkpointing for the extraction pipeline.

A full-corpus extraction is a long, expensive, interruptible job: four LLM calls per document,
thousands of documents, hours of wall clock, and a non-zero rate of rate-limit storms and machine
sleeps. Any design that holds progress in memory and writes at the end will eventually lose a run.

So progress is stored per *stage* per *document*, committed as it happens, in SQLite:

    id | original_text | step1_pronoun_resolved | step2_references_resolved
       | step3_flattened | step4_triplets_json | status | error | attempts | updated_at

Restarting re-reads the table and skips every stage already recorded, so a crash costs at most one
call. The intermediate columns are not just bookkeeping -- they are the supervision signal for
fine-tuning. Each column pair (input stage, output stage) is a ready-made training set, which is
exactly how ``finetuning/create_training_datasets.py`` builds its data. Distilling a large model into
a small one falls out of having checkpointed honestly.

SQLite is chosen over JSON-on-disk for the atomic commit: a JSON dump interrupted mid-write is a
corrupt file, and this data costs real money to regenerate.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Iterator

# Ordered pipeline stages. Column names are also the stage identifiers used by the runner and by the
# fine-tuning dataset builder, so renaming one is a schema migration.
STEPS: tuple[str, ...] = (
    "step1_pronoun_resolved",
    "step2_references_resolved",
    "step3_flattened",
    "step4_triplets_json",
)
FINAL_STEP = STEPS[-1]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline (
    id                        TEXT PRIMARY KEY,
    original_text             TEXT NOT NULL,
    step1_pronoun_resolved    TEXT,
    step2_references_resolved TEXT,
    step3_flattened           TEXT,
    step4_triplets_json       TEXT,
    status                    TEXT NOT NULL DEFAULT 'pending',
    error                     TEXT,
    attempts                  INTEGER NOT NULL DEFAULT 0,
    updated_at                REAL
);
CREATE INDEX IF NOT EXISTS idx_status ON pipeline(status);
"""


class CheckpointStore:
    """SQLite-backed pipeline state. Safe to open concurrently from several worker processes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # WAL lets readers proceed during writes, which is what makes concurrent workers plus a live
        # progress query possible. timeout absorbs brief write contention instead of raising.
        self.conn = sqlite3.connect(str(self.path), timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- population --------------------------------------------------------
    def enqueue(self, documents: Iterable[tuple[str, str]]) -> int:
        """Register (doc_id, text) pairs. Existing rows are left untouched.

        Idempotent by design: re-running the loader after adding documents to the corpus enqueues
        only the new ones and never resets work already done.
        """
        added = 0
        with self.conn:
            for doc_id, text in documents:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO pipeline (id, original_text, status, updated_at) "
                    "VALUES (?, ?, 'pending', ?)",
                    (doc_id, text, time.time()),
                )
                added += cur.rowcount
        return added

    # -- work selection ----------------------------------------------------
    def pending_ids(self, limit: int | None = None) -> list[str]:
        """Documents that have not reached the final stage, oldest first."""
        sql = (f"SELECT id FROM pipeline WHERE {FINAL_STEP} IS NULL AND status != 'failed' "
               "ORDER BY updated_at IS NULL DESC, updated_at ASC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [r["id"] for r in self.conn.execute(sql)]

    def get(self, doc_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM pipeline WHERE id = ?", (doc_id,)).fetchone()

    def next_step(self, doc_id: str) -> str | None:
        """First stage with no recorded output, or None when the document is complete.

        This is what makes resume free: the pipeline asks where to start rather than tracking it.
        """
        row = self.get(doc_id)
        if row is None:
            return None
        for step in STEPS:
            if row[step] is None:
                return step
        return None

    # -- recording ---------------------------------------------------------
    def record(self, doc_id: str, step: str, value: str) -> None:
        """Commit one stage output. Committed immediately -- that is the whole point."""
        if step not in STEPS:
            raise ValueError(f"unknown step {step!r}")
        status = "done" if step == FINAL_STEP else "in_progress"
        with self.conn:
            self.conn.execute(
                f"UPDATE pipeline SET {step} = ?, status = ?, error = NULL, updated_at = ? "
                "WHERE id = ?",
                (value, status, time.time(), doc_id),
            )

    def fail(self, doc_id: str, error: str, *, max_attempts: int = 3) -> None:
        """Record a failure. Marked 'failed' only after ``max_attempts``.

        Distinguishing transient from permanent matters: a rate-limit storm should not permanently
        retire a document, but a passage the content filter always rejects should stop consuming
        quota. The attempt counter draws that line without human triage.
        """
        with self.conn:
            self.conn.execute(
                "UPDATE pipeline SET attempts = attempts + 1, error = ?, "
                "status = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'pending' END, "
                "updated_at = ? WHERE id = ?",
                (error[:2000], max_attempts, time.time(), doc_id),
            )

    def reset_failed(self) -> int:
        """Requeue everything marked failed, clearing the attempt counter.

        For the common case where a whole batch failed for an environmental reason -- expired key,
        wrong endpoint, exhausted quota -- and the documents themselves are fine.
        """
        with self.conn:
            cur = self.conn.execute(
                "UPDATE pipeline SET status = 'pending', attempts = 0, error = NULL "
                "WHERE status = 'failed'")
        return cur.rowcount

    # -- reading results ---------------------------------------------------
    def stats(self) -> dict[str, int]:
        counts = {"total": 0, "done": 0, "failed": 0, "pending": 0}
        counts["total"] = self.conn.execute("SELECT COUNT(*) FROM pipeline").fetchone()[0]
        for status in ("done", "failed"):
            counts[status] = self.conn.execute(
                "SELECT COUNT(*) FROM pipeline WHERE status = ?", (status,)).fetchone()[0]
        counts["pending"] = counts["total"] - counts["done"] - counts["failed"]
        for step in STEPS:
            counts[step] = self.conn.execute(
                f"SELECT COUNT(*) FROM pipeline WHERE {step} IS NOT NULL").fetchone()[0]
        return counts

    def iter_triplets(self) -> Iterator[tuple[str, list[dict]]]:
        """Yield (doc_id, triplets) for every completed document.

        Rows whose JSON is unparseable are skipped rather than raised on: one bad row should not
        prevent a graph build over the other few thousand good ones.
        """
        rows = self.conn.execute(
            f"SELECT id, {FINAL_STEP} FROM pipeline WHERE status = 'done' "
            f"AND {FINAL_STEP} IS NOT NULL")
        for doc_id, payload in rows:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            triplets = data.get("triplets", []) if isinstance(data, dict) else data
            if isinstance(triplets, list):
                yield doc_id, triplets

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
