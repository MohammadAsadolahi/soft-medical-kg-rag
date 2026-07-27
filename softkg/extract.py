"""The extraction half: turn free-text biomedical passages into typed standalone triplets.

    raw abstract
        -> stage 1  resolve pronouns
        -> stage 2  resolve indirect references
        -> stage 3  flatten bullets and headings into standalone sentences
        -> stage 4  extract (e1:TYPE) --[verb]--> (e2:TYPE)

Stages 1-3 are collectively *decontextualization*, and they are the reason this is a pipeline rather
than one prompt. A triplet is indexed alone, divorced from its paragraph, so any fact whose meaning
lives in the surrounding discourse becomes noise the moment it is extracted. "It reduced LDL by 12%
in this cohort" yields the entity "it". Rewriting the passage so every sentence stands on its own
*before* extraction converts those from garbage nodes into usable ones.

Splitting into separate calls rather than one instruction costs 4x the tokens and buys two things:
each stage is separately checkpointed and separately inspectable, and -- because each stage's input
and output are both persisted -- each stage becomes its own supervised dataset. That is what makes
the fine-tuning in ``finetuning/`` possible: the expensive frontier-model run is simultaneously a
distillation corpus.

Stage 3 is currently a pass-through. Its prompt is retained in the original research code but was
disabled after measurement: on biomedical abstracts, which are near-universally continuous prose,
the flattening call fired on a small minority of documents and its main observable effect was
occasionally dropping title lines. It stays as an explicit no-op stage so the column, the ordering,
and the resume logic are unchanged if a corpus of guideline documents or bulleted patient leaflets
makes it worth re-enabling.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from .checkpoint import FINAL_STEP, STEPS, CheckpointStore
from .llm import LLMClient
from .prompts import (DECONTEXTUALIZE_PRONOUNS, DECONTEXTUALIZE_REFERENCES,
                      EXTRACT_RELATIONS, EXTRACT_RELATIONS_TOOL)
from .schema import normalize_type

logger = logging.getLogger(__name__)

# Rewriting stages must return the passage, not a summary. A reply far shorter than the input means
# the model collapsed the text -- a silent, expensive corruption if accepted. Reject and retry
# instead. The bound is deliberately loose: legitimate pronoun resolution changes length only
# slightly, so anything under half the original is pathological rather than borderline.
MIN_REWRITE_RATIO = 0.5


@dataclass
class ExtractionStats:
    documents: int = 0
    completed: int = 0
    failed: int = 0
    triplets: int = 0
    rejected_rewrites: int = 0
    dropped_triplets: int = 0
    calls: dict[str, int] = field(default_factory=dict)

    def bump(self, step: str) -> None:
        self.calls[step] = self.calls.get(step, 0) + 1


class RewriteRejected(RuntimeError):
    """A decontextualization stage returned something that is not the passage."""


class ExtractionPipeline:
    """Runs documents through the four stages, checkpointing after each.

    ``store`` owns all state; the pipeline itself is stateless apart from counters, so it can be
    instantiated per worker process against the same database.
    """

    def __init__(
        self,
        llm: LLMClient,
        store: CheckpointStore,
        *,
        model: str | None = None,
        enable_flattening: bool = False,
        flatten_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.model = model
        self.enable_flattening = enable_flattening
        self.flatten_prompt = flatten_prompt
        self.stats = ExtractionStats()

        if enable_flattening and not flatten_prompt:
            raise ValueError("enable_flattening=True requires flatten_prompt")

        self._handlers: dict[str, Callable[[str], str]] = {
            "step1_pronoun_resolved": self._resolve_pronouns,
            "step2_references_resolved": self._resolve_references,
            "step3_flattened": self._flatten,
            "step4_triplets_json": self._extract,
        }

    # -- stage implementations --------------------------------------------
    def _rewrite(self, prompt: str, text: str) -> str:
        reply = (self.llm.chat(prompt, text, model=self.model) or "").strip()
        if not reply:
            raise RewriteRejected("empty reply")
        if len(reply) < MIN_REWRITE_RATIO * len(text):
            raise RewriteRejected(
                f"reply is {len(reply)} chars for a {len(text)}-char input; "
                "treating as a collapsed rewrite")
        return reply

    def _resolve_pronouns(self, text: str) -> str:
        return self._rewrite(DECONTEXTUALIZE_PRONOUNS, text)

    def _resolve_references(self, text: str) -> str:
        return self._rewrite(DECONTEXTUALIZE_REFERENCES, text)

    def _flatten(self, text: str) -> str:
        """Stage 3. A no-op unless explicitly enabled -- see the module docstring."""
        if not self.enable_flattening:
            return text
        assert self.flatten_prompt is not None
        return self._rewrite(self.flatten_prompt, text)

    def _extract(self, text: str) -> str:
        """Stage 4. Returns the JSON string that gets persisted verbatim.

        Persisting the model's own JSON (rather than a re-serialised cleaned version) keeps the
        column usable as a fine-tuning target: the assistant turn should be exactly what a model is
        expected to emit. Normalisation happens later, at graph-build time.
        """
        payload = self.llm.tool_call(
            EXTRACT_RELATIONS,
            f"Extract all relationships from this text:\n\n{text}",
            EXTRACT_RELATIONS_TOOL,
            EXTRACT_RELATIONS_TOOL["function"]["name"],
            model=self.model,
        )
        triplets = payload.get("triplets", [])
        kept, dropped = [], 0
        for t in triplets:
            e1 = (t.get("e1") or "").strip()
            e2 = (t.get("e2") or "").strip()
            verb = (t.get("verb") or "").strip()
            # A triplet missing either endpoint cannot be indexed in any subspace. A missing verb is
            # survivable -- the entity-pair subspaces still work -- so only endpoints are required.
            if not e1 or not e2:
                dropped += 1
                continue
            kept.append({
                "e1": e1,
                "e1_type": normalize_type(t.get("e1_type")),
                "verb": verb,
                "e2": e2,
                "e2_type": normalize_type(t.get("e2_type")),
            })
        self.stats.dropped_triplets += dropped
        self.stats.triplets += len(kept)
        return json.dumps({"triplets": kept}, ensure_ascii=False)

    # -- driver ------------------------------------------------------------
    def process_document(self, doc_id: str) -> bool:
        """Advance one document to completion. Returns True if it finished.

        Picks up at whichever stage has no recorded output, so calling this on an already-complete
        document is free and calling it on a half-done one resumes rather than restarts.
        """
        row = self.store.get(doc_id)
        if row is None:
            logger.warning("%s not in store", doc_id)
            return False
        if row[FINAL_STEP] is not None:
            return True

        # Each stage consumes the previous stage's output; stage 1 consumes the raw text.
        values = {step: row[step] for step in STEPS}
        current = row["original_text"]

        for i, step in enumerate(STEPS):
            if values[step] is not None:
                current = values[step]
                continue
            try:
                self.stats.bump(step)
                produced = self._handlers[step](current)
            except RewriteRejected as exc:
                self.stats.rejected_rewrites += 1
                self.store.fail(doc_id, f"{step}: {exc}")
                logger.warning("%s rejected at %s: %s", doc_id, step, exc)
                return False
            except Exception as exc:
                self.store.fail(doc_id, f"{step}: {type(exc).__name__}: {exc}")
                logger.warning("%s failed at %s: %s", doc_id, step, exc)
                return False

            self.store.record(doc_id, step, produced)
            current = produced

        self.stats.completed += 1
        return True

    def run(self, *, limit: int | None = None, log_every: int = 25) -> ExtractionStats:
        """Process every incomplete document."""
        ids = self.store.pending_ids(limit=limit)
        self.stats.documents = len(ids)
        logger.info("extracting %d documents", len(ids))

        for n, doc_id in enumerate(ids, 1):
            if not self.process_document(doc_id):
                self.stats.failed += 1
            if n % log_every == 0 or n == len(ids):
                logger.info("  %d/%d done=%d failed=%d triplets=%d",
                            n, len(ids), self.stats.completed, self.stats.failed,
                            self.stats.triplets)
        return self.stats
