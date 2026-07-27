"""The soft pattern language: parsing ``MEM( head ^ relation ^ tail )``.

A pattern is a triplet-shaped query in which every slot is independently one of three kinds:

    CONCRETE   a literal mention        ``cholesterol``, ``breast cancer``
    TYPE       a schema type            ``FOOD``, ``DISEASE_DISORDER``
    WILDCARD   unconstrained            ``?``

Nothing in a pattern has to match a stored string. CONCRETE slots are resolved by vector similarity
against the matching subspace of the graph, TYPE slots by exact match on a node's assigned type, and
WILDCARD slots impose no constraint at all. This is the sense in which the graph is *soft*: the
structure is exact where structure is reliable (the closed type vocabulary) and fuzzy where it is not
(open-vocabulary entity surface forms).

The alternative -- have an LLM write Cypher against the extracted graph -- fails on vocabulary
mismatch. A query asking about "heart attack" finds nothing when the extractor wrote "myocardial
infarction", and a strict query returns the empty set with no signal about how close it came. Soft
patterns degrade continuously instead: a slot that matches nothing exactly still ranks the nearest
candidates.

The *shape* of a pattern -- which kinds occupy which slots -- selects the retrieval strategy, so
``PatternShape`` is the useful unit of analysis, not the raw text. Shape codes are written head-first
as three characters from ``c`` / ``T`` / ``?`` (e.g. ``cTc``, ``c??``), which makes it easy to audit
what a query-rewriting prompt is actually producing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .schema import is_type_token

# Tolerant by design. Models drop the closing paren, vary the internal spacing, emit lowercase
# "mem(", wrap lines in bullets or backticks, or number the lines. All of that is recoverable and
# none of it is worth losing a pattern over.
_MEM_RE = re.compile(r"MEM\s*\((.*?)(?:\)|$)", re.IGNORECASE | re.DOTALL)
_LEAD_NOISE_RE = re.compile(r"^[\s\-*•\d.)`\"']+")


class SlotKind(str, Enum):
    CONCRETE = "c"
    TYPE = "T"
    WILDCARD = "?"


def classify_slot(raw: str) -> SlotKind:
    token = raw.strip()
    # Treat empty and near-empty slots as wildcards: a dropped slot is an unconstrained slot, which
    # is what the model meant, rather than a literal search for "-" or "".
    if not token or token in {"?", "-", "_", "*", "??", "any", "anything"}:
        return SlotKind.WILDCARD
    if is_type_token(token):
        return SlotKind.TYPE
    return SlotKind.CONCRETE


@dataclass(frozen=True)
class Pattern:
    """One parsed soft pattern."""

    head: str
    relation: str
    tail: str
    head_kind: SlotKind
    relation_kind: SlotKind
    tail_kind: SlotKind

    @property
    def shape(self) -> str:
        """Three-character shape code, head first. e.g. ``cTc``, ``c??``, ``T?c``."""
        return f"{self.head_kind.value}{self.relation_kind.value}{self.tail_kind.value}"

    @property
    def is_degenerate(self) -> bool:
        """True when only one slot constrains anything.

        A degenerate pattern (``c??``) carries no structure: it reduces to plain nearest-neighbour
        lookup on a single entity string, and none of the typed machinery participates. Tracking this
        is what reveals that a permissive prompt has quietly disabled the mechanism -- the retrieval
        still returns results, so nothing looks broken.
        """
        constrained = sum(k is not SlotKind.WILDCARD
                          for k in (self.head_kind, self.relation_kind, self.tail_kind))
        return constrained <= 1

    @property
    def has_type(self) -> bool:
        return SlotKind.TYPE in (self.head_kind, self.tail_kind)

    def texts(self) -> set[str]:
        """Every query-side string this pattern needs embedded.

        Includes the composite strings used by the multi-slot strategies, so a caller can batch-embed
        all of a query's requirements in one pass instead of encoding slot by slot.
        """
        out: set[str] = set()
        head_c = self.head_kind is SlotKind.CONCRETE
        tail_c = self.tail_kind is SlotKind.CONCRETE
        rel_c = self.relation_kind is SlotKind.CONCRETE
        if head_c:
            out.add(self.head)
        if tail_c:
            out.add(self.tail)
        if rel_c:
            out.add(self.relation)
        if head_c and tail_c:
            out.add(f"{self.head} {self.tail}")
        if head_c and rel_c:
            out.add(f"{self.head} {self.relation}")
        if rel_c and tail_c:
            out.add(f"{self.relation} {self.tail}")
        if head_c and rel_c and tail_c:
            out.add(f"{self.head} {self.relation} {self.tail}")
        return {t for t in out if t}

    def __str__(self) -> str:
        return f"MEM( {self.head} ^ {self.relation} ^ {self.tail} )"


def parse_pattern(line: str) -> Pattern | None:
    """Parse a single ``MEM(...)`` line. Returns None if it is not a usable pattern."""
    line = _LEAD_NOISE_RE.sub("", line.strip())
    if not line:
        return None

    match = _MEM_RE.search(line)
    body = match.group(1) if match else line

    # Accept the documented "^" separator and the "|" that models occasionally substitute. Comma is
    # deliberately NOT accepted: entity mentions legitimately contain commas.
    parts = re.split(r"\s*[\^|]\s*", body)
    if len(parts) < 3:
        return None
    head, relation, tail = (p.strip().strip('"').strip("'") for p in parts[:3])
    # Strip a trailing paren left by a model that closed the call inside the last slot.
    tail = tail.rstrip(")").strip()

    pattern = Pattern(
        head=head, relation=relation, tail=tail,
        head_kind=classify_slot(head),
        relation_kind=classify_slot(relation),
        tail_kind=classify_slot(tail),
    )
    # A pattern with nothing but wildcards matches the whole graph and ranks it by nothing.
    if (pattern.head_kind is SlotKind.WILDCARD and pattern.tail_kind is SlotKind.WILDCARD
            and pattern.relation_kind is SlotKind.WILDCARD):
        return None
    return pattern


def parse_patterns(reply: str) -> list[Pattern]:
    """Parse an LLM reply into patterns, preserving order and dropping duplicates.

    Order is preserved because query-rewriting prompts tend to emit their best-guess pattern first,
    and a caller that truncates should keep the front.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[Pattern] = []
    for line in (reply or "").splitlines():
        pattern = parse_pattern(line)
        if pattern is None:
            continue
        key = (pattern.head.lower(), pattern.relation.lower(), pattern.tail.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(pattern)
    return out


def shape_summary(patterns_by_query: dict[str, list[Pattern]]) -> dict[str, object]:
    """Aggregate statistics over generated patterns.

    Exists because "did the query rewriter produce structured patterns or degenerate ones?" is a
    question worth answering *before* interpreting any retrieval result. A prompt change that halves
    the degenerate rate is a substantive change to what is being measured.
    """
    from collections import Counter

    shapes: Counter[str] = Counter()
    n_patterns = n_typed_slots = n_slots = 0
    all_degenerate = 0
    empty = 0

    for patterns in patterns_by_query.values():
        if not patterns:
            empty += 1
            continue
        if all(p.is_degenerate for p in patterns):
            all_degenerate += 1
        for p in patterns:
            n_patterns += 1
            shapes[p.shape] += 1
            for kind in (p.head_kind, p.relation_kind, p.tail_kind):
                n_slots += 1
                if kind is SlotKind.TYPE:
                    n_typed_slots += 1

    n_queries = max(1, len(patterns_by_query))
    return {
        "queries": len(patterns_by_query),
        "patterns": n_patterns,
        "patterns_per_query": n_patterns / n_queries,
        "queries_without_patterns": empty,
        "queries_all_degenerate": all_degenerate,
        "degenerate_rate": all_degenerate / n_queries,
        "typed_slot_rate": n_typed_slots / max(1, n_slots),
        "shapes": dict(shapes.most_common()),
    }
