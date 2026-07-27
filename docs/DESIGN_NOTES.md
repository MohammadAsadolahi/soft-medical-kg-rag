# Design notes

Decisions, bugs found, and things that did not work. Kept because the reasoning is more useful than
the conclusions, and because most of these were only found by treating the code as suspect rather than
assuming it worked.

---

## Two inert mechanisms in the original prototype

The research prototype this package is derived from contained two defects that made parts of it
decorative. Both are worth recording because neither produced an error, a warning, or an obviously
wrong result.

### L2 thresholds that filtered nothing

Candidate selection was written as:

```python
self.serach_index(self.entity1_index, entity_1, k=4096, threshold=200)
# ... keep results where distance < threshold
```

with thresholds of `100`, `200`, and `2000` at various call sites. The embeddings are L2-normalised, so
for unit vectors `‖a − b‖² = 2 − 2·cos(a,b)`, bounding the squared L2 distance at **4** and the distance
at **2**. Every threshold in the code was one to three orders of magnitude above the maximum
attainable value, so every filter admitted its entire input.

The intent — "keep only reasonably close candidates" — never executed. What actually bounded the
candidate set was the `k` argument, silently and at a value nobody had chosen for that purpose.

**Fixed by** replacing thresholds with explicit `top_k` parameters
([`softkg/retrieve.py`](../softkg/retrieve.py)), so the cutoff is real, visible, and tunable.

**Generalisable lesson:** a threshold on a quantity whose range you have not written down is not a
filter, it is a comment. Normalising the vectors changed the meaning of every distance constant in the
codebase, and nothing flagged it.

### A score combination that was not monotone

Two similarity signals were combined as:

```python
reranked[item] = (tmp_result[item][0] + 1) * (tmp_result[item][1] + 1)
pre_final = sorted(reranked, key=reranked.get)   # ascending
```

a product of raw L2 distances, sorted ascending. A product of distances is not monotone in either
component once the components have different scales: a candidate that is mediocre on both axes can
outrank one that is excellent on one and merely fair on the other. The ranking was being driven by an
interaction term nobody intended.

**Fixed by** scoring in cosine similarity and combining by arithmetic mean over the components present
for each candidate. Bounded, monotone in every component, interpretable.

### A reranker that returned a constant

The prototype's cross-encoder rerank step was:

```python
def CE_rerank(self, query, candidates):
    """Cross-encoder reranking — SKIPPED for now. Returns candidates as-is with score=0."""
    return [((can[0], can[1]), 0.0) for can in candidates]
```

Every candidate received score `0.0`. Downstream code sorted by that score, so the final ranking was
**dictionary insertion order** — an artifact of which pattern happened to be processed first. The
docstring is honest about the stub; what was not obvious is that the stub silently became the ranking
function.

**Fixed by** computing real cosine scores at that point. Reranking is now an explicit, optional,
separately-evaluated stage rather than a placeholder in the critical path.

---

## Stage 3 is a deliberate no-op

The bullet-flattening stage converts `heading + list` structures into standalone sentences. It is
implemented, its prompt is preserved, and it is **disabled by default**.

On biomedical abstracts — which are near-universally continuous prose — it fired on a small minority of
documents, and its main observable effect was occasionally dropping title lines, which is a net loss
because the title often carries the topic that the first sentence refers back to.

It is retained as an explicit pass-through stage rather than deleted, so that the column, the stage
ordering, and the resume logic are unchanged if a corpus of clinical guidelines or bulleted patient
leaflets makes it worth enabling. Removing it would mean a schema migration to restore it.

**Lesson:** a stage that does nothing on your corpus should be *visibly* doing nothing, not silently
absent.

---

## Rejecting collapsed rewrites

Reasoning models asked to rewrite a passage occasionally return a summary, a title, or an empty string
instead. Each of those silently destroys a document: the pipeline records a successful stage, and the
document proceeds to extraction with most of its content gone. There is no error to catch.

Three layers of defence, in order of cost:

1. **A response contract** at the top of each rewriting prompt — return the full passage exactly once,
   never omit sentences, never return an empty string, never output only a title.
2. **Explicit no-change examples** in the prompt, because "return it unchanged" is a strangely
   difficult instruction for a model primed to be helpful.
3. **A length check in code** — a reply under half the input's length is rejected and retried
   ([`softkg/extract.py`](../softkg/extract.py)). The bound is loose on purpose: legitimate pronoun
   resolution changes length slightly, so anything under half is pathological rather than borderline.

The third layer exists because the first two reduce the rate without eliminating it, and the failure is
both silent and expensive.

---

## Distinguishing transient from permanent failure

A document that fails needs different handling depending on why. A rate-limit storm should not
permanently retire it; a passage the content filter always rejects should stop consuming quota.

The checkpoint store keeps an attempt counter and only marks a document `failed` after N attempts, with
`reset_failed()` for the common case where an entire batch failed environmentally (expired key, wrong
endpoint, exhausted quota) and the documents themselves are fine. This draws the line without human
triage.

`BadRequestError` is deliberately **not** retried in [`softkg/llm.py`](../softkg/llm.py): retrying sends
an identical payload and fails identically, so it burns quota to reach the same place.

---

## Embedding cache fingerprints

The cache is keyed by text, but a text's vector depends on the model, the task adapter, and the
truncation length. A cache written under one configuration and read under another yields vectors that
are silently incomparable — the worst available failure mode, because scores stay plausible and only
the ranking is wrong.

Each cache namespace therefore stores a fingerprint of `(model, task, max_len)` and refuses to load
under a mismatch, with a message saying to delete the directory. It also verifies that the key count
matches the vector count, and writes vectors to a temp file before renaming, so an interrupted save
cannot leave a cache whose keys and vectors disagree.

A related check on the graph: saved subspace vectors must have exactly as many rows as there are
triplets. Loading vectors built from a different triplet list would misattribute every score, and the
symptom would be subtly wrong rankings rather than an exception.

---

## The 32-token budget, measured rather than assumed

Graph-side strings are embedded with `max_length=32`, which looks aggressive for the composite
subspaces (`pair`, `full`). Measured over the extracted graph:

| Subspace | median words | p99 | max |
|---|---|---|---|
| `head` | 2 | 6 | 13 |
| `tail` | 2 | 6 | 12 |
| `verb` | 2 | 4 | 8 |
| `pair` | 4 | 10 | 17 |
| `full` | 6 | 12 | 20 |

No string in any subspace approaches the limit, because the extraction prompt caps entities at 5–6
words and verbs at 4. The token budget is a *consequence* of the extraction constraints, not an
independent parameter — and raising it would add padding, not fidelity. This is recorded at the
constant's definition so the next reader does not have to re-derive it.

---

## Max-pooling is a hypothesis

Aggregating fact scores to a document score by `max` encodes a specific claim: *a document is as
relevant as its single best-matching fact*. The alternative — summing a document's top N facts — encodes
*a document is relevant if it contains several relevant facts*.

These are different notions of relevance and they favour different documents. Max favours a focused
paper containing one precise statement; sum favours a review that touches the topic repeatedly. Which
is right depends on the task, so both are implemented (`--aggregation max` / `sum3` / `sum5`) and it is
treated as something to measure rather than to settle by argument.

It is also a confound worth eliminating early: if a graph-based retriever underperforms, "the
aggregation is wrong" and "the substrate is wrong" are very different diagnoses, and only one of them
is cheap to test.

---

## Things the design does not do, and why

**No entity canonicalisation.** Surface forms are left exactly as extracted. Linking to UMLS via
MetaMap is the standard move and it does help with synonymy, but it introduces its own error rate and —
more importantly here — a million-concept vocabulary provides no useful level of abstraction to query
with. The twelve-type schema exists precisely to provide that abstraction, and softness in the
embedding space is what covers synonymy instead.

**No approximate nearest neighbours.** At tens of thousands of triplets, brute-force matmul is
milliseconds and exact. An ANN index would add recall loss, index build time, and tuning parameters in
exchange for speed that is not needed. The two `topk` helpers localise the change if scale demands it.

**No multi-hop traversal.** Each pattern resolves independently; there is no join across facts. This is
the clearest limitation. Compositional questions — *"what foods affect a biomarker that predicts
disease X"* — are exactly the structure a whole-document embedding cannot represent, and therefore
exactly where a graph substrate should have its strongest advantage. Single-hop topical relevance is
the case where the advantage is thinnest, because a document embedding already captures topic well.
[T2RAG](https://github.com/Emory-Melody/T2RAG) is the relevant prior work.

**No cross-encoder in the default path.** A reranker can be layered on the output, but it is a separate
concern: it improves any candidate list and so tells you little about whether the *retrieval substrate*
is working. Keeping it out of the default path keeps the thing under study visible.

---

## The one that matters most: shape before scores

Covered in [METHOD §7](METHOD.md#7-the-pattern-shape-problem), restated here because it is the finding
with the widest applicability.

The permissive query-rewriting prompt *offers* schema types as an option. Strong models decline the
option and emit `MEM( concrete ^ ? ^ ? )` — a degenerate pattern that exercises none of the typed
machinery. Retrieval keeps working and returns plausible results, so nothing looks wrong. But an
evaluation in that state is not measuring typed soft-pattern retrieval; it is measuring dense retrieval
over short entity strings and reporting the answer under the graph's name.

This is why pattern shape is instrumented as a first-class metric and printed before any retrieval
number, and why both prompts ship so the mechanism can be A/B'd without confounding it with a change of
model or format.

**The general form:** when a prompt makes a mechanism optional, a capable model will decline it, the
system will keep working, and your measurement will be of something other than what you intended. Verify
the mechanism fired before interpreting what it achieved.
