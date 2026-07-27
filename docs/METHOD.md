# Method

A detailed walk through the system, in the order data flows through it. The [README](../README.md)
gives the summary; this document gives the reasoning and the mechanics.

---

## 1. The problem

Biomedical retrieval-augmented generation has a granularity mismatch. The evidence that answers a
clinical question is usually a **single relation between typed entities** — *plant sterols reduce LDL
cholesterol* — but it lives inside a document that is mostly about something else: study design,
cohort characteristics, statistical methodology.

Two standard approaches each lose something specific:

**Dense whole-document retrieval** embeds the document as one vector. The one sentence that answers
the question is averaged together with several hundred words of methodology. Documents about a topic
rank well; documents containing a *fact* do not, unless the fact happens to be the topic.

**Lexical retrieval** matches surface forms. Biomedical text is exceptionally hostile to this:
`heart attack` / `myocardial infarction` / `MI`, `vitamin B12` / `cobalamin`, `high blood pressure` /
`hypertension`. Every synonym pair is a miss.

The structural fix is to index facts rather than documents. This is what knowledge-graph RAG proposes,
and where it usually breaks down:

**Strict KG retrieval** (LLM-generated Cypher over an extracted graph) inherits the extractor's
vocabulary. The graph stores whatever surface form appeared in the source text, so a generated query
matches only if it guesses that form exactly. Worse, failure is silent and total: a query that misses
returns the empty set, with no indication of how near it came. Canonicalising nodes against an
ontology (UMLS via MetaMap, say) helps with synonymy but introduces its own errors, and a
million-concept vocabulary gives no usable level of abstraction to query *with*.

## 2. The proposal: soft patterns over a typed graph

Split the difference along the axis where reliability actually differs.

The **type** of an entity is a closed-vocabulary, low-cardinality decision. An extractor can make it
reliably, and two nodes assigned the same type genuinely are the same kind of thing. Keep that exact.

The **surface form** of an entity is open-vocabulary and unbounded. No extractor will be consistent
about it and no normalisation scheme will be complete. Make that soft.

A query is then a triplet-shaped pattern in which each slot is independently concrete, typed, or
wild:

```
MEM( FOOD ^ lowers ^ cholesterol )
      │        │          └── concrete: vector similarity in the `tail` subspace
      │        └── concrete: vector similarity, folded in via the `full` subspace
      └── TYPE: exact match against head_type
```

Read as: *a fact whose head is typed FOOD, whose tail is semantically near "cholesterol", and whose
relation is semantically near "lowers"*. Nothing needs to match a stored string. The query works
whether the graph recorded `serum LDL`, `blood cholesterol`, or `hypercholesterolaemia`.

### The type schema is answer-shaped

[`softkg/schema.py`](../softkg/schema.py) defines twelve types: `ACTIVITY`, `BODY_PART`, `CHEMICALS`,
`DISCRIMINATIVES`, `DISEASE_DISORDER`, `DRUGS`, `BIO_MARKER`, `FOOD`, `GENES`,
`HEALTH_PROCEDURES`, `SIGN_OR_SYMPTOM`, `RISK_FACTOR`.

This is not an attempt at a biomedical ontology, and it would be a poor one. It is a set of
**answer categories** — each type names something a clinician might ask *about*. That property is what
makes a type usable as a query-side wildcard: *"what disease causes X"* maps to `DISEASE_DISORDER`,
*"what food helps X"* maps to `FOOD`. A query rewriter has to be able to guess the right type from the
question, which puts a hard ceiling on how fine the schema can be.

The `NONE` type exists so structured output can always be satisfied, but the extraction prompt is
explicit that a triplet whose entity fits nothing should be **dropped** rather than typed `NONE`.

---

## 3. Extraction

### 3.1 Decontextualization first

A triplet is indexed on its own, divorced from its paragraph. Any fact whose meaning depends on
surrounding discourse becomes noise at the moment of extraction:

> *Fish oil supplementation was tested in 240 adults. **It** reduced LDL by 12% in **this cohort**.*

Extracting directly yields `(it) --[reduced]--> (LDL)` — an unusable node — and loses the referent
entirely. So stages 1–3 rewrite the passage so every sentence stands alone *before* anything is
extracted. This is the single largest quality lever in the pipeline.

**Stage 1 — pronoun resolution.** `he, she, it, they, them, their, his, her, its, we, our, him` are
replaced with their referents. The prompt is explicit about what *not* to touch: abbreviations stay
(`PBDEs`, `CVD`), relative pronouns stay (`which`, `that`, `who`), and demonstratives are left for
stage 2. Without those exclusions the model expands acronyms and rewrites relative clauses, both of
which change meaning.

**Stage 2 — indirect reference transfer.** `this`, `that`, `these`, `such`, `the former`,
`doing so`, `to achieve so` are replaced with the concept they refer to.

These are separate calls rather than one prompt because combining them measurably increased
unrequested paraphrasing — the model shifted from substituting to rewriting, which is a different and
much more damaging operation.

Both prompts open with a **response contract**: return the full passage exactly once, never omit
sentences, never return an empty string, never output only a title. Reasoning models asked to rewrite
a passage will occasionally return a summary instead, and each such response silently destroys a
document. The contract plus explicit no-change examples suppress it, and
[`softkg/extract.py`](../softkg/extract.py) additionally *rejects* any rewrite shorter than half its
input rather than accepting a collapsed passage.

**Stage 3 — bullet flattening.** Converts heading+list structures into standalone sentences.
Implemented but disabled by default; see [DESIGN_NOTES](DESIGN_NOTES.md#stage-3-is-a-deliberate-no-op).

### 3.2 Typed extraction via forced tool call

Stage 4 emits `(e1, e1_type, verb, e2, e2_type)`. Two decisions matter.

**Structured output is grammar-constrained, not parsed.** A tool schema is declared, `tool_choice`
pins the model to it, and the arguments the model is compelled to produce are the payload. The
function is never implemented — the point is that the provider constrains decoding, so malformed JSON
is impossible by construction rather than something to retry around.

**Typing is fused with extraction, not a second pass.** The extractor still holds the sentence context
that disambiguates `cholesterol`-as-`CHEMICALS` (a substance in food) from
`cholesterol`-as-`BIO_MARKER` (a measured serum level). A downstream classifier receives the bare
string and has to guess.

The extraction prompt is long — roughly 135 lines — and every line is there because of an observed
failure. It constrains entity length (5–6 words) and verb length (1–4 words), which is what keeps
graph-side strings short enough to embed without truncation. Most of its bulk suppresses specific
classes of junk:

| Junk class | Example the prompt exists to kill |
|---|---|
| Study framing | `(the study) --[showed]--> (X reduces Y)` instead of `(X) --[reduces]--> (Y)` |
| Policy / administration | `(stakeholder engagement) --[improves]--> (implementation pace)` |
| Nutrition inventory | `(dates) --[contain]--> (vitamins)`, exhaustively, from a composition table |
| Tautological measurement | `(procedure) --[has effective dose]--> (effective dose)` |
| Confounder lists | `(BMI) --[confounds]--> (X)` from a regression adjustment set |
| Clinician-behaviour commentary | `(physicians) --[prefer]--> (evidence source)` |

Shortening this prompt costs precision directly. It is kept as data in
[`softkg/prompts.py`](../softkg/prompts.py), separate from calling code, so it can be diffed and
versioned as the artifact it is.

### 3.3 Checkpointing as a first-class concern

Every stage output for every document is committed to SQLite as it happens
([`softkg/checkpoint.py`](../softkg/checkpoint.py)). A crash costs at most one LLM call. Over a job
that runs for hours through rate-limit storms and machine sleeps, this is the difference between a
pipeline that finishes and one that doesn't.

The intermediate columns are not only bookkeeping. Each adjacent pair — `original_text` →
`step1_pronoun_resolved`, `step1` → `step2`, `step3` → `step4_triplets_json` — is a complete
supervised dataset for that stage. **The expensive extraction run is simultaneously a distillation
corpus**, which is what [`finetuning/`](../finetuning/) consumes. Checkpointing honestly is what makes
that free.

---

## 4. The graph

### 4.1 Five subspaces

Each triplet is embedded five times ([`softkg/graph.py`](../softkg/graph.py)):

| Subspace | Text | Purpose |
|---|---|---|
| `head` | `plant sterols` | head entity alone |
| `tail` | `LDL cholesterol` | tail entity alone |
| `pair` | `plant sterols LDL cholesterol` | both endpoints, relation ignored |
| `verb` | `reduce` | relation alone |
| `full` | `plant sterols reduce LDL cholesterol` | the fact as a sentence |

The reason is that a pattern constrains different parts of a triplet depending on its shape, so the
right comparison target changes with it. `MEM( FOOD ^ lowers ^ cholesterol )` should match on the
**tail** while filtering heads by type; scoring it against `full` drags the verb and the other endpoint
in as noise. `MEM( plant sterols ^ reduce ^ LDL )` is the opposite — its slots are only jointly
meaningful, so `full` is correct and per-slot scoring would fragment it. A single index forces every
query shape through the same comparison and throws away the structure extraction just paid for.

### 4.2 Type indices

Two dictionaries map each type to the row indices carrying it, one for heads and one for tails. A
typed slot becomes an array lookup, and the similarity scan is then computed *only* over those rows —
which is both exact and, since a type filter often removes an order of magnitude, cheaper than the
unfiltered scan.

### 4.3 Encoder asymmetry

Jina embeddings v3 exposes task adapters: the same weights produce different vectors for
`retrieval.passage` and `retrieval.query`. That asymmetry matters more here than in ordinary dense
retrieval. Graph-side strings are terse noun phrases (`dietary cadmium intake`); query-side strings are
pattern slots and verb phrases (`FOOD lowers`, `reduces risk of`). Embedding both with one
undifferentiated role puts them in the same distribution and blurs precisely the distinction the graph
is built on. [`softkg/embedder.py`](../softkg/embedder.py) exposes the two roles as separate methods
rather than a flag, because getting it backwards degrades every score and raises no error.

All vectors are L2-normalised on write, so cosine similarity is a plain dot product everywhere
downstream.

Similarity search is brute-force matrix multiplication rather than an ANN index. At this scale — tens
of thousands of triplets, 1024 dimensions — a full scan is milliseconds, exact, and free of the recall
loss and tuning surface an approximate index adds. The two `topk` helpers are the only places that
would change to swap in FAISS.

---

## 5. Retrieval

### 5.1 Query rewriting

An LLM turns the question into soft patterns. This is the only LLM call at query time. See
[§7](#7-the-pattern-shape-problem) — it is more consequential than it looks.

### 5.2 Shape dispatch

[`softkg/retrieve.py`](../softkg/retrieve.py) dispatches on the pattern's three-character shape code.
Every branch answers two questions: which rows are **eligible** (type filters), and what should they be
**scored against** (subspaces).

| Shape | Eligible rows | Scored against |
|---|---|---|
| `cTc`, `Tcc` (one typed endpoint, verb given) | type index on the typed endpoint | opposite-endpoint subspace, combined with `full` against the concrete half |
| `c?T`, `T?c` (one typed endpoint, no verb) | type index | opposite-endpoint subspace |
| `Tc?`, `?cT` (typed endpoint, verb only) | type index | `verb` |
| `T?T`, `TcT` (both endpoints typed) | intersection of both type indices | `verb` if given, else `full` against the original question |
| `ccc` (fully specified) | all | `full` |
| `c?c` (both endpoints, no verb) | all | intersection of `head`+`tail`, unioned with `pair` |
| `cc?`, `?cc` | all | endpoint subspace, combined with `full` against the given half |
| `c??`, `??c` (degenerate) | all | single endpoint subspace |
| `?c?` (verb only) | all | `verb` |

Two composite-scoring notes. When several similarity components apply, they are averaged over the
components *present* for each candidate, not over all possible components — a fact should not be
penalised for missing a channel it was never eligible for, since that would measure the funnel rather
than relevance. And a verb constraint is applied by **rescoring** an already-selected candidate set
rather than by re-filtering, which is how a relation narrows a set chosen by its endpoints without
being able to eliminate it.

A **raw-query channel** runs alongside the patterns: the unmodified question matched against `full`.
It is the safety net when pattern generation misfires, and switching it on alone with no patterns is
exactly the no-LLM ablation.

### 5.3 Facts to documents

Each fact's best score is kept, along with which pattern produced it. Facts are then grouped by source
document and aggregated:

- **`max`** (default) — a document is as relevant as its single best-matching fact.
- **`sum{N}`** — sum of a document's N best facts.

These encode genuinely different notions of relevance (*one strong fact* vs *many weak facts*), so both
are implemented and which suits a task is empirical rather than assumed.

Because the retrieval unit is a fact, every ranked document arrives with the extracted statements that
caused it to rank — `DocumentHit.explain()`. This is a property of the mechanism, not a post-hoc
explanation layer.

---

## 6. Evaluation design

See [`evaluation/README.md`](../evaluation/README.md) for the harness. The controls it enforces:

1. **Same encoder for the dense baseline.** Otherwise the comparison conflates retrieval structure
   with embedding quality.
2. **Restricted and full frames.** Extraction may not cover every document. Scoring the graph on the
   full corpus penalises it for documents it never received; scoring baselines only on the extracted
   subset flatters it by deleting distractors. Both are reported.
3. **A coverage oracle.** Perfect ranking within reachable documents, separating *weak retrieval* from
   *missing extraction*.
4. **An LLM ablation.** The pattern layer removed, so its contribution is measured rather than assumed.
5. **Paired bootstrap with win/loss/tie counts.** Per-query variance is large; and the counts
   distinguish a broad shift from a few outliers, which have the same mean and different meanings.

---

## 7. The pattern shape problem

The most transferable lesson here is not about graphs.

The original query-rewriting prompt *invites* schema types — *"if intent of the user can be queried by
a category of schema then we can use the category names as entity"*. A strong instruction-following
model takes the cheapest legal option and emits a single degenerate pattern:

```
MEM( cholesterol ^ ? ^ ? )
```

That is nearest-neighbour lookup on one entity string. **None of the typed machinery participates.** No
type filter runs, no subspace beyond `head` is consulted, no structure is exercised. And retrieval
still returns plausible-looking results, so nothing appears broken.

An evaluation run in that state does not measure typed soft-pattern retrieval. It measures dense
retrieval over short entity strings, and reports the answer under the graph's name.

Two consequences shaped this codebase:

**Pattern shape is instrumented as a first-class metric.** `shape_summary()` in
[`softkg/patterns.py`](../softkg/patterns.py) reports patterns-per-query, typed-slot rate, degenerate
rate, and full shape distribution. The benchmark prints these *before* any retrieval metric. A number
without them is not interpretable.

**Both prompts ship.** `SEARCH_PATTERNS` (permissive) and `SEARCH_PATTERNS_TYPED` (a type mandatory in
at least one slot, 4–8 patterns, both directions) differ only in the instruction about types — same
schema, same format, same task. That makes them a clean A/B over pattern shape alone, and it means the
question *"does the typed mechanism help?"* can be asked without confounding it with a change of model
or output format.

The general form: when a prompt makes a mechanism *optional*, a capable model will decline it, the
system will keep working, and the resulting measurement will be of something other than what was
intended.
