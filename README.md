# Soft Knowledge Graphs for Biomedical Retrieval

**A retrieval system that indexes biomedical documents as typed, standalone facts, and queries them
with partially-specified graph patterns resolved by vector similarity instead of exact matching.**

Retrieval-augmented generation over clinical literature fails in a characteristic way. A question like
*"what foods lower LDL cholesterol"* is answered by a sentence buried in an abstract that is mostly
about study design; a whole-document embedding averages that sentence away, and lexical search misses
it when the paper says *"dietary intervention reduced serum low-density lipoprotein"*. The
information is a **relation between typed entities**, but it is being retrieved as a bag of words.

The obvious fix — build a knowledge graph and query it with Cypher — trades one failure for a worse
one. An LLM-extracted graph has open-vocabulary node labels, so a generated query asking for
`heart attack` returns nothing when the extractor wrote `myocardial infarction`, and a strict query
that misses returns the empty set with no signal about how close it came.

This project takes the middle path: keep the graph's *structure* where structure is reliable, and make
its *surface forms* soft.

```
        ┌─────────────────────────── EXTRACTION (offline) ───────────────────────────┐
        │                                                                            │
        │   abstract ──▶ resolve pronouns ──▶ resolve references ──▶ extract facts    │
        │                        │                    │                     │        │
        │                  "it reduces LDL"    "to achieve that"    (fish oil:FOOD)  │
        │                        ↓                    ↓              --[reduces]-->  │
        │              "fish oil reduces LDL"  "to achieve weight    (LDL:BIO_MARKER)│
        │                                        loss"                               │
        └────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                        ┌─────────── SOFT KNOWLEDGE GRAPH ───────────┐
                        │  typed triplets, indexed in 5 subspaces    │
                        │  head · tail · pair · verb · whole-fact    │
                        │  + exact type index over 12 entity types   │
                        └────────────────────────────────────────────┘
                                              ▲
        ┌─────────────────────────── RETRIEVAL (online) ─────────────┴──────────────┐
        │                                                                            │
        │  "what food lowers cholesterol"                                            │
        │           │                                                                │
        │           ▼ LLM rewrites the question as soft patterns                     │
        │      MEM( FOOD ^ lowers ^ cholesterol )     ← TYPE slot: exact type match   │
        │      MEM( cholesterol ^ lowered by ^ FOOD )    concrete slot: vector match  │
        │      MEM( ACTIVITY ^ reduces ^ cholesterol )   "?" slot: unconstrained      │
        │           │                                                                │
        │           ▼ resolve each slot against the matching subspace                 │
        │      ranked facts ──▶ ranked documents, each with its supporting facts      │
        └────────────────────────────────────────────────────────────────────────────┘
```

The result is a retriever whose unit of evidence is a fact rather than a document: every ranked
result arrives with the specific extracted statements that caused it to rank.

---

## The core idea: hard types, soft text

A soft pattern is a triplet in which each of the three slots is independently one of three kinds:

| Slot kind | Written as | Resolved by |
|---|---|---|
| **Concrete** | `cholesterol` | vector similarity against the relevant subspace |
| **Type** | `FOOD`, `DISEASE_DISORDER` | **exact** match against the node's assigned type |
| **Wildcard** | `?` | unconstrained |

Nothing in a pattern has to match a stored string. `MEM( FOOD ^ lowers ^ cholesterol )` means *"a fact
whose head is typed FOOD, whose tail is semantically near 'cholesterol', and whose relation is
semantically near 'lowers'"*. That query works whether the graph stored `serum LDL`,
`blood cholesterol`, or `hypercholesterolaemia`.

Two design commitments make this work, and they pull in opposite directions on purpose:

**The type vocabulary is closed and exact.** Twelve coarse biomedical types
([`softkg/schema.py`](softkg/schema.py)), no thresholds, no embeddings. Two nodes typed
`DISEASE_DISORDER` are the same kind of thing by construction. This is what lets a query say *"any
disease"* and mean it — an abstraction that a 1M-concept ontology like UMLS cannot offer, because
there is no way to write *"any concept that could answer this question"* over a vocabulary that large.
The types are deliberately chosen to be **answer-shaped**: each one names a category a clinician might
ask *about*.

**Everything textual is soft.** Entity mentions and relation verbs stay as the extractor wrote them,
matched by similarity. No canonicalisation step, no entity linking, no synonym dictionary. Vocabulary
mismatch is handled by the embedding space rather than by a normalisation pipeline that has to be
right in advance.

Retrieval degrades continuously rather than falling off a cliff: a slot that matches nothing exactly
still ranks the nearest candidates.

### Why five subspaces and not one

Each triplet is embedded five times ([`softkg/graph.py`](softkg/graph.py)):

| Subspace | Indexed text | Used when |
|---|---|---|
| `head` | `cholesterol` | the head is concrete and the tail is typed or wild |
| `tail` | `cardiovascular disease` | the mirror case |
| `pair` | `cholesterol cardiovascular disease` | both endpoints given, relation unspecified |
| `verb` | `increases risk of` | only the relation constrains the query |
| `full` | `cholesterol increases risk of cardiovascular disease` | the whole fact is specified |

A pattern constrains different parts of a triplet depending on its shape, so the right comparison
target changes with it. *"What FOOD lowers cholesterol"* should match on the **tail** while filtering
heads by type — scoring that against a whole-fact embedding drags the verb and the other endpoint in
as noise. A fully specified pattern is the opposite: its endpoints are only jointly meaningful, so the
whole fact is the right unit. One undifferentiated index forces every query shape through the same
comparison and discards the structure that extraction just paid for.

The **shape** of a pattern therefore selects the retrieval strategy. That dispatch is the substance of
[`softkg/retrieve.py`](softkg/retrieve.py), and shapes are written as three-character codes (`cTc`,
`c?T`, `T?c`) so that what a query-rewriting prompt actually produces can be audited.

---

## Repository layout

```
softkg/                      the library
  schema.py                  12 answer-shaped biomedical entity types
  prompts.py                 extraction + query-rewriting prompts (the substance of the method)
  llm.py                     OpenAI-compatible client; forced tool calls for structured extraction
  checkpoint.py              resumable per-stage SQLite store
  extract.py                 the 4-stage decontextualize → extract pipeline
  embedder.py                Jina v3 with task adapters and a fingerprinted on-disk cache
  patterns.py                the MEM(...) soft pattern language and its parser
  graph.py                   typed triplet store, 5 vector subspaces, exact type index
  retrieve.py                pattern-shape dispatch, scoring, fact → document aggregation

scripts/
  run_extraction.py          corpus → checkpointed triplets
  build_graph.py             triplets → embedded, persisted graph
  search.py                  query the graph, with provenance

finetuning/                  distilling the pipeline into an 8B open model — see its own README
  qwen3_decontextualization_qlora.ipynb
  create_training_datasets.py
  finetune_qwen3.py

evaluation/                  BEIR-style benchmark harness — see its own README
  metrics.py                 nDCG / P / Recall / MAP / MRR, trec_eval semantics, self-tested
  significance.py            paired bootstrap, self-tested
  run_benchmark.py           soft-KG vs BM25 vs dense vs hybrid, with controls

docs/
  METHOD.md                  the method in detail, and why each choice was made
  DESIGN_NOTES.md            decisions, failure modes found, and what they cost
```

---

## Installation

```bash
git clone https://github.com/MohammadAsadolahi/soft-medical-kg-rag.git
cd soft-medical-kg-rag
pip install -r requirements.txt
```

CPU is sufficient for everything except fine-tuning. The graph embeds on CPU in minutes for a corpus
of a few thousand documents, and the embedding cache means you pay that cost once.

## Quickstart

The pipeline is three commands. A BEIR-format corpus (`corpus.jsonl`, `queries.jsonl`,
`qrel_test.tsv`) is the expected input; [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
is a good starting point for biomedical retrieval.

**1. Extract typed facts.** Resumable — interrupt and re-run freely.

```bash
export SOFTKG_API_KEY=...
python scripts/run_extraction.py --corpus data/nfcorpus/corpus.jsonl --db data/checkpoint.db

python scripts/run_extraction.py --db data/checkpoint.db --stats     # progress at any time
```

**2. Build the graph.** Embeds all five subspaces and persists them.

```bash
python scripts/build_graph.py --db data/checkpoint.db --out data/graph
```

**3. Search, with provenance.**

```bash
python scripts/search.py --graph data/graph -q "what foods lower cholesterol" \
    --corpus data/nfcorpus/corpus.jsonl
```

Real output, abridged:

```
Soft patterns (4):
  [Tcc] MEM( FOOD ^ lowers ^ cholesterol )
  [ccT] MEM( cholesterol ^ lowered by ^ FOOD )
  [ccT] MEM( cholesterol ^ reduced by ^ ACTIVITY )
  [Tcc] MEM( DRUGS ^ lowers ^ cholesterol )

 1. MED-1249   score=0.3353
      [0.335] (soy protein meat analogues : FOOD) --[reduces]--> (plasma cholesterol : BIO_MARKER)
              via [Tcc] MEM( FOOD ^ lowers ^ cholesterol )
      [0.330] (plant protein diet : FOOD) --[reduces]--> (plasma cholesterol : BIO_MARKER)
              via [Tcc] MEM( FOOD ^ lowers ^ cholesterol )
 2. MED-1303   score=0.3313
      [0.331] (Avena sativa : FOOD) --[reduces]--> (cholesterol : CHEMICALS)
              via [Tcc] MEM( FOOD ^ lowers ^ cholesterol )
 3. MED-1319   score=0.3223
      [0.322] (rural Chinese diets : FOOD) --[are associated with]--> (low plasma cholesterol : BIO_MARKER)
              via [ccT] MEM( cholesterol ^ lowered by ^ FOOD )
```

`Avena sativa` is oats — retrieved for a query that never mentions oats, via a pattern that never
mentions them either. The type slot did the work: the pattern asked for *any* `FOOD` standing in a
lowering relation to something near `cholesterol`.

You can also drive the graph without any LLM, which is useful for probing it directly and is exactly
what the no-LLM ablation does:

```bash
# hand-written patterns
python scripts/search.py --graph data/graph -q "..." --pattern "MEM( FOOD ^ lowers ^ cholesterol )"

# raw question against the fact index, no patterns at all
python scripts/search.py --graph data/graph -q "..." --no-patterns
```

### As a library

```python
from softkg import Embedder, LLMClient, SoftRetriever, build_graph, triplets_from_checkpoint

triplets = triplets_from_checkpoint("data/checkpoint.db")
embedder = Embedder(cache_dir="data/emb_cache")
graph = build_graph(triplets, embedder)

retriever = SoftRetriever(graph, embedder, llm=LLMClient(api_key=..., endpoint=...))
for hit in retriever.search("what foods lower cholesterol"):
    print(hit.explain())
```

---

## Extraction: decontextualization is the hard part

A triplet is indexed alone, torn out of its paragraph. Any fact whose meaning depends on surrounding
discourse becomes noise the moment it is extracted — *"It reduced LDL by 12% in this cohort"* yields
the entity `it`. So the pipeline rewrites each passage so that every sentence stands on its own
**before** extracting anything:

| Stage | What it does | Why separate |
|---|---|---|
| 1. Pronoun resolution | `it` → `fish oil` | Combining stages 1 and 2 measurably increased unrequested paraphrasing — the model started rewriting rather than substituting |
| 2. Indirect reference transfer | `to achieve that` → `to achieve weight loss` | Handles demonstratives and discourse anaphora, which stage 1 is explicitly told to leave alone |
| 3. Bullet flattening | headings + list items → standalone sentences | Retained as an explicit no-op; see [DESIGN_NOTES](docs/DESIGN_NOTES.md) for why it is disabled by default |
| 4. Typed extraction | text → `(e1:TYPE) --[verb]--> (e2:TYPE)` | Forced tool call, so the JSON is grammar-constrained rather than parsed out of prose |

Splitting this into separate calls costs roughly 4× the tokens of a single prompt. It buys two things.
Each stage is separately checkpointed and separately inspectable — and because every stage's input
*and* output are persisted, **each stage becomes its own supervised dataset**. The expensive
frontier-model extraction run is simultaneously a distillation corpus, which is what makes the
fine-tuning below possible at all.

Stage 4 uses the forced-tool-call pattern: a function schema is declared, `tool_choice` pins the model
to it, and the arguments the model is compelled to produce *are* the payload. The function is never
implemented. Extracting tens of thousands of triplets by asking for JSON in the prompt and parsing the
reply is a categorically worse experience.

Entity typing is fused into extraction rather than run as a separate classification pass, because the
extractor still holds the sentence context that disambiguates `cholesterol`-the-`CHEMICALS` from
`cholesterol`-the-`BIO_MARKER`. A downstream classifier sees only the bare string and has to guess.

Everything is checkpointed per stage per document in SQLite ([`softkg/checkpoint.py`](softkg/checkpoint.py)),
so a crash costs at most one LLM call. On a job that runs for hours and includes rate-limit storms,
this is not a nicety.

---

## Fine-tuning: distilling the pipeline into an open 8B model

Running frontier models over a whole corpus is expensive and sends clinical text to a third party.
Because the pipeline checkpointed every intermediate stage, the extraction run doubles as supervision
for training an open model to do the same job locally.

[`finetuning/`](finetuning/) contains the QLoRA training work — dataset construction from the
checkpoint database, the Kaggle notebook that trains **Qwen3-8B** on dual T4s via DDP, and the
evaluation of the resulting adapter. Pronoun resolution is the stage treated in most depth, because it
turned out to be the most interesting one to train.

It is a **near-copy task**: the target output is the input with a handful of spans substituted. That
makes it deceptively hard, and it breaks the usual fine-tuning intuitions. Loss goes very low while the
model still introduces micro-mutations to text it was supposed to copy verbatim — so token-level loss
is nearly useless as a quality signal, and the notebook instead reports exact-match rate alongside
span-level precision/recall/F1 over the *changes* the model made, with every prediction categorised
(exact / conservative / hallucinated / missed). Three techniques that normally help had to be dropped
or inverted for this reason; the reasoning is documented in [`finetuning/README.md`](finetuning/README.md).

---

## Evaluation

[`evaluation/`](evaluation/) is a BEIR-style harness comparing soft-KG retrieval against lexical,
dense, and hybrid baselines. Its design reflects the view that a retrieval number is only as
trustworthy as the controls around it:

- **The dense baseline uses the same encoder as the graph.** Otherwise the comparison conflates
  retrieval structure with embedding quality, and an apparent gain may be nothing but a better encoder.
- **Two evaluation frames.** `restricted` scores only documents the graph actually covers
  (like-for-like); `full` scores the whole corpus (deployment-honest). Reporting one alone either
  penalises the graph for documents it never received or flatters it by deleting distractors. A
  conclusion that holds in only one frame is a conclusion about the frame.
- **A coverage oracle.** A perfect ranking restricted to documents the graph can reach, which
  separates *"the retrieval mechanism is weak"* from *"the graph never saw the answer"* — two problems
  with entirely different fixes.
- **An LLM ablation.** The same system with pattern generation removed, isolating what the
  query-rewriting layer actually contributes rather than assuming it contributes something.
- **Paired bootstrap significance testing** with win/loss/tie counts, because per-query variance on a
  few hundred queries is large enough that differences of a couple of nDCG points are routinely noise.
  The counts also distinguish a broad shift from a handful of outlier queries — very different
  situations with the same mean.

Both `metrics.py` and `significance.py` carry executable self-tests against hand-computed values, so
the measuring instruments are verified rather than trusted:

```bash
python evaluation/metrics.py        # nDCG/P/Recall/AP/RR vs hand-computed trec_eval values
python evaluation/significance.py   # bootstrap behaviour on constructed data
```

Running the benchmark:

```bash
# cache the generated patterns once
python evaluation/run_benchmark.py --data data/nfcorpus --graph data/graph \
    --generate-patterns --patterns data/patterns.json --prompt typed

# evaluate -- no API access needed from here
python evaluation/run_benchmark.py --data data/nfcorpus --graph data/graph \
    --patterns data/patterns.json --systems soft_kg,soft_kg_nollm,bm25,dense,hybrid,oracle
```

### One finding worth stating up front

The repository ships **two** query-rewriting prompts, and the difference between them is the single
most important lesson from this work.

The original prompt *offers* schema types — *"if intent of the user can be queried by a category of
schema then we can use the category names as entity"*. Strong instruction-following models take the
cheapest legal option: they emit a single `MEM( cholesterol ^ ? ^ ? )`. That is a degenerate pattern.
It reduces to nearest-neighbour lookup on one entity string, and **none of the typed machinery
participates at all**. Retrieval still returns plausible results, so nothing looks broken.

Measuring the mechanism therefore requires measuring whether it *fired*.
[`softkg/patterns.py`](softkg/patterns.py) reports pattern shape distribution, typed-slot rate, and
degenerate rate for exactly this reason, and the harness prints them before any retrieval metric. A
benchmark run that does not pin down the pattern shape distribution is not interpretable — the number
it produces may be measuring plain dense retrieval wearing a knowledge graph as a costume.

`SEARCH_PATTERNS_TYPED` makes a type mandatory in at least one slot and demands several patterns
covering both directions. Same schema, same output format, same task — only the instruction about
types changes, which keeps the two comparable as a clean A/B over pattern shape alone.

---

## Design notes and limitations

Fuller treatment in [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md). The short version:

- **Two bugs in the original research prototype were load-bearing.** Its candidate filters thresholded
  L2 distance at 100 and 200, but between L2-normalised vectors the maximum possible distance is 2 —
  every threshold admitted the entire index and filtered nothing. And its score combination multiplied
  raw distances as `(d1+1)*(d2+1)`, which is not monotone in either component. Both are replaced here
  (explicit top-k; mean of cosines) and both are documented at the site of the change. Finding them
  required reading the code as if it were wrong rather than trusting that it worked.
- **Max-pooling facts to documents is a choice, not a default.** *"One strongly relevant fact"* and
  *"many weakly relevant facts"* are genuinely different notions of document relevance. Both are
  implemented (`--aggregation max` / `sum3`), because which one suits a task is empirical.
- **Extraction coverage bounds everything.** No retrieval tuning recovers a document whose facts were
  never extracted; the coverage oracle exists to keep that distinction visible.
- **The 12-type schema is coarse by design**, and coarseness is a real cost. `BIO_MARKER` absorbs a
  wide range of outcome measures. A finer schema would discriminate better but would be harder for a
  query rewriter to use correctly — the types have to be guessable from a question.
- **Single-hop by construction.** Each pattern is resolved independently; there is no traversal or join
  across facts. Compositional questions (*"what foods affect a biomarker that predicts a disease"*) are
  the natural next step and are where a graph substrate should have the clearest advantage over dense
  retrieval, since that is exactly the structure a whole-document embedding cannot represent.

---

## Related work

- **[T2RAG](https://github.com/Emory-Melody/T2RAG)** — triplet-level retrieval for RAG, with iterative
  resolution over multi-hop questions. The closest neighbour to this work, and the strongest evidence
  that triplet substrates pay off specifically when questions are compositional.
- **GraphRAG / community-summary approaches** — graph structure used for global summarisation rather
  than fact-level retrieval.
- **[BEIR](https://github.com/beir-cellar/beir) / NFCorpus** — the evaluation setting and metric
  conventions this harness follows.
- **[Jina embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3)** — task-adapted encoder;
  the passage/query asymmetry matters more here than in ordinary dense retrieval, because graph-side
  strings are terse noun phrases while query-side strings are pattern slots and verb phrases.

## Author

**Mohammad Asadolahi** — AI researcher, trustworthy ML for healthcare.
[GitHub](https://github.com/MohammadAsadolahi) · [Google Scholar](https://scholar.google.com/citations?user=0kVRbmMAAAAJ)

## License

MIT — see [LICENSE](LICENSE).
