# Evaluation harness

A BEIR-style benchmark for soft-KG retrieval against lexical, dense, and hybrid baselines.

The design assumption throughout is that **a retrieval number is only as trustworthy as the controls
around it**. Most of the code here exists to remove a specific way of fooling yourself.

## Contents

| File | Purpose |
|---|---|
| `metrics.py` | nDCG / P / Recall / MAP / MRR with `trec_eval` semantics. Self-tested. |
| `significance.py` | Paired bootstrap with confidence intervals and win/loss/tie counts. Self-tested. |
| `run_benchmark.py` | Builds runs for each system, evaluates in both frames, reports significance. |

Both measurement modules run their own tests against hand-computed values:

```bash
python evaluation/metrics.py
python evaluation/significance.py
```

Verifying the measuring instrument before trusting its readings is cheap, and a metric implementation
that disagrees with `trec_eval` produces numbers that cannot be compared to any published result.

## Usage

```bash
# 1. cache the LLM-generated soft patterns once. Resumable -- each reply is written through.
python evaluation/run_benchmark.py --data data/nfcorpus --graph data/graph \
    --generate-patterns --patterns data/patterns_typed.json --prompt typed

# 2. evaluate. No API access needed from here on, so the comparison is reproducible.
python evaluation/run_benchmark.py --data data/nfcorpus --graph data/graph \
    --patterns data/patterns_typed.json \
    --systems soft_kg,soft_kg_nollm,bm25,dense,hybrid,oracle
```

Useful variations:

```bash
--frames restricted          # like-for-like only
--aggregation sum3           # sum of a document's 3 best facts instead of max
--metric Recall@100          # which metric significance testing uses
--prompt permissive          # the A/B against the type-forcing prompt
```

Expected dataset layout (BEIR):

```
data/nfcorpus/
  corpus.jsonl        {"_id": "MED-123", "title": "...", "text": "..."}
  queries.jsonl       {"_id": "PLAIN-1", "text": "..."}
  qrel_test.tsv       query-id \t corpus-id \t score
```

## Systems

| Name | What it is |
|---|---|
| `soft_kg` | soft patterns + raw-query channel — the full method |
| `soft_kg_nollm` | raw-query channel only — isolates the LLM pattern layer's contribution |
| `bm25` | lexical baseline over the same documents |
| `dense` | whole-document embeddings from **the same encoder the graph uses** |
| `hybrid` | reciprocal-rank fusion (k=60) of soft_kg + bm25 + dense |
| `oracle` | perfect ranking restricted to documents the graph can reach — a ceiling, not a system |

## The five controls, and what each one prevents

**1. The dense baseline shares the graph's encoder.**
Comparing a Jina-v3-backed graph against a MiniLM-backed dense baseline measures the encoder, not the
retrieval structure. Any gain could be entirely attributable to the embedding model. Using the same
encoder for both makes the comparison about structure, which is the thing under study.

**2. Two evaluation frames.**
Extraction may not cover every document in the corpus. That creates a trap in both directions:

- Scoring the graph on the **full** corpus penalises it for documents it was never given.
- Scoring the baselines on only the **extracted subset** flatters the graph by deleting distractors it
  would otherwise have to outrank.

Both are reported. In the `restricted` frame, judgements pointing outside the extracted subset are
dropped for every system alike, and BM25/dense indices are **rebuilt per frame** — an index over the
subset is a different system from an index over the corpus, and reusing one for both is precisely the
like-for-like error this harness exists to avoid. A conclusion that holds in only one frame is a
conclusion about the frame.

**3. A coverage oracle.**
A perfect ranking limited to documents that have triplets. This separates two diagnoses that look
identical in the headline number and have completely different fixes:

- *low oracle* → extraction coverage is the binding constraint; no retrieval tuning will help
- *high oracle, low system* → the retrieval mechanism is leaving reachable relevance on the table

**4. An LLM ablation.**
`soft_kg_nollm` removes pattern generation entirely, leaving the raw question against the fact index.
If the full system does not beat it, the query-rewriting layer is contributing nothing — worth knowing
before attributing any result to "LLM-guided graph retrieval".

**5. Paired bootstrap significance testing.**
Per-query nDCG variance on a few hundred queries is large enough that differences of one or two points
are routinely noise. The bootstrap resamples queries with replacement, scoring both systems on the
identical resample so query-difficulty variance cancels. Win/loss/tie counts are reported alongside,
because a mean difference produced by a broad shift and one produced by three outlier queries are very
different situations with the same mean.

## Pattern shape is reported before any retrieval metric

The harness prints patterns-per-query, typed-slot rate, and degenerate rate before it prints a single
retrieval number. This is not diagnostics-for-completeness — it is a precondition for interpreting the
result at all.

A permissive query-rewriting prompt leads capable models to emit `MEM( concrete ^ ? ^ ? )`: a
degenerate pattern that reduces to nearest-neighbour lookup on one entity string and exercises **none**
of the typed machinery. Retrieval still returns plausible results, so nothing appears broken. A
benchmark run in that state is measuring dense retrieval over short strings and reporting it under the
graph's name.

So: confirm the mechanism fired before interpreting what it achieved. See
[`docs/METHOD.md` §7](../docs/METHOD.md#7-the-pattern-shape-problem).

## Notes on the metric implementation

`metrics.py` follows `trec_eval` / BEIR conventions so numbers are comparable to published results:

- **nDCG@k** uses *linear* gains (`gain = graded relevance`, not `2^rel − 1`), with IDCG computed from
  the ideal ordering of all judged documents truncated at k. This is `trec_eval`'s `ndcg_cut_k`; the
  exponential-gain variant gives visibly different numbers and is a common source of
  non-reproducibility.
- **Queries in the qrels but missing from a run score zero.** A system that returns nothing for a query
  does not get to skip it.
- **Queries with no positive judgements are excluded**, since the metrics are undefined for them, and
  the count is reported separately.
- **Ties break deterministically** by document id, so runs are reproducible.
