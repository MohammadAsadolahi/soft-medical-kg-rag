# Distilling the extraction pipeline into an open 8B model

Running frontier models over an entire corpus is expensive, and it means sending clinical text to a
third party. Both are real obstacles to deploying this pipeline anywhere near patient data.

Because the extraction pipeline checkpoints **every intermediate stage** for every document
([`softkg/checkpoint.py`](../softkg/checkpoint.py)), the expensive run is simultaneously a supervised
corpus. Each adjacent pair of columns is a ready-made training set:

| Stage | Input column | Target column |
|---|---|---|
| 1 — pronoun resolution | `original_text` | `step1_pronoun_resolved` |
| 2 — indirect reference transfer | `step1_pronoun_resolved` | `step2_references_resolved` |
| 4 — typed relation extraction | `step3_flattened` | `step4_triplets_json` |

No annotation effort, no separate labelling pass. This is the payoff for checkpointing honestly rather
than holding progress in memory.

## Contents

```
create_training_datasets.py                  checkpoint DB → ShareGPT-format JSONL per stage
finetune_qwen3.py                            QLoRA training driver (local / single-GPU)
qwen3_decontextualization_qlora.ipynb        the Kaggle notebook: Qwen3-8B on 2×T4 via DDP,
                                             with the full evaluation of stage 1
kernel-metadata.json                         Kaggle kernel configuration
training_prompts.json                        the condensed system prompts used for training
dataset_meta.json                            split sizes, seed, validation counts
```

## Building the datasets

```bash
python finetuning/create_training_datasets.py --db data/checkpoint.db
```

Every row is validated before use — all stage columns non-empty, stage 4 parseable as JSON with a
non-empty `triplets` array, and every triplet carrying all five required fields. Rows failing any check
are skipped and counted rather than silently included; a malformed target teaches the model to produce
malformed output.

The held-out split is drawn by a **seeded shuffle over documents**, so the same 50 documents are held
out across all three stages. This matters: the stages are chained, so a document in stage 2's training
set that also appears in stage 1's evaluation set would leak.

Training uses **condensed** system prompts rather than the full production ones. The production
extraction prompt is ~135 lines of accumulated heuristics that exist to steer a zero-shot model; a
fine-tuned model learns that behaviour from supervised examples instead, and paying those tokens on
every training example would waste most of the sequence budget. The condensed prompts are saved to
`training_prompts.json` so inference uses exactly what training saw.

## Training

Local / single GPU:

```bash
python finetuning/finetune_qwen3.py --step 1     # or 2, 4, or 'all'
```

The Kaggle notebook is the more developed path — dual T4 via `torchrun` DDP, resumable from a
checkpoint dataset, and it uploads the resulting adapter as a versioned Kaggle dataset so a run that
hits the session time limit can continue.

| Setting | Value | Why |
|---|---|---|
| Base model | Qwen3-8B (4-bit, Unsloth) | fits QLoRA in T4 VRAM |
| Adapter | LoRA + **rsLoRA** | DoRA OOMs on T4 |
| Rank / alpha | 16 / 32 | capacity for precise span tracking |
| LoRA dropout | **0** | see below |
| NEFTune | **disabled** | see below |
| Learning rate | 5e-6, cosine, 10% warmup | see below |
| Early stopping | patience 5 on eval loss | |
| Multi-GPU | DDP via `torchrun` | uses both T4s |

## Why stage 1 is the interesting one

Pronoun resolution is a **near-copy task**: the target is the input with a handful of spans
substituted. That sounds easy and is deceptively hard, and it inverts three standard fine-tuning
intuitions. All three settings above marked "see below" are there for the same underlying reason.

**LoRA dropout → 0.** Dropout is regularisation against memorising the training set. But near-copy
fidelity *is* the task — the model must reproduce unchanged text exactly — and dropout degrades exactly
that.

**NEFTune → disabled.** NEFTune adds noise to embeddings and normally improves instruction-following
robustness. Here it directly attacks the objective: noised embeddings produce paraphrases of spans that
were supposed to be copied verbatim.

**Learning rate → 5e-6**, well below the usual 1e-4–2e-5 range for LoRA. At higher rates loss collapsed
within a fraction of an epoch while output quality got *worse* — the model learned to emit
approximately-the-input rather than exactly-the-input, which is very cheap in loss and useless in
practice.

**Data rebalancing.** A large share of passages contain no resolvable pronoun, so the correct output is
the input unchanged. Under-representing those teaches the model that it should always change
*something*, producing hallucinated substitutions on clean text. No-change and heavy-change examples
are both oversampled to correct this.

### Loss is nearly useless as a quality signal here

This is the substantive methodological point. On a near-copy task, token-level loss goes very low while
the model still introduces micro-mutations to text it was supposed to reproduce — a changed article, a
dropped hyphen, a re-cased word. Loss is dominated by the ~95% of tokens that are trivially copied, so
it barely registers the ~5% that carry the whole task.

The notebook therefore evaluates at the level the task actually operates on. For each held-out sample it
diffs input against reference to obtain the set of *intended* changes, diffs input against prediction to
obtain the *made* changes, and scores those sets against each other:

- **exact match** — prediction identical to reference (the only unambiguous success)
- **change precision / recall / F1** — over span-level substitutions, so a model that finds the right
  pronouns but mangles surrounding text is not credited
- **character similarity** — how much of the passage survived
- and a per-sample **category**:

| Category | Meaning |
|---|---|
| `EXACT` | prediction matches reference |
| `EXACT (no-change)` | correctly left a clean passage untouched |
| `PARTIAL (conservative)` | correct substitutions, missed some |
| `PARTIAL (mixed)` | some correct, some spurious |
| `OVER-EDITED` | changed far more than required |
| `MISSED ALL` | found no substitutions when substitutions were needed |
| `HALLUCINATION` | invented substitutions that were not needed |

The categories are what make the failure mode legible. A model can sit at high F1 and still be
unusable, because F1 over changed spans says nothing about whether the *unchanged* 95% survived —
which is precisely what matters when the output feeds an extraction stage.

The notebook also implements and measures **word-level input anchoring** at inference: copy unchanged
spans verbatim from the input and keep only the model's deliberate substitutions. This targets the
"F1 = 1.0 but not exact match" case directly. It is reported side-by-side with raw output rather than
applied silently, because a post-processing step that repairs one metric can quietly damage another —
and in this case the side-by-side comparison is what reveals whether it actually helped.

Full per-sample results, both raw and post-processed, are in the notebook's committed outputs.

## Reusing a trained adapter

The pipeline talks to any OpenAI-compatible endpoint, so a fine-tuned model served locally by vLLM
drops in:

```bash
python scripts/run_extraction.py --corpus data/corpus.jsonl --db data/checkpoint.db \
    --endpoint http://localhost:8000/v1 --model qwen3-8b-extraction \
    --api-key dummy --no-reasoning-effort
```

`--no-reasoning-effort` is required: the `reasoning_effort` parameter is specific to reasoning models
and is a hard request error on models that do not accept it.
