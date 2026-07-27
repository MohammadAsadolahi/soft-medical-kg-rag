"""
Create Training Datasets for Qwen3-8B Fine-Tuning
==================================================
Extracts processed data from nfcorpus_checkpoint.db and creates
JSONL datasets in ShareGPT conversation format for 3 separate models:

  Step 1: Pronoun Resolution       (original_text → step1_pronoun_resolved)
  Step 2: Indirect Ref Transfer    (step1 → step2_references_resolved)
  Step 4: Relation Extraction      (step3_flattened → step4_triplets_json)

Each dataset is split into train (N-50) and eval (50) sets.

Uses CONDENSED system prompts for training efficiency — the model learns
detailed behavior from 1,385 supervised examples rather than a mega-prompt.
This keeps total token count within max_seq_length=4096 for 8GB VRAM.
"""

import argparse
import sqlite3
import json
import random
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from softkg.schema import ENTITY_SCHEMA_PROSE as ENTITIES_SCHEMA  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DB_PATH = ROOT / "data" / "checkpoint.db"
OUTPUT_DIR = ROOT / "finetuning" / "training_data"
EVAL_SIZE = 50
SEED = 42

# ---------------------------------------------------------------------------
# Condensed Training Prompts
# The original mega-prompts are designed for zero-shot/few-shot with GPT-5.
# For fine-tuning with 1,385 supervised examples, the model learns the task
# from data — a concise system prompt is sufficient and saves tokens.
# ---------------------------------------------------------------------------

TRAIN_PROMPT_STEP1 = """Replace every pronoun (he, she, it, they, them, their, his, her, its, we, our, him) with the specific entity it refers to based on context.

Rules:
- Preserve original meaning, structure, and punctuation.
- Do NOT expand abbreviations or acronyms.
- Do NOT rewrite relative pronouns (which, that, who, whom, where, when).
- Do NOT resolve demonstratives (this, that, these, those).
- If a pronoun has no clear referent, leave it unchanged.
- If no pronouns to resolve, return the text exactly as-is.

Output ONLY the final passage with no explanations."""

TRAIN_PROMPT_STEP2 = """Replace indirect references (this, that, these, those, such, the former, the latter, doing so, to achieve so) with the full concept or phrase they refer to from preceding sentences. Make each sentence independently understandable.

Rules:
- Only replace references you are certain about.
- Do NOT paraphrase, summarize, or alter other parts.
- If replacement would force awkward grammar, keep the original phrase.
- If no indirect references exist, return text exactly as-is.

Output ONLY the final text with no explanations."""

TRAIN_PROMPT_STEP4 = f"""Extract medical knowledge as (entity_1, verb, entity_2) triplets from the text. Output valid JSON.

ENTITY CONSTRAINTS:
- Short noun phrases only, max 6 words.
- Never use clauses or sentence fragments as entities.

VERB CONSTRAINTS:
- 1-4 words capturing the core relationship (e.g., "treats", "causes", "reduces risk of").
- No quantitative data or qualifying clauses in verbs.

WHAT TO EXTRACT:
- Medical findings, results, conclusions, and established biomedical facts.
- Skip pure methodology, meta statements, policy/admin content.
- Extract mechanisms, associations, exposures, treatments, risks, and clinical outcomes.

ENTITY TYPES — classify each entity:
{ENTITIES_SCHEMA}

DEDUPLICATION:
- Skip identical (e1, verb, e2) triplets. Keep related-but-distinct ones.

Output JSON format: {{"triplets": [{{"e1": "...", "e1_type": "...", "verb": "...", "e2": "...", "e2_type": "..."}}]}}"""


def load_completed_rows():
    """Load all completed rows from checkpoint DB."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT id, original_text, step1_pronoun_resolved,
               step2_references_resolved, step3_flattened, step4_triplets_json
        FROM pipeline
        WHERE status = 'done'
          AND step1_pronoun_resolved IS NOT NULL
          AND step2_references_resolved IS NOT NULL
          AND step4_triplets_json IS NOT NULL
        ORDER BY id
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def validate_row(row):
    """Validate a single row has non-empty fields and valid JSON for step4."""
    for field in ["original_text", "step1_pronoun_resolved",
                  "step2_references_resolved", "step3_flattened",
                  "step4_triplets_json"]:
        if not row.get(field) or not row[field].strip():
            return False, f"empty {field}"
    try:
        data = json.loads(row["step4_triplets_json"])
        if "triplets" not in data or not isinstance(data["triplets"], list):
            return False, "step4 missing triplets array"
        if len(data["triplets"]) == 0:
            return False, "step4 has 0 triplets"
        for t in data["triplets"]:
            for k in ["e1", "e1_type", "verb", "e2", "e2_type"]:
                if k not in t:
                    return False, f"triplet missing {k}"
    except json.JSONDecodeError:
        return False, "step4 invalid JSON"
    return True, "ok"


def make_conversation(system_prompt, user_text, assistant_text):
    """Create a ShareGPT-format conversation dict."""
    return {
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def build_step1_sample(row):
    """Pronoun Resolution: original → step1."""
    return make_conversation(
        system_prompt=TRAIN_PROMPT_STEP1,
        user_text=row["original_text"],
        assistant_text=row["step1_pronoun_resolved"],
    )


def build_step2_sample(row):
    """Indirect Reference Transfer: step1 → step2."""
    return make_conversation(
        system_prompt=TRAIN_PROMPT_STEP2,
        user_text=row["step1_pronoun_resolved"],
        assistant_text=row["step2_references_resolved"],
    )


def build_step4_sample(row):
    """Relation Extraction: step3 → JSON triplets."""
    return make_conversation(
        system_prompt=TRAIN_PROMPT_STEP4,
        user_text=f"Extract all relationships from this text:\n\n{row['step3_flattened']}",
        assistant_text=row["step4_triplets_json"],
    )


def write_jsonl(samples, path):
    """Write samples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"  Wrote {len(samples)} samples → {path}")


def compute_char_stats(samples):
    """Compute character length stats for conversations."""
    lengths = []
    for s in samples:
        total = sum(len(c["content"]) for c in s["conversations"])
        lengths.append(total)
    lengths.sort()
    n = len(lengths)
    return {
        "count": n,
        "min": lengths[0],
        "median": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
    }


def main():
    global DB_PATH, OUTPUT_DIR, EVAL_SIZE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DB_PATH),
                    help="extraction checkpoint database (default: data/checkpoint.db)")
    ap.add_argument("--out", default=str(OUTPUT_DIR),
                    help="output directory for the JSONL datasets")
    ap.add_argument("--eval-size", type=int, default=EVAL_SIZE,
                    help="held-out documents (default: 50)")
    args = ap.parse_args()
    DB_PATH, OUTPUT_DIR, EVAL_SIZE = Path(args.db), Path(args.out), args.eval_size

    if not DB_PATH.exists():
        raise SystemExit(
            f"checkpoint database not found: {DB_PATH}\n"
            "Run an extraction first (scripts/run_extraction.py), or pass --db.")

    logger.info(f"Loading completed rows from {DB_PATH}...")
    rows = load_completed_rows()
    logger.info(f"Loaded {len(rows)} completed rows")

    # Validate all rows
    valid_rows = []
    invalid_count = 0
    for row in rows:
        ok, reason = validate_row(row)
        if ok:
            valid_rows.append(row)
        else:
            invalid_count += 1
            if invalid_count <= 5:
                logger.warning(f"  Skipping {row['id']}: {reason}")

    if invalid_count > 5:
        logger.warning(f"  ... and {invalid_count - 5} more invalid rows")
    logger.info(f"Valid rows: {len(valid_rows)} / {len(rows)}")

    if len(valid_rows) < EVAL_SIZE + 10:
        raise ValueError(f"Not enough valid rows ({len(valid_rows)}) for eval split of {EVAL_SIZE}")

    # Shuffle and split
    random.seed(SEED)
    indices = list(range(len(valid_rows)))
    random.shuffle(indices)
    eval_indices = set(indices[:EVAL_SIZE])
    train_rows = [valid_rows[i] for i in range(len(valid_rows)) if i not in eval_indices]
    eval_rows = [valid_rows[i] for i in eval_indices]

    logger.info(f"Split: {len(train_rows)} train, {len(eval_rows)} eval")

    # Build datasets for each step
    steps = {
        "step1_pronoun": ("Step 1 (Pronoun Resolution)", build_step1_sample),
        "step2_reference": ("Step 2 (Indirect Reference)", build_step2_sample),
        "step4_extraction": ("Step 4 (Relation Extraction)", build_step4_sample),
    }

    for step_key, (step_name, builder) in steps.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Building {step_name} dataset...")

        train_samples = [builder(r) for r in train_rows]
        eval_samples = [builder(r) for r in eval_rows]

        # Write JSONL files
        write_jsonl(train_samples, OUTPUT_DIR / f"{step_key}_train.jsonl")
        write_jsonl(eval_samples, OUTPUT_DIR / f"{step_key}_eval.jsonl")

        # Stats
        train_stats = compute_char_stats(train_samples)
        eval_stats = compute_char_stats(eval_samples)
        logger.info(f"  Train char stats: {train_stats}")
        logger.info(f"  Eval char stats:  {eval_stats}")

        # Rough token estimates (chars / 4)
        est_tokens_median = train_stats["median"] // 4
        est_tokens_max = train_stats["max"] // 4
        logger.info(f"  Est. tokens - median: ~{est_tokens_median}, max: ~{est_tokens_max}")

    # Write metadata
    meta = {
        "total_valid": len(valid_rows),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "seed": SEED,
        "invalid_skipped": invalid_count,
        "steps": list(steps.keys()),
    }
    meta_path = OUTPUT_DIR / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"\nMetadata saved to {meta_path}")

    # Save condensed prompts for inference use
    prompts = {
        "step1_pronoun": TRAIN_PROMPT_STEP1,
        "step2_reference": TRAIN_PROMPT_STEP2,
        "step4_extraction": TRAIN_PROMPT_STEP4,
    }
    prompts_path = OUTPUT_DIR / "training_prompts.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logger.info(f"Training prompts saved to {prompts_path}")

    # Verify: read back and check one sample from each
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION: Reading back samples...")
    for step_key, (step_name, _) in steps.items():
        train_path = OUTPUT_DIR / f"{step_key}_train.jsonl"
        with open(train_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sample = json.loads(first_line)
        convs = sample["conversations"]
        assert len(convs) == 3, f"Expected 3 turns, got {len(convs)}"
        assert convs[0]["role"] == "system"
        assert convs[1]["role"] == "user"
        assert convs[2]["role"] == "assistant"
        assert len(convs[0]["content"]) > 50, "System prompt too short"
        assert len(convs[1]["content"]) > 10, "User input too short"
        assert len(convs[2]["content"]) > 10, "Assistant output too short"

        if step_key == "step4_extraction":
            parsed = json.loads(convs[2]["content"])
            assert "triplets" in parsed
            assert len(parsed["triplets"]) > 0
            logger.info(f"  {step_name}: OK (triplets={len(parsed['triplets'])})")
        else:
            logger.info(f"  {step_name}: OK (assistant len={len(convs[2]['content'])})")

    logger.info("\nAll datasets created and verified successfully!")


if __name__ == "__main__":
    main()
