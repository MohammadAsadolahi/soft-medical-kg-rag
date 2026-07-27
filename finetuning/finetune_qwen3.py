#!/usr/bin/env python
"""
finetune_qwen3.py — QLoRA Fine-Tuning for Medical Extraction Pipeline
======================================================================
Fine-tunes Qwen3-8B with QLoRA using Unsloth for one of three pipeline steps:
  --step 1: Pronoun Resolution
  --step 2: Indirect Reference Transfer
  --step 4: Relation Extraction

Requires a CUDA GPU. Build the datasets first with create_training_datasets.py.

Usage:
  python finetuning/finetune_qwen3.py --step 1      # pronoun resolution
  python finetuning/finetune_qwen3.py --step 2      # indirect reference transfer
  python finetuning/finetune_qwen3.py --step 4      # relation extraction
  python finetuning/finetune_qwen3.py --step all    # all three, sequentially

This is the local single-GPU driver. The Kaggle notebook
(qwen3_decontextualization_qlora.ipynb) is the more developed path: dual-T4 DDP, resumable
from a checkpoint dataset, and it carries the full span-level evaluation of stage 1.
See README.md in this directory for why stage 1 needs unusual hyperparameters.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "training_data"
OUTPUT_BASE = ROOT / "finetuned_models"

# Step → dataset file prefix and max_seq_length
STEP_CONFIG = {
    1: {
        "name": "step1_pronoun",
        "desc": "Pronoun Resolution",
        "max_seq_length": 2048,
    },
    2: {
        "name": "step2_reference",
        "desc": "Indirect Reference Transfer",
        "max_seq_length": 2048,
    },
    4: {
        "name": "step4_extraction",
        "desc": "Relation Extraction",
        "max_seq_length": 4096,
    },
}


def check_dependencies():
    """Verify required packages are installed."""
    missing = []
    try:
        import unsloth  # noqa: F401
    except ImportError:
        missing.append("unsloth")
    try:
        import trl  # noqa: F401
    except ImportError:
        missing.append("trl")
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")

    if missing:
        logger.error(f"Missing packages: {missing}")
        logger.error("Install with: pip install unsloth")
        sys.exit(1)


def check_gpu():
    """Check GPU availability and VRAM."""
    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA not available. QLoRA requires a CUDA GPU.")
        sys.exit(1)

    device = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9 if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {device} | VRAM: {vram_gb:.1f} GB")

    if vram_gb < 7.5:
        logger.warning(
            f"Low VRAM ({vram_gb:.1f} GB). Qwen3-8B QLoRA needs ~6-8 GB. "
            "If you get OOM, try: --model unsloth/Qwen3-4B-unsloth-bnb-4bit"
        )
    return vram_gb


def load_model(model_name, max_seq_length):
    """Load model with QLoRA quantization via Unsloth."""
    logger.info(f"Loading model: {model_name} (max_seq={max_seq_length})...")
    t0 = time.time()

    try:
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        logger.info(f"Model loaded via FastModel in {time.time()-t0:.1f}s")
    except (ImportError, Exception) as e:
        logger.warning(f"FastModel failed ({e}), trying FastLanguageModel...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        logger.info(f"Model loaded via FastLanguageModel in {time.time()-t0:.1f}s")

    return model, tokenizer


def apply_lora(model, r=8, alpha=8):
    """Apply LoRA adapters to the model."""
    logger.info(f"Applying LoRA (r={r}, alpha={alpha}) to all linear layers...")

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    try:
        from unsloth import FastModel
        model = FastModel.get_peft_model(
            model,
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    except (ImportError, Exception):
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    return model


def load_and_format_dataset(step_name, tokenizer, max_seq_length):
    """Load JSONL dataset and apply chat template formatting."""
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    train_path = str(DATA_DIR / f"{step_name}_train.jsonl")
    eval_path = str(DATA_DIR / f"{step_name}_eval.jsonl")

    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run create_training_datasets.py first."
        )

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "test": eval_path},
    )

    dataset = standardize_sharegpt(dataset)

    def formatting_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset["train"].column_names)

    # Log a sample
    sample = dataset["train"][0]["text"]
    logger.info(f"Sample formatted text (first 500 chars):\n{sample[:500]}...")

    # Token length distribution
    train_lens = []
    for text in dataset["train"]["text"]:
        tokens = tokenizer(text, return_length=True, truncation=False)
        train_lens.append(tokens["length"][0])
    train_lens.sort()
    n = len(train_lens)
    logger.info(
        f"Token lengths — min: {train_lens[0]}, median: {train_lens[n//2]}, "
        f"p90: {train_lens[int(n*0.9)]}, p99: {train_lens[int(n*0.99)]}, max: {train_lens[-1]}"
    )
    over_max = sum(1 for l in train_lens if l > max_seq_length)
    if over_max:
        logger.warning(f"{over_max}/{n} samples exceed max_seq_length={max_seq_length} and will be truncated")

    logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} eval")
    return dataset, tokenizer


def train_step(
    step_num,
    model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
    lora_r=8,
    lora_alpha=8,
    num_epochs=3,
    batch_size=1,
    grad_accum=16,
    lr=1e-4,
    warmup_steps=20,
    eval_steps=25,
    logging_steps=5,
    save_steps=50,
):
    """Train one pipeline step."""
    cfg = STEP_CONFIG[step_num]
    step_name = cfg["name"]
    step_desc = cfg["desc"]
    max_seq_length = cfg["max_seq_length"]
    output_dir = OUTPUT_BASE / step_name

    logger.info("=" * 70)
    logger.info(f"TRAINING: Step {step_num} — {step_desc}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  max_seq_length: {max_seq_length}")
    logger.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    logger.info(f"  Training: epochs={num_epochs}, batch={batch_size}, grad_accum={grad_accum}")
    logger.info(f"  LR: {lr}, warmup: {warmup_steps}, eval every {eval_steps} steps")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70)

    # Load model
    model, tokenizer = load_model(model_name, max_seq_length)

    # Apply LoRA
    model = apply_lora(model, r=lora_r, alpha=lora_alpha)

    # Load dataset
    dataset, tokenizer = load_and_format_dataset(step_name, tokenizer, max_seq_length)

    # Configure trainer
    from trl import SFTTrainer, SFTConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_seq_length=max_seq_length,
        packing=True,
        dataset_text_field="text",
        # Eval
        eval_strategy="steps",
        eval_steps=eval_steps,
        # Logging
        logging_steps=logging_steps,
        logging_first_step=True,
        report_to="none",
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        # Misc
        seed=3407,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
    )

    # Log GPU memory before training
    import torch
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU memory before training — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB")

    # Train
    logger.info("Starting training...")
    t0 = time.time()

    train_result = trainer.train()

    elapsed = time.time() - t0
    logger.info(f"Training completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"Training metrics: {train_result.metrics}")

    # Final eval
    logger.info("Running final evaluation...")
    eval_result = trainer.evaluate()
    logger.info(f"Final eval metrics: {eval_result}")

    # Save LoRA adapter
    adapter_dir = output_dir / "final_adapter"
    logger.info(f"Saving LoRA adapter to {adapter_dir}...")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Save training summary
    summary = {
        "step": step_num,
        "step_name": step_name,
        "description": step_desc,
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "learning_rate": lr,
        "train_samples": len(dataset["train"]),
        "eval_samples": len(dataset["test"]),
        "training_time_seconds": elapsed,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Training summary saved to {summary_path}")

    # Clean up GPU memory thoroughly for sequential training
    del model, trainer, dataset
    import gc
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory after cleanup: {alloc:.2f} GB allocated")
    logger.info(f"Step {step_num} ({step_desc}) — DONE\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tune Qwen3-8B for Extraction Pipeline")
    parser.add_argument("--step", required=True, help="Step to train: 1, 2, 4, or 'all'")
    parser.add_argument("--model", default="unsloth/Qwen3-8B-unsloth-bnb-4bit",
                        help="Model name (default: unsloth/Qwen3-8B-unsloth-bnb-4bit)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha (default: 8)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup steps (default: 20)")
    parser.add_argument("--eval-steps", type=int, default=25, help="Eval every N steps (default: 25)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("QLoRA Fine-Tuning Pipeline for Medical Extraction")
    logger.info("=" * 70)

    # Dependency check
    check_dependencies()
    vram = check_gpu()

    # Determine steps to train
    if args.step.lower() == "all":
        steps_to_train = [1, 2, 4]
    else:
        step_num = int(args.step)
        if step_num not in STEP_CONFIG:
            logger.error(f"Invalid step: {step_num}. Valid: 1, 2, 4, all")
            sys.exit(1)
        steps_to_train = [step_num]

    # Verify datasets exist
    for step_num in steps_to_train:
        cfg = STEP_CONFIG[step_num]
        train_path = DATA_DIR / f"{cfg['name']}_train.jsonl"
        eval_path = DATA_DIR / f"{cfg['name']}_eval.jsonl"
        if not train_path.exists() or not eval_path.exists():
            logger.error(f"Dataset not found for step {step_num}: {train_path}")
            logger.error("Run: python create_training_datasets.py")
            sys.exit(1)

    # Train each step
    all_summaries = []
    for step_num in steps_to_train:
        summary = train_step(
            step_num=step_num,
            model_name=args.model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            warmup_steps=args.warmup,
            eval_steps=args.eval_steps,
        )
        all_summaries.append(summary)

    # Final report
    logger.info("\n" + "=" * 70)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("=" * 70)
    for s in all_summaries:
        logger.info(
            f"  Step {s['step']} ({s['description']}): "
            f"train_loss={s['train_metrics'].get('train_loss', 'N/A'):.4f}, "
            f"eval_loss={s['eval_metrics'].get('eval_loss', 'N/A'):.4f}, "
            f"time={s['training_time_seconds']:.0f}s"
        )
    logger.info("\nFine-tuned adapters saved to:")
    for s in all_summaries:
        adapter_path = OUTPUT_BASE / s["step_name"] / "final_adapter"
        logger.info(f"  Step {s['step']}: {adapter_path}")


if __name__ == "__main__":
    main()
