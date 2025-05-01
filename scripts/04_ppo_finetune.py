# AI-Assisted (2025-04-18): initial PPO fine-tuning script for taxonomy merge preferences.
# Review focus: device management, reward design, logging.

"""04_ppo_finetune.py — Fine-tune LLM via PPO on taxonomy merge preferences.

This script reads:
  - `data/taxo_raw.csv`: original taxonomy labels.
  - `data/merge_pairs.csv`: merge pairs (source_id, target_id).
It creates a PPO training dataset where the model learns to prefer correct merges over negative samples.

Usage:
    python scripts/04_ppo_finetune.py \
      --model google/gemma-2b-it \
      --batch-size 4 \
      --learning-rate 1e-5 \
      --epochs 3 \
      --output-dir models/ppo_finetuned \
      [--dry-run]

MANDATORY CUSTOMIZATION REQUIRED! Adapt dataset loading, reward function, and PPO config to your project.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_taxonomy(path: Path) -> List[str]:
    """Load and concatenate taxonomy labels from CSV."""
    df = pd.read_csv(path, sep=",", header=None, dtype=str, keep_default_na=False)
    return [
        " | ".join([c for c in row if isinstance(c, str) and c.strip()])
        for row in df.values.tolist()
    ]


def load_merge_pairs(path: Path) -> List[Tuple[int, int]]:
    """Load merge pairs (source_id, target_id) for positive examples."""
    df = pd.read_csv(path)
    return list(zip(df.source_id.astype(int), df.target_id.astype(int)))


def build_prompts(labels: List[str], merge_pairs: List[Tuple[int, int]]) -> List[str]:
    """Build a list of prompt strings for PPO training."""
    prompts: List[str] = []
    for src, tgt in merge_pairs:
        prompts.append(f"Should '{labels[src]}' be merged into '{labels[tgt]}'? Answer 'Yes' or 'No'.")
    return prompts


def compute_reward(response: str, is_positive: bool) -> float:
    """Compute reward based on model response and expected outcome."""
    resp = response.strip().lower()
    correct = resp.startswith("yes")
    return 1.0 if (correct and is_positive) or (not correct and not is_positive) else -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLM via PPO on taxonomy merge preferences.")
    parser.add_argument("--model", default="google/gemma-2b-it", help="HuggingFace model name.")
    parser.add_argument("--batch-size", type=int, default=4, help="PPO batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="PPO learning rate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/ppo_finetuned"), help="Where to save the fine-tuned model.")
    parser.add_argument("--dry-run", action="store_true", help="Print settings and exit.")
    parser.add_argument("--resume-from", type=str, help="Path to PPO checkpoint to resume training from.")
    args = parser.parse_args()

    if args.dry_run:
        print(vars(args))
        return

    # Prepare checkpoint resume
    start_epoch = 0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    log_path = Path("logs/ppo_finetuning.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting PPO fine-tuning on device={device}")

    # Load tokenizer and models with value heads required by PPO
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    base_model.config.pad_token_id = tokenizer.eos_token_id

    # PPO setup
    ppo_config = PPOConfig(
        model_name=args.model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_epochs=1,
        mini_batch_size=args.batch_size
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # Resume training from checkpoint if provided
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        base_model.load_state_dict(checkpoint["model"])  # type: ignore
        ref_model.load_state_dict(checkpoint["ref_model"])  # type: ignore
        ppo_trainer.optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore
        start_epoch = checkpoint["epoch"] + 1  # type: ignore
        logging.getLogger(__name__).info(f"Resuming PPO from epoch {start_epoch}")

    # Data loading
    labels = load_taxonomy(Path("data/taxo_raw.csv"))
    merge_pairs = load_merge_pairs(Path("data/merge_pairs.csv"))
    prompts = build_prompts(labels, merge_pairs)

    # Ensure models on correct device
    base_model.to(device)
    ref_model.to(device)

    # Training loop (batched via Hugging Face tokenizer)
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            # Tokenize batch
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            input_ids = enc["input_ids"]  # (batch_size, seq_len)
            attention_mask = enc["attention_mask"]

            # Generate responses on GPU
            response_ids = ppo_trainer.generate(input_ids, attention_mask=attention_mask)

            # Decode text responses
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            # Compute rewards (positive examples)
            rewards = [compute_reward(resp, is_positive=True) for resp in responses]

            # PPO optimization step accepts batch tensors
            stats = ppo_trainer.step(input_ids, response_ids, rewards)

            # Log stats manually
            step_index = epoch * (len(prompts) // args.batch_size + 1) + (start // args.batch_size)
            logger.info(f"Step {step_index}: reward mean={sum(rewards)/len(rewards):.3f}")
            ppo_trainer.log_stats(stats, names=["reward"], step=step_index)

            # Save checkpoint after each epoch
            ckpt_dir = args.output_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": base_model.state_dict(),
                "ref_model": ref_model.state_dict(),
                "optimizer": ppo_trainer.optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved PPO checkpoint to {ckpt_path}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"PPO fine-tuning complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main() 