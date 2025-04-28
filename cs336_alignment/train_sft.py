#!/usr/bin/env python3
"""
Supervised fine-tuning script for instruction-tuning a causal LM, with Weights & Biases logging.

Features:
- Configurable hyperparameters via CLI
- Gradient accumulation for large effective batch sizes
- Cosine learning rate decay with linear warmup
- Optional validation split and loss logging
- Integrated Weights & Biases support
"""
import os
import argparse
import random
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# Import W&B
import wandb

from sft_data import get_packed_sft_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Tuning (SFT) Training Script with W&B")
    parser.add_argument("--model-path",      required=True,  help="Path to base model directory")
    parser.add_argument("--data-path",       required=True,  help="Path to instruction tuning JSONL(.gz)")
    parser.add_argument("--output-dir",      default="sft_output", help="Dir to save fine-tuned model")
    parser.add_argument("--seq-length",      type=int, default=512, help="Context length for training")
    parser.add_argument("--batch-size",      type=int, default=1,   help="Micro-batch size per device")
    parser.add_argument("--grad-accum-steps",type=int, default=32,  help="Gradient accumulation steps")
    parser.add_argument("--learning-rate",   type=float, default=2e-5,help="Initial learning rate")
    parser.add_argument("--max-steps",        type=int, default=None, help="Maximum number of optimizer steps to run (overrides epochs)")
    parser.add_argument("--epochs",          type=int, default=1,   help="Number of training epochs")
    parser.add_argument("--warmup-ratio",    type=float, default=0.03, help="Warmup ratio of total steps")
    parser.add_argument("--seed",            type=int, default=42,  help="Random seed")
    parser.add_argument("--validate",        action="store_true", help="Perform validation each epoch")
    parser.add_argument("--log-interval",    type=int, default=50,   help="Print and log loss every N steps")
    parser.add_argument("--wandb-project",   type=str, default=None, help="W&B project name (disable if None)")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize W&B if requested
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Prepare output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and model (float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
    ).cuda()

    # Prepare dataset
    dataset = get_packed_sft_dataset(
        tokenizer, args.data_path, args.seq_length, shuffle=True
    )
    # Optionally split for validation
    if args.validate:
        val_size = int(0.05 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if args.validate else None

    # Compute scheduling
    if args.max_steps is not None:
        total_train_steps = args.max_steps
    else:
        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
        total_train_steps = steps_per_epoch * args.epochs

    warmup_steps = int(total_train_steps * args.warmup_ratio)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0
    stop_training = False

    for epoch in range(1, args.epochs+1):
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].cuda()
            labels    = batch['labels'].cuda()
            outputs   = model(input_ids, labels=labels)
            loss      = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += outputs.loss.item()

            # Gradient accumulation step
            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.log_interval == 0:
                    avg = running_loss / args.log_interval
                    lr  = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch} | Step {global_step}/{total_train_steps} | loss {avg:.4f} | lr {lr:.2e}")
                    if args.wandb_project:
                        wandb.log({"train/loss": avg, "train/lr": lr}, step=global_step)
                    running_loss = 0.0

                    # Validation
                    if args.validate:
                        model.eval()
                        val_loss = 0.0
                        val_steps = 0
                        with torch.no_grad():
                            for vb in val_loader:
                                vi = vb['input_ids'].cuda()
                                vl = vb['labels'].cuda()
                                out = model(vi, labels=vl)
                                val_loss += out.loss.item()
                                val_steps += 1
                        avg_val = val_loss / max(val_steps,1)
                        print(f"*** Epoch {epoch} Validation loss: {avg_val:.4f}")
                        if args.wandb_project:
                            wandb.log({"eval/loss": avg_val, "eval/epoch": epoch}, step=global_step)
                        model.train()

                if global_step >= total_train_steps:
                    stop_training = True
                    break


        if stop_training:
            print(f"Reached max_steps={total_train_steps}, stopping training.")
            break

    # Save model and tokenizer
    print(f"Saving fine-tuned model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    if args.wandb_project:
        wandb.finish()

if __name__ == '__main__':
    main()
