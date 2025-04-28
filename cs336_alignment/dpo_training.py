# dpo_timebound_with_loss_fn.py

import time
import itertools
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
import wandb

from compute_dpo import compute_per_instance_dpo_loss  # your existing loss fn

# — hyperparameters —
β = 0.1
BATCH_SIZE = 1
LR = 1e-5
MAX_LEN = 512
TRAIN_DURATION_SEC = 30 * 60    # 30 minutes
VALIDATION_INTERVAL = 50        # steps

device = torch.device("cuda:0")


def evaluate_val(model, ref_model, tokenizer, val_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            # each batch is a list of dicts: {"chosen": str, "rejected": str}
            for ex in batch:
                chosen  = ex["chosen"]
                rejected= ex["rejected"]

                # compute DPO loss on (chosen vs rejected)
                loss_pos = compute_per_instance_dpo_loss(
                    lm=model,
                    lm_ref=ref_model,
                    tokenizer=tokenizer,
                    beta=1.0,
                    prompt="",                # no prompt in HH preference data
                    response_chosen=chosen,
                    response_rejected=rejected,
                )
                # reversed roles
                loss_neg = compute_per_instance_dpo_loss(
                    lm=model,
                    lm_ref=ref_model,
                    tokenizer=tokenizer,
                    beta=1.0,
                    prompt="",
                    response_chosen=rejected,
                    response_rejected=chosen,
                )

                # smaller loss ⇒ model “prefers” that response
                if loss_pos < loss_neg:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def main():
    # init W&B
    wandb.init(
        project="dpo-training",
        name="llama3-dpo-timebound",
        config={
            "beta": β,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "max_length": MAX_LEN,
            "train_duration_min": TRAIN_DURATION_SEC / 60,
            "validation_interval": VALIDATION_INTERVAL,
        }
    )

    # load tokenizer + models on a single GPU
    tokenizer = AutoTokenizer.from_pretrained("sft_qwen_checkpoint")
    model     = AutoModelForCausalLM.from_pretrained("sft_qwen_checkpoint").to(device)
    ref_model = AutoModelForCausalLM.from_pretrained("sft_qwen_checkpoint").to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # load HH preference data (only has 'chosen' & 'rejected')
    ds = load_dataset("anthropic/hh-rlhf", split="train")
    val_split = int(0.9 * len(ds))
    train_ds = ds.select(range(0, val_split))
    val_ds   = ds.select(range(val_split, len(ds)))

    # collate_fn returns the raw list so each batch is list[dict]
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda examples: examples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=lambda examples: examples,
    )

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader),
    )

    best_val_acc = 0.0
    step = 0
    t0 = time.time()

    # time-bound training loop
    for batch in itertools.cycle(train_loader):
        if time.time() - t0 > TRAIN_DURATION_SEC:
            break

        model.train()
        optimizer.zero_grad()

        batch_loss = torch.tensor(0.0, device=device)
        for ex in batch:
            chosen   = ex["chosen"]
            rejected = ex["rejected"]
            loss = compute_per_instance_dpo_loss(
                lm=model,
                lm_ref=ref_model,
                tokenizer=tokenizer,
                beta=β,
                prompt="",
                response_chosen=chosen,
                response_rejected=rejected,
            )
            batch_loss = batch_loss + loss

        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        # log training loss
        wandb.log({"train/loss": batch_loss.item(), "step": step})

        # periodic validation & checkpoint
        if step % VALIDATION_INTERVAL == 0:
            val_acc = evaluate_val(model, ref_model, tokenizer, val_loader)
            wandb.log({"validation/accuracy": val_acc, "step": step})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained("best_dpo_qwen")

    # final best accuracy
    wandb.log({"best/validation_accuracy": best_val_acc})
    print(f"Done 30 min. Best val-accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
