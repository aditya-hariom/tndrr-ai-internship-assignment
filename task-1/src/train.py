"""
src/train.py

Robust, transparent PyTorch LoRA fine-tuning engine for Qwen2.5-0.5B-Instruct.
Features:
- Prompt loss masking (loss computed exclusively on target JSON response tokens)
- Architecture verification of LoRA target modules (q_proj, k_proj, v_proj, o_proj)
- Exact parameter accounting (total vs trainable parameters and %)
- Cosine learning rate schedule with linear warmup
- Validation loss monitoring and best-checkpoint selection
- Saves adapter weights compatible with Hugging Face PEFT (.save_pretrained)
- Logs step-level and epoch-level loss metrics for experiment tracking
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, load_yaml_config
from src.model import load_base_model_and_tokenizer, apply_lora, get_device
from src.dataset import JsonActionDataset

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """Generates cosine learning rate decay schedule with initial linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def evaluate_validation_loss(model, val_loader, device):
    """Computes average cross-entropy loss over validation set with prompt loss masking."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)

def train(cfg):
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    device = get_device()

    print("=" * 70)
    print("STARTING LoRA FINE-TUNING PIPELINE")
    print(f"Device: {device} | Model: {cfg['model']['base_model_name_or_path']}")
    print("=" * 70)

    # 1. Load Base Model & Tokenizer
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=cfg["model"]["base_model_name_or_path"]
    )

    # 2. Apply LoRA Adapter
    peft_model, param_stats = apply_lora(model, cfg["lora"])
    peft_model.to(device)

    # 3. Load Pre-tokenized Datasets
    max_seq_len = cfg["training"].get("max_seq_length", 256)
    train_dataset = JsonActionDataset(
        jsonl_path=cfg["paths"]["train_data"],
        tokenizer=tokenizer,
        max_seq_length=max_seq_len
    )
    val_dataset = JsonActionDataset(
        jsonl_path=cfg["paths"]["validation_data"],
        tokenizer=tokenizer,
        max_seq_length=max_seq_len
    )

    train_batch_size = cfg["training"]["per_device_train_batch_size"]
    eval_batch_size = cfg["training"].get("per_device_eval_batch_size", 2)
    grad_accum_steps = cfg["training"]["gradient_accumulation_steps"]

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    # 4. Hyperparameters & Optimizer
    num_epochs = cfg["training"]["num_train_epochs"]
    total_update_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    warmup_steps = int(total_update_steps * cfg["training"]["warmup_ratio"])
    learning_rate = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])

    print(f"\nTraining Schedule:")
    print(f"  * Epochs:                    {num_epochs}")
    print(f"  * Train Samples:             {len(train_dataset)}")
    print(f"  * Batch Size per Device:     {train_batch_size}")
    print(f"  * Gradient Accumulation:     {grad_accum_steps} (Effective Batch Size: {train_batch_size * grad_accum_steps})")
    print(f"  * Total Optimizer Steps:     {total_update_steps}")
    print(f"  * Warmup Steps:              {warmup_steps}")
    print(f"  * Peak Learning Rate:        {learning_rate}")
    print(f"  * Weight Decay:              {weight_decay}")

    # Optimize only trainable LoRA parameters
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial Validation Loss
    print("\nComputing baseline pre-training validation loss...")
    initial_val_loss = evaluate_validation_loss(peft_model, val_loader, device)
    print(f"Initial Validation Loss: {initial_val_loss:.4f}")

    best_val_loss = float("inf")
    best_epoch = -1
    training_history = []
    global_step = 0
    start_time = time.time()

    peft_model.train()
    optimizer.zero_grad()

    # 5. Training Loop
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        accumulated_loss = 0.0

        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")
        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = peft_model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            accumulated_loss += loss.item() * grad_accum_steps
            running_loss += loss.item() * grad_accum_steps

            if step % grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                avg_step_loss = accumulated_loss / grad_accum_steps
                accumulated_loss = 0.0
                pbar.set_postfix({"loss": f"{avg_step_loss:.4f}", "lr": f"{current_lr:.2e}"})

        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = evaluate_validation_loss(peft_model, val_loader, device)
        epoch_duration = time.time() - epoch_start

        print(f"\n[Epoch {epoch} Complete] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Time: {epoch_duration:.1f}s")

        # Save epoch checkpoint
        epoch_ckpt_dir = output_dir / f"checkpoint-epoch-{epoch}"
        peft_model.save_pretrained(epoch_ckpt_dir)
        tokenizer.save_pretrained(epoch_ckpt_dir)

        # Track best model
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_adapter_dir = output_dir / "best_adapter"
            peft_model.save_pretrained(best_adapter_dir)
            tokenizer.save_pretrained(best_adapter_dir)
            print(f"  * New best validation loss achieved ({best_val_loss:.4f})! Saved to {best_adapter_dir}")

        training_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(epoch_val_loss, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "is_best": is_best,
            "duration_seconds": round(epoch_duration, 2)
        })

    total_training_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s")
    print(f"Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
    print("=" * 70)

    # Save comprehensive training log
    metrics_log_path = output_dir / "training_metrics.json"
    with open(metrics_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "param_stats": param_stats,
            "initial_val_loss": round(initial_val_loss, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_epoch": best_epoch,
            "total_duration_seconds": round(total_training_time, 2),
            "history": training_history
        }, f, indent=2)
    print(f"Training metrics saved to: {metrics_log_path}")
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B-Instruct with LoRA")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train(cfg)

if __name__ == "__main__":
    main()
