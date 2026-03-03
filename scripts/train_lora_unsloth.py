#!/usr/bin/env python3
"""Stable LoRA/QLoRA training script using Unsloth + Transformers Trainer.

This avoids TRL SFTTrainer API drift across versions.
Expected dataset format: JSONL rows with `messages` chat arrays.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA/QLoRA with Unsloth + Transformers Trainer")
    p.add_argument("--train", required=True, help="Path to train JSONL")
    p.add_argument("--val", default=None, help="Path to val JSONL")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    p.add_argument("--output-dir", default="models/qwen25-7b-groupchat-lora", help="Output directory")
    p.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    p.add_argument("--epochs", type=float, default=2.0, help="Train epochs")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    p.add_argument("--eval-batch-size", type=int, default=1, help="Per-device eval batch size")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--seed", type=int, default=3407, help="Random seed")
    p.add_argument("--save-steps", type=int, default=200, help="Save every N steps")
    p.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    p.add_argument("--dataset-num-proc", type=int, default=4, help="Dataset map workers")
    p.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    p.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit load")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="chatfun-lora", help="W&B project")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.load_in_4bit and args.no_load_in_4bit:
        raise SystemExit("Pass only one of --load-in-4bit or --no-load-in-4bit")

    load_in_4bit = not args.no_load_in_4bit
    if args.load_in_4bit:
        load_in_4bit = True

    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {"train": args.train}
    if args.val:
        data_files["validation"] = args.val
    raw = load_dataset("json", data_files=data_files)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(batch: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]
        return {"text": texts}

    text_train = raw["train"].map(
        to_text,
        batched=True,
        num_proc=args.dataset_num_proc,
        remove_columns=raw["train"].column_names,
    )

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_ds = text_train.map(
        tokenize_batch,
        batched=True,
        num_proc=args.dataset_num_proc,
        remove_columns=text_train.column_names,
    )

    eval_ds = None
    if "validation" in raw:
        text_eval = raw["validation"].map(
            to_text,
            batched=True,
            num_proc=args.dataset_num_proc,
            remove_columns=raw["validation"].column_names,
        )
        eval_ds = text_eval.map(
            tokenize_batch,
            batched=True,
            num_proc=args.dataset_num_proc,
            remove_columns=text_eval.column_names,
        )

    report_to = ["wandb"] if args.wandb else []
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    ta_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "report_to": report_to,
        "run_name": args.wandb_run_name,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
    }

    if eval_ds is not None:
        ta_kwargs["evaluation_strategy"] = "steps"
        ta_kwargs["eval_steps"] = args.save_steps

    allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in allowed}
    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    train_output = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = dict(train_output.metrics)
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"Done. Adapter saved to: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
