#!/usr/bin/env python3
"""Stable LoRA/QLoRA training script using Unsloth + Transformers Trainer.

This avoids TRL SFTTrainer API drift across versions.
Expected dataset format: JSONL rows with `messages` chat arrays.
Legacy `text` rows are also supported for older flat-text datasets.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA/QLoRA with Unsloth + Transformers Trainer")
    p.add_argument("--train", required=True, help="Path to train JSONL")
    p.add_argument("--val", default=None, help="Path to val JSONL")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    p.add_argument("--output-dir", default="models/qwen25-7b-groupchat-lora", help="Output directory")
    p.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    p.add_argument("--epochs", type=float, default=2.0, help="Train epochs")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    p.add_argument("--eval-batch-size", type=int, default=1, help="Per-device eval batch size")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--seed", type=int, default=3407, help="Random seed")
    p.add_argument("--save-steps", type=int, default=200, help="Save every N steps")
    p.add_argument("--eval-steps", type=int, default=None, help="Evaluate every N steps (defaults to save_steps)")
    p.add_argument("--logging-steps", type=int, default=1, help="Log every N steps")
    p.add_argument("--dataset-num-proc", type=int, default=4, help="Dataset map workers")
    p.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    p.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit load")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="chatfun-lora", help="W&B project")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name")
    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to checkpoint directory to resume training from (e.g. .../checkpoint-7000)",
    )
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

    def _normalize_token_ids(rendered: Any) -> list[int]:
        if isinstance(rendered, dict):
            input_ids = rendered.get("input_ids")
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                return list(input_ids[0])
            if isinstance(input_ids, list):
                return list(input_ids)
            raise ValueError("Chat template returned dict without usable input_ids")
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if isinstance(rendered, list) and rendered and isinstance(rendered[0], list):
            return list(rendered[0])
        if isinstance(rendered, list):
            return list(rendered)
        raise ValueError(f"Unsupported chat template output type: {type(rendered).__name__}")

    def _validate_message_row(messages: Any, row_index: int) -> list[dict[str, Any]]:
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Invalid messages at batch row {row_index}: expected at least 2 messages")
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError(f"Invalid messages at batch row {row_index}: each message must be an object")
        return messages

    def tokenize_messages(batch: dict) -> dict:
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for row_index, messages in enumerate(batch["messages"]):
            messages = _validate_message_row(messages, row_index)

            prompt_messages = messages[:-1]
            full_ids = _normalize_token_ids(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            prompt_ids = _normalize_token_ids(
                tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )

            full_ids = full_ids[: args.max_seq_length]
            prompt_len = min(len(prompt_ids), len(full_ids), args.max_seq_length)
            labels = ([-100] * prompt_len) + full_ids[prompt_len:]

            all_input_ids.append(full_ids)
            all_attention_masks.append([1] * len(full_ids))
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }

    def tokenize_text(batch: dict) -> dict:
        text_rows = batch.get("text")
        if not isinstance(text_rows, list):
            raise ValueError("Expected batched `text` column to be a list of strings")

        encoded = tokenizer(
            text=text_rows,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        input_ids = [list(ids) for ids in encoded["input_ids"]]
        attention_masks = [list(mask) for mask in encoded["attention_mask"]]
        labels = [list(ids) for ids in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    train_columns = set(raw["train"].column_names)
    if "messages" in train_columns:
        tokenize_fn = tokenize_messages
    elif "text" in train_columns:
        tokenize_fn = tokenize_text
    else:
        raise SystemExit(
            "Unsupported dataset format. Expected a `messages` column or legacy `text` column, "
            f"but found: {sorted(train_columns)}"
        )

    train_ds = raw["train"].map(
        tokenize_fn,
        batched=True,
        num_proc=args.dataset_num_proc,
        remove_columns=raw["train"].column_names,
    )

    eval_ds = None
    if "validation" in raw:
        validation_columns = set(raw["validation"].column_names)
        if validation_columns != train_columns:
            raise SystemExit(
                "Train/validation schema mismatch. "
                f"train={sorted(train_columns)} validation={sorted(validation_columns)}"
            )
        eval_ds = raw["validation"].map(
            tokenize_fn,
            batched=True,
            num_proc=args.dataset_num_proc,
            remove_columns=raw["validation"].column_names,
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

    allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if eval_ds is not None:
        ta_kwargs["do_eval"] = True
        ta_kwargs["eval_steps"] = args.eval_steps if args.eval_steps is not None else args.save_steps
        if "evaluation_strategy" in allowed:
            ta_kwargs["evaluation_strategy"] = "steps"
        elif "eval_strategy" in allowed:
            ta_kwargs["eval_strategy"] = "steps"

    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in allowed}
    training_args = TrainingArguments(**ta_kwargs)
    if eval_ds is not None:
        strategy = getattr(training_args, "evaluation_strategy", None)
        if strategy is None:
            strategy = getattr(training_args, "eval_strategy", None)
        print(f"Eval enabled. strategy={strategy} eval_steps={getattr(training_args, 'eval_steps', None)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
    )

    train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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
