#!/usr/bin/env python3
"""Train a QLoRA adapter with Unsloth on chat-formatted JSONL datasets.

Expected dataset schema (one JSON object per line):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

from datasets import load_dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA/QLoRA with Unsloth on local JSONL datasets")
    parser.add_argument("--train", required=True, help="Path to train JSONL")
    parser.add_argument("--val", default=None, help="Path to val JSONL (optional)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--output-dir", default="models/qwen25-7b-groupchat-lora", help="Output directory")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--dataset-num-proc", type=int, default=4, help="Dataset mapping processes")
    parser.add_argument("--save-steps", type=int, default=200, help="Save every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="chatfun-lora", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (optional)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.load_in_4bit and args.no_load_in_4bit:
        raise SystemExit("Pass only one of --load-in-4bit or --no-load-in-4bit")

    load_in_4bit = True
    if args.no_load_in_4bit:
        load_in_4bit = False
    elif args.load_in_4bit:
        load_in_4bit = True

    train_path = Path(args.train)
    val_path = Path(args.val) if args.val else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    report_to = "none"
    if args.wandb:
        report_to = "wandb"
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    data_files = {"train": str(train_path)}
    if val_path:
        data_files["validation"] = str(val_path)
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

    def formatting_prompts_func(batch: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]
        return {"text": texts}

    train_dataset = raw["train"].map(
        formatting_prompts_func,
        batched=True,
        num_proc=args.dataset_num_proc,
        remove_columns=raw["train"].column_names,
    )
    val_dataset = None
    if "validation" in raw:
        val_dataset = raw["validation"].map(
            formatting_prompts_func,
            batched=True,
            num_proc=args.dataset_num_proc,
            remove_columns=raw["validation"].column_names,
        )

    sft_kwargs = {
        "output_dir": str(output_dir),
        "packing": False,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": report_to,
        "run_name": args.wandb_run_name,
        "seed": args.seed,
    }
    sft_sig = inspect.signature(SFTConfig.__init__)
    if "max_seq_length" in sft_sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_sig.parameters:
        sft_kwargs["max_length"] = args.max_seq_length

    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": sft_config,
        "dataset_text_field": "text",
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    train_output = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metrics = train_output.metrics
    if val_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"Done. Adapter saved to: {output_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
