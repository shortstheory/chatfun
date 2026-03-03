#!/usr/bin/env python3
"""Interactive chat CLI for base model + LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with a LoRA-adapted model")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--max-seq-length", type=int, default=4096, help="Max context length")
    p.add_argument("--max-new-tokens", type=int, default=220, help="Max generated tokens")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    p.add_argument("--speaker", default=None, help="Optional TARGET_SPEAKER value")
    p.add_argument("--context", default=None, help="Optional path to initial transcript text file")
    p.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    p.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit loading")
    return p.parse_args()


def build_user_prompt(context_lines: list[str], speaker: str | None) -> str:
    transcript = "\n".join(context_lines).strip()
    if not transcript:
        transcript = "(empty)"
    if speaker:
        return (
            "Group chat transcript so far:\n"
            f"{transcript}\n\n"
            f"TARGET_SPEAKER={speaker}\n"
            "Continue the conversation naturally. Keep speaker tags in each line: Name: message"
        )
    return (
        "Group chat transcript so far:\n"
        f"{transcript}\n\n"
        "Continue the conversation naturally. Keep speaker tags in each line: Name: message"
    )


def main() -> int:
    args = parse_args()
    if args.load_in_4bit and args.no_load_in_4bit:
        raise SystemExit("Pass only one of --load-in-4bit or --no-load-in-4bit")

    load_in_4bit = not args.no_load_in_4bit
    if args.load_in_4bit:
        load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    context_lines: list[str] = []
    if args.context:
        text = Path(args.context).read_text(encoding="utf-8")
        context_lines.extend([line for line in text.splitlines() if line.strip()])

    print("Interactive mode. Commands: /exit, /clear, /speaker <name>, /show")
    if args.speaker:
        print(f"Initial TARGET_SPEAKER={args.speaker}")

    current_speaker = args.speaker
    device = "cuda" if torch.cuda.is_available() else model.device

    while True:
        try:
            user_line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_line:
            continue
        if user_line == "/exit":
            break
        if user_line == "/clear":
            context_lines.clear()
            print("Context cleared.")
            continue
        if user_line == "/show":
            print("\n".join(context_lines[-20:]) if context_lines else "(empty)")
            continue
        if user_line.startswith("/speaker "):
            current_speaker = user_line[len("/speaker ") :].strip() or None
            print(f"TARGET_SPEAKER={current_speaker}" if current_speaker else "TARGET_SPEAKER cleared")
            continue

        # Add your new live line to context in speaker-tag format if missing.
        if ":" in user_line:
            context_lines.append(user_line)
        else:
            context_lines.append(f"You: {user_line}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are simulating a friends group chat. Continue naturally with realistic speaker voices, "
                    "casual tone, and speaker-tagged lines in the format 'Name: message'."
                ),
            },
            {"role": "user", "content": build_user_prompt(context_lines, current_speaker)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        print(generated)

        # Keep generated lines as rolling context for future turns.
        for line in generated.splitlines():
            line = line.strip()
            if line:
                context_lines.append(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
