#!/usr/bin/env python3
"""Simulate a group chat by sampling a random speaker each turn."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from unsloth import FastLanguageModel

TARGET_SPEAKER_RE = re.compile(r"TARGET_SPEAKER=([^\n\r]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate group chat with random speaker per turn")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--turns", type=int, default=20, help="Number of generated turns")
    p.add_argument("--speakers", default=None, help="Comma-separated speaker names (overrides speaker-source)")
    p.add_argument(
        "--speaker-source",
        default="datasets_next_turn/train.jsonl",
        help="Dataset JSONL used to infer speakers when --speakers is not provided",
    )
    p.add_argument("--context", default=None, help="Optional path to initial transcript text file")
    p.add_argument("--seed-line", default="arnav: hey guys", help="Initial line if context is empty")
    p.add_argument("--output", default=None, help="Optional output transcript file")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible speaker sampling")
    p.add_argument("--max-seq-length", type=int, default=3072, help="Max context length")
    p.add_argument("--max-new-tokens", type=int, default=96, help="Max generated tokens per turn")
    p.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    p.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    p.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit loading")
    return p.parse_args()


def infer_speakers(path: Path) -> list[str]:
    speakers: set[str] = set()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = row.get("meta", {})
            target = meta.get("target_speaker") if isinstance(meta, dict) else None
            if isinstance(target, str) and target.strip():
                speakers.add(target.strip())
                continue

            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                match = TARGET_SPEAKER_RE.search(content)
                if match:
                    speakers.add(match.group(1).strip())
    return sorted(speakers)


def build_prompt(context_lines: list[str], target_speaker: str) -> list[dict]:
    transcript = "\n".join(context_lines).strip() or "(empty)"
    return [
        {
            "role": "system",
            "content": (
                "You are simulating a friends group chat. Continue naturally with realistic speaker voices, "
                "casual tone, and speaker-tagged lines in the format 'Name: message'."
            ),
        },
        {
            "role": "user",
            "content": (
                "Group chat transcript so far:\n"
                f"{transcript}\n\n"
                f"TARGET_SPEAKER={target_speaker}\n"
                "Write exactly the next single message text from that speaker. Do not include speaker name."
            ),
        },
    ]


def sanitize_reply(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "..."
    first = lines[0]
    if first.startswith("TARGET_SPEAKER="):
        return lines[1] if len(lines) > 1 else "..."
    if ":" in first and len(first.split(":", 1)[0].split()) <= 3:
        first = first.split(":", 1)[1].strip() or "..."
    return first


def main() -> int:
    args = parse_args()
    if args.load_in_4bit and args.no_load_in_4bit:
        raise SystemExit("Pass only one of --load-in-4bit or --no-load-in-4bit")
    load_in_4bit = not args.no_load_in_4bit
    if args.load_in_4bit:
        load_in_4bit = True

    if args.speakers:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:
        speakers = infer_speakers(Path(args.speaker_source))
    if not speakers:
        raise SystemExit("No speakers found. Provide --speakers or a valid --speaker-source dataset.")

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
        context_text = Path(args.context).read_text(encoding="utf-8")
        context_lines = [line.strip() for line in context_text.splitlines() if line.strip()]
    if not context_lines and args.seed_line.strip():
        context_lines.append(args.seed_line.strip())

    out_path = Path(args.output) if args.output else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else model.device
    print(f"Loaded {len(speakers)} speakers. Generating {args.turns} turns...")

    for i in range(args.turns):
        speaker = rng.choice(speakers)
        messages = build_prompt(context_lines, speaker)
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
        reply = sanitize_reply(generated)
        line = f"{speaker}: {reply}"
        context_lines.append(line)
        print(line)

        if out_path:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
