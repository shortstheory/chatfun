#!/usr/bin/env python3
"""Evaluate a base model + LoRA adapter on a chat JSONL test set."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on a held-out JSONL test set")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--adapter", default=None, help="Path to one LoRA adapter directory")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to one training run directory containing checkpoint-* folders and optionally adapter/",
    )
    parser.add_argument("--test", required=True, help="Path to test JSONL")
    parser.add_argument("--out", default="processed/eval_predictions.jsonl", help="Prediction output JSONL")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max context length")
    parser.add_argument("--min-new-tokens", type=int, default=0, help="Minimum generated tokens")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N examples")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit loading")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text).lower())


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_words(reference)
    pred_tokens = tokenize_words(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(ref_counts[token], pred_counts[token]) for token in ref_counts.keys() & pred_counts.keys())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_examples(path: Path, limit: int | None) -> list[dict]:
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
                continue
            examples.append(row)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def generate_prediction(
    model,
    tokenizer,
    prompt_messages: list[dict],
    device: str,
    min_new_tokens: int,
    max_new_tokens: int | None,
    temperature: float,
    top_p: float,
) -> str:
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    generate_kwargs = {
        **inputs,
        "min_new_tokens": min_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = max_new_tokens
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**generate_kwargs)

    generated_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def find_eval_targets(model_dir: Path) -> list[Path]:
    if not model_dir.exists() or not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    targets: list[Path] = []
    adapter_dir = model_dir / "adapter"
    if adapter_dir.is_dir():
        targets.append(adapter_dir)

    checkpoints = sorted(
        [path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda path: int(path.name.split("-", 1)[1]) if path.name.split("-", 1)[1].isdigit() else path.name,
    )
    targets.extend(checkpoints)

    if not targets:
        raise SystemExit(f"No adapter/ or checkpoint-* directories found in {model_dir}")
    return targets


def evaluate_examples(
    model,
    tokenizer,
    examples: list[dict],
    out_path: Path,
    device: str,
    min_new_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    verbose: bool,
) -> dict:
    total = 0
    exact_matches = 0
    exact_matches_normalized = 0
    token_f1_sum = 0.0
    reference_words = 0
    prediction_words = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            messages = example["messages"]
            prompt_messages = messages[:-1]
            reference = str(messages[-1].get("content", "")).strip()
            prediction = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                device=device,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            total += 1
            exact_match = prediction == reference
            normalized_exact_match = normalize_text(prediction) == normalize_text(reference)
            token_f1_value = token_f1(reference, prediction)
            if exact_match:
                exact_matches += 1
            if normalized_exact_match:
                exact_matches_normalized += 1
            token_f1_sum += token_f1_value
            reference_words += len(tokenize_words(reference))
            prediction_words += len(tokenize_words(prediction))

            result = {
                "index": index,
                "prediction": prediction,
                "reference": reference,
                "exact_match": exact_match,
                "normalized_exact_match": normalized_exact_match,
                "token_f1": token_f1_value,
                "meta": example.get("meta", {}),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if verbose:
                print(
                    f"[{index}/{len(examples)}] "
                    f"norm_em={result['normalized_exact_match']} "
                    f"token_f1={result['token_f1']:.3f}"
                )

    return {
        "examples": total,
        "exact_match": exact_matches / total,
        "normalized_exact_match": exact_matches_normalized / total,
        "avg_token_f1": token_f1_sum / total,
        "avg_reference_words": reference_words / total,
        "avg_prediction_words": prediction_words / total,
        "predictions_saved_to": str(out_path),
    }


def print_metrics(metrics: dict) -> None:
    print()
    print(f"examples={metrics['examples']}")
    print(f"exact_match={metrics['exact_match']:.4f}")
    print(f"normalized_exact_match={metrics['normalized_exact_match']:.4f}")
    print(f"avg_token_f1={metrics['avg_token_f1']:.4f}")
    print(f"avg_reference_words={metrics['avg_reference_words']:.2f}")
    print(f"avg_prediction_words={metrics['avg_prediction_words']:.2f}")
    print(f"predictions_saved_to={metrics['predictions_saved_to']}")


def main() -> int:
    args = parse_args()
    if args.load_in_4bit and args.no_load_in_4bit:
        raise SystemExit("Pass only one of --load-in-4bit or --no-load-in-4bit")
    if bool(args.adapter) == bool(args.model_dir):
        raise SystemExit("Pass exactly one of --adapter or --model-dir")

    load_in_4bit = not args.no_load_in_4bit
    if args.load_in_4bit:
        load_in_4bit = True

    test_path = Path(args.test)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_examples(test_path, args.limit)
    if not examples:
        raise SystemExit(f"No valid examples found in {test_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else model.device

    if args.model_dir:
        targets = find_eval_targets(Path(args.model_dir))
        summary_rows: list[dict] = []

        for target in targets:
            print(f"\n=== Evaluating {target} ===")
            peft_model = PeftModel.from_pretrained(model, str(target))
            FastLanguageModel.for_inference(peft_model)

            batch_out = target / out_path.name
            metrics = evaluate_examples(
                model=peft_model,
                tokenizer=tokenizer,
                examples=examples,
                out_path=batch_out,
                device=device,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                verbose=False,
            )
            print_metrics(metrics)
            summary_rows.append({"target": str(target), **metrics})

            model = peft_model.unload()

        summary_rows.sort(key=lambda row: row["avg_token_f1"], reverse=True)
        print("\nSummary:")
        print(f"{'rank':<4} {'token_f1':>8} {'norm_em':>8} {'pred_w':>8} target")
        print("-" * 72)
        for rank, row in enumerate(summary_rows, start=1):
            print(
                f"{rank:<4} "
                f"{row['avg_token_f1']:>8.4f} "
                f"{row['normalized_exact_match']:>8.4f} "
                f"{row['avg_prediction_words']:>8.2f} "
                f"{row['target']}"
            )
        return 0

    peft_model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(peft_model)
    metrics = evaluate_examples(
        model=peft_model,
        tokenizer=tokenizer,
        examples=examples,
        out_path=out_path,
        device=device,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=True,
    )
    print_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
