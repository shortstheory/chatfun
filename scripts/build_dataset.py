#!/usr/bin/env python3
"""Build chunked conversation datasets from parsed WhatsApp turns.

Input: JSONL from parse_whatsapp.py (full format with timestamp/speaker/text)
Output: train/val/test JSONL files using conversation chunks
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class Turn:
    timestamp: str
    speaker: str
    text: str
    source_file: str
    line_start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunked train/val/test JSONL from parsed turns")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to parsed turns JSONL (processed/turns.jsonl)")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory for split JSONL files")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--chunk-turns", type=int, default=32, help="Turns per conversation chunk (default: 32)")
    parser.add_argument("--prompt-turns", type=int, default=16, help="Turns used as prompt context within each chunk")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride in turns (default: 8)")
    parser.add_argument("--min-turns", type=int, default=12, help="Minimum turns required to keep a chunk (default: 12)")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are simulating a friends group chat. Continue naturally with realistic speaker voices, "
            "casual tone, and speaker-tagged lines in the format 'Name: message'."
        ),
        help="System prompt prepended to each sample",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata fields (time/source/chunk indices) in each output record",
    )
    parser.add_argument(
        "--speaker-control",
        action="store_true",
        help="Add TARGET_SPEAKER=<name> control tag using the first completion turn speaker",
    )
    return parser.parse_args()


def load_turns(path: Path) -> list[Turn]:
    turns: list[Turn] = []

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {idx}: {exc}") from exc

            timestamp = row.get("timestamp")
            speaker = row.get("speaker")
            text = row.get("text")
            if not isinstance(timestamp, str) or not isinstance(speaker, str) or not isinstance(text, str):
                continue
            if not text.strip():
                continue

            source_file = str(row.get("source_file", "unknown"))
            line_start_raw = row.get("line_start", 0)
            line_start = int(line_start_raw) if isinstance(line_start_raw, int) or str(line_start_raw).isdigit() else 0

            turns.append(
                Turn(
                    timestamp=timestamp,
                    speaker=speaker.strip(),
                    text=text.strip(),
                    source_file=source_file,
                    line_start=line_start,
                )
            )

    def sort_key(turn: Turn) -> tuple[datetime, str, int]:
        return (datetime.fromisoformat(turn.timestamp), turn.source_file, turn.line_start)

    turns.sort(key=sort_key)
    return turns


def format_turn(turn: Turn) -> str:
    return f"{turn.speaker}: {turn.text}"


def iter_windows(turns: list[Turn], chunk_turns: int, stride: int, min_turns: int) -> Iterable[tuple[int, list[Turn]]]:
    if len(turns) < min_turns:
        return

    produced = set()
    max_start = max(0, len(turns) - min_turns)

    start = 0
    while start <= max_start:
        window = turns[start : start + chunk_turns]
        if len(window) >= min_turns:
            produced.add(start)
            yield start, window
        start += stride

    # Ensure the tail of the timeline is represented.
    tail_start = max(0, len(turns) - chunk_turns)
    if tail_start not in produced:
        window = turns[tail_start : tail_start + chunk_turns]
        if len(window) >= min_turns:
            yield tail_start, window


def build_examples(
    turns: list[Turn],
    chunk_turns: int,
    prompt_turns: int,
    stride: int,
    min_turns: int,
    system_prompt: str,
    include_metadata: bool,
    speaker_control: bool,
) -> list[dict]:
    examples: list[dict] = []

    for chunk_index, (start_idx, window) in enumerate(iter_windows(turns, chunk_turns, stride, min_turns)):
        split_idx = min(prompt_turns, len(window) - 1)
        if split_idx < 1:
            continue

        prompt_slice = window[:split_idx]
        completion_slice = window[split_idx:]
        if not completion_slice:
            continue

        prompt_text = "\n".join(format_turn(t) for t in prompt_slice)
        completion_text = "\n".join(format_turn(t) for t in completion_slice)
        target_speaker = completion_slice[0].speaker

        if speaker_control:
            user_content = (
                "Group chat transcript so far:\n"
                f"{prompt_text}\n\n"
                f"TARGET_SPEAKER={target_speaker}\n"
                "Continue the conversation naturally. Keep speaker tags in each line: Name: message"
            )
        else:
            user_content = (
                "Group chat transcript so far:\n"
                f"{prompt_text}\n\n"
                "Continue the conversation naturally. Keep speaker tags in each line: Name: message"
            )

        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": completion_text},
            ]
        }

        if include_metadata:
            record["meta"] = {
                "chunk_index": chunk_index,
                "start_turn_index": start_idx,
                "end_turn_index": start_idx + len(window) - 1,
                "prompt_turns": len(prompt_slice),
                "completion_turns": len(completion_slice),
                "start_timestamp": window[0].timestamp,
                "end_timestamp": window[-1].timestamp,
                "source_file_start": window[0].source_file,
                "source_file_end": window[-1].source_file,
                "target_speaker": target_speaker,
            }

        examples.append(record)

    return examples


def split_counts(total: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    test_n = int(total * test_ratio)
    val_n = int(total * val_ratio)
    train_n = total - val_n - test_n

    if train_n < 1:
        raise ValueError("Split ratios leave no training data; reduce val/test ratios")

    return train_n, val_n, test_n


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or (args.val_ratio + args.test_ratio) >= 1:
        raise SystemExit("Invalid split ratios: require val_ratio >= 0, test_ratio >= 0, and val+test < 1")
    if args.chunk_turns < 2 or args.prompt_turns < 1 or args.stride < 1 or args.min_turns < 2:
        raise SystemExit("Invalid chunk settings: chunk_turns>=2, prompt_turns>=1, stride>=1, min_turns>=2")
    if args.prompt_turns >= args.chunk_turns:
        raise SystemExit("prompt_turns must be less than chunk_turns so assistant has completion turns")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    turns = load_turns(input_path)
    if not turns:
        raise SystemExit("No valid turns found in input file")

    examples = build_examples(
        turns=turns,
        chunk_turns=args.chunk_turns,
        prompt_turns=args.prompt_turns,
        stride=args.stride,
        min_turns=args.min_turns,
        system_prompt=args.system_prompt,
        include_metadata=args.include_metadata,
        speaker_control=args.speaker_control,
    )
    if not examples:
        raise SystemExit("No chunked examples generated; check chunk settings")

    train_n, val_n, test_n = split_counts(len(examples), args.val_ratio, args.test_ratio)

    # Chronological split by chunk order.
    train_rows = examples[:train_n]
    val_rows = examples[train_n : train_n + val_n]
    test_rows = examples[train_n + val_n :]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    wrote_train = write_jsonl(train_path, train_rows)
    wrote_val = write_jsonl(val_path, val_rows)
    wrote_test = write_jsonl(test_path, test_rows)

    stats = {
        "input_turns": len(turns),
        "examples": len(examples),
        "chunk_turns": args.chunk_turns,
        "prompt_turns": args.prompt_turns,
        "stride": args.stride,
        "min_turns": args.min_turns,
        "speaker_control": args.speaker_control,
        "split": {"train": wrote_train, "val": wrote_val, "test": wrote_test},
        "ratios": {"val_ratio": args.val_ratio, "test_ratio": args.test_ratio},
    }
    stats_path = output_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Loaded turns: {len(turns)}")
    print(f"Built chunked examples: {len(examples)}")
    print(f"Wrote train: {wrote_train} -> {train_path}")
    print(f"Wrote val:   {wrote_val} -> {val_path}")
    print(f"Wrote test:  {wrote_test} -> {test_path}")
    print(f"Stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
