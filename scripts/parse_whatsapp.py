#!/usr/bin/env python3
"""Parse WhatsApp text exports into JSONL chat turns.

Supported quirks seen in this repository:
- Optional invisible Unicode marker before timestamp (e.g. LRM `\u200e`)
- Multiline messages
- System/event lines and media placeholders
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Optional invisible marker appears before some exported rows.
LEADING_MARKERS = "\u200e\u200f\ufeff"

# [12/7/23, 09:15:50] Name: message
HEADER_RE = re.compile(
    rf"^[{LEADING_MARKERS}]?"
    r"\[(?P<date>\d{1,2}/\d{1,2}/\d{2}), (?P<time>\d{1,2}:\d{2}:\d{2})\] "
    r"(?P<speaker>[^:]+): (?P<text>.*)$"
)

# These are common non-conversational event messages from WhatsApp exports.
SYSTEM_TEXT_PATTERNS = [
    re.compile(r"\b(created group|added|removed|left|joined using this group's invite link)\b", re.IGNORECASE),
    re.compile(r"\b(changed (this group's icon|the group name|group settings|group description))\b", re.IGNORECASE),
    re.compile(r"\b(messages and calls are end-to-end encrypted)\b", re.IGNORECASE),
    re.compile(r"\b(you're now an admin|only admins can)\b", re.IGNORECASE),
]

MEDIA_PLACEHOLDER_PATTERNS = [
    re.compile(r"\b(image omitted|video omitted|audio omitted|sticker omitted|document omitted|GIF omitted)\b", re.IGNORECASE),
    re.compile(r"\bThis message was deleted\.?\b", re.IGNORECASE),
]

EDIT_MARKER_RE = re.compile(r"\s*<This message was edited>\s*", re.IGNORECASE)


@dataclass
class ParsedMessage:
    timestamp: str
    speaker: str
    text: str
    source_file: str
    line_start: int


def parse_timestamp(date_text: str, time_text: str) -> str:
    dt = datetime.strptime(f"{date_text} {time_text}", "%m/%d/%y %H:%M:%S")
    return dt.isoformat()


def is_system_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in SYSTEM_TEXT_PATTERNS)


def is_media_placeholder(text: str) -> bool:
    return any(pattern.search(text) for pattern in MEDIA_PLACEHOLDER_PATTERNS)


def iter_messages(path: Path) -> Iterable[ParsedMessage]:
    current: dict[str, object] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            match = HEADER_RE.match(line)
            if match:
                if current is not None:
                    yield ParsedMessage(**current)

                timestamp = parse_timestamp(match.group("date"), match.group("time"))
                current = {
                    "timestamp": timestamp,
                    "speaker": match.group("speaker").strip(),
                    "text": match.group("text"),
                    "source_file": str(path),
                    "line_start": line_number,
                }
            elif current is not None:
                # Preserve multiline content from exported messages.
                text = str(current["text"]) + "\n" + line
                current["text"] = text

    if current is not None:
        yield ParsedMessage(**current)


def collect_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.txt") if p.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse WhatsApp .txt exports into JSONL turns")
    parser.add_argument("--in", dest="input_path", required=True, help="Input .txt file or directory")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL path")
    parser.add_argument(
        "--exclude-speakers",
        nargs="*",
        default=["Meta AI"],
        help="Speaker names to exclude (default: Meta AI)",
    )
    parser.add_argument(
        "--drop-system",
        action="store_true",
        help="Drop system/event text rows (recommended)",
    )
    parser.add_argument(
        "--drop-media-placeholders",
        action="store_true",
        help="Drop lines like 'image omitted' or 'This message was deleted.'",
    )
    parser.add_argument(
        "--strip-edit-marker",
        action="store_true",
        help="Remove '<This message was edited>' marker from text",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Write only `speaker` and `text` fields (recommended for training-ready output)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(input_path)
    if not files:
        raise SystemExit(f"No .txt files found under: {input_path}")

    excluded_speakers = set(args.exclude_speakers)

    total_raw = 0
    kept = 0
    dropped_speaker = 0
    dropped_system = 0
    dropped_media = 0

    with output_path.open("w", encoding="utf-8") as out:
        for file_path in files:
            for msg in iter_messages(file_path):
                total_raw += 1

                if msg.speaker in excluded_speakers:
                    dropped_speaker += 1
                    continue

                text = msg.text.strip()
                if args.strip_edit_marker:
                    text = EDIT_MARKER_RE.sub("", text).strip()

                if args.drop_system and is_system_text(text):
                    dropped_system += 1
                    continue

                if args.drop_media_placeholders and is_media_placeholder(text):
                    dropped_media += 1
                    continue

                if args.minimal:
                    record = {
                        "speaker": msg.speaker,
                        "text": text,
                    }
                else:
                    record = {
                        "timestamp": msg.timestamp,
                        "speaker": msg.speaker,
                        "text": text,
                        "source_file": Path(msg.source_file).name,
                        "line_start": msg.line_start,
                    }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Input files: {len(files)}")
    print(f"Raw parsed messages: {total_raw}")
    print(f"Dropped by speaker: {dropped_speaker}")
    print(f"Dropped system/event: {dropped_system}")
    print(f"Dropped media/deleted placeholders: {dropped_media}")
    print(f"Written records: {kept}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
