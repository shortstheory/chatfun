#!/usr/bin/env python3
"""Run a local Telegram bot backed by a base model + LoRA adapter."""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import threading
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel

try:
    from telegram import Update
    from telegram.constants import ChatAction
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
except ImportError as exc:  # pragma: no cover - import guard for friendlier setup errors
    raise SystemExit(
        "python-telegram-bot is required. Install bot deps with: pip install -r requirements-bot.txt"
    ) from exc


DEFAULT_SINGLE_SYSTEM_PROMPT = (
    "You are replying inside Telegram as a single friendly person. "
    "Reply with only one short natural message. "
    "Do not invent other speakers, do not write speaker tags, and do not write a transcript."
)
DEFAULT_GROUP_SYSTEM_PROMPT = (
    "You are simulating a friends group chat. Continue naturally with realistic speaker voices, "
    "casual tone, and speaker-tagged lines in the format 'Name: message'. "
    "You may generate multiple lines from different speakers."
)
DEFAULT_HISTORY_LIMIT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Telegram bot using a LoRA adapter")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument(
        "--telegram-token",
        default=os.environ.get("TELEGRAM_BOT_TOKEN"),
        help="Telegram bot token, or set TELEGRAM_BOT_TOKEN",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max context length")
    parser.add_argument("--min-new-tokens", type=int, default=0, help="Minimum generated tokens")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT, help="Saved turns per chat")
    parser.add_argument(
        "--db-path",
        default="processed/telegram_bot.sqlite3",
        help="SQLite path for local chat history",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional custom system prompt override",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "group"],
        default="single",
        help="Choose between a one-person Telegram reply and transcript-style group simulation",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Force 4-bit loading")
    parser.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit loading")
    return parser.parse_args()


def build_messages(history: list[tuple[str, str]], system_prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in history:
        if content.strip():
            messages.append({"role": role, "content": content})
    return messages


def sanitize_reply(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            prefix, rest = line.split(":", 1)
            if prefix and " " not in prefix and len(prefix) <= 32:
                line = rest.strip()
        cleaned_lines.append(line)
        if cleaned_lines:
            break

    reply = " ".join(cleaned_lines).strip()
    return reply or "lol"


def sanitize_group_reply(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned_lines.append(line)

    reply = "\n".join(cleaned_lines).strip()
    return reply or "Arnav: lol"


def split_group_messages(reply: str) -> list[str]:
    messages: list[str] = []
    for raw_line in reply.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        messages.append(line)
    return messages or [reply]


def sanitize_group_line(text: str) -> str:
    return text.strip()


class HistoryStore:
    """Tiny SQLite store for per-chat message history."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def get_history(self, chat_id: str, limit: int) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()
        ordered_rows = reversed(rows)
        return [(row["role"], row["content"]) for row in ordered_rows]

    def append(self, chat_id: str, role: str, content: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content),
            )
            connection.commit()

    def clear(self, chat_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            connection.commit()


class LocalLoraBot:
    """Loads the local model once and serves synchronous generations."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str,
        max_seq_length: int,
        min_new_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        load_in_4bit: bool,
        system_prompt: str,
        mode: str,
    ) -> None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=False,
            full_finetuning=False,
        )
        self.model = PeftModel.from_pretrained(model, adapter_path)
        FastLanguageModel.for_inference(self.model)

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else str(self.model.device)
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.mode = mode
        self._lock = asyncio.Lock()

    def _generate_sync(self, history: list[tuple[str, str]]) -> str:
        messages = build_messages(history, self.system_prompt)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=self.min_new_tokens,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if self.mode == "group":
            return sanitize_group_reply(decoded)
        return sanitize_reply(decoded)

    def _stream_group_lines_sync(self, history: list[tuple[str, str]], on_line) -> None:
        messages = build_messages(history, self.system_prompt)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        def run_generation() -> None:
            with torch.no_grad():
                self.model.generate(**generation_kwargs)

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()

        pending = ""
        for chunk in streamer:
            pending += chunk
            while "\n" in pending:
                raw_line, pending = pending.split("\n", 1)
                line = sanitize_group_line(raw_line)
                if line:
                    on_line(line)

        worker.join()

        tail = sanitize_group_line(pending)
        if tail:
            on_line(tail)
            return
        on_line("Arnav: lol")

    async def generate(self, history: list[tuple[str, str]]) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, history)

    async def stream_group_lines(self, history: list[tuple[str, str]]):
        async with self._lock:
            queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def worker() -> None:
                try:
                    self._stream_group_lines_sync(
                        history,
                        lambda line: loop.call_soon_threadsafe(queue.put_nowait, line),
                    )
                except Exception as exc:  # pragma: no cover - runtime inference errors
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=worker, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "I am running locally from this desktop. Send a message and I will reply using the LoRA model. "
        "Use /reset to clear our chat history."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    store: HistoryStore = context.application.bot_data["history_store"]
    chat_id = str(update.effective_chat.id)
    store.clear(chat_id)
    await update.message.reply_text("Cleared local chat history for this chat.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Commands:\n"
        "/start - intro\n"
        "/reset - clear saved conversation\n"
        "/help - show this message"
    )


def format_user_message(message_text: str, user_name: str | None) -> str:
    name = (user_name or "Telegram user").strip()
    return f"{name}: {message_text.strip()}"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return

    store: HistoryStore = context.application.bot_data["history_store"]
    lora_bot: LocalLoraBot = context.application.bot_data["lora_bot"]
    history_limit: int = context.application.bot_data["history_limit"]
    mode: str = context.application.bot_data["mode"]

    chat_id = str(update.effective_chat.id)
    sender = update.effective_user.first_name if update.effective_user else None
    user_text = update.message.text.strip()
    if not user_text:
        return

    print(f"[telegram_bot] received chat_id={chat_id} sender={sender or 'unknown'} text={user_text!r}", flush=True)
    store.append(chat_id, "user", format_user_message(user_text, sender))
    history = store.get_history(chat_id, history_limit)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    if mode == "group":
        print(f"[telegram_bot] streaming group generation chat_id={chat_id} history_turns={len(history)}", flush=True)
        started_at = time.perf_counter()
        sent_count = 0
        async for response_message in lora_bot.stream_group_lines(history):
            store.append(chat_id, "assistant", response_message)
            await update.message.reply_text(response_message)
            sent_count += 1
            print(
                f"[telegram_bot] streamed line chat_id={chat_id} line_index={sent_count} text={response_message!r}",
                flush=True,
            )
        duration_s = time.perf_counter() - started_at
        print(
            f"[telegram_bot] replied chat_id={chat_id} messages={sent_count} duration_s={duration_s:.2f}",
            flush=True,
        )
        return

    print(f"[telegram_bot] generating chat_id={chat_id} history_turns={len(history)}", flush=True)
    started_at = time.perf_counter()
    reply = await lora_bot.generate(history)
    duration_s = time.perf_counter() - started_at
    print(f"[telegram_bot] generated chat_id={chat_id} duration_s={duration_s:.2f} reply={reply!r}", flush=True)

    store.append(chat_id, "assistant", reply)
    await update.message.reply_text(reply)
    print(f"[telegram_bot] replied chat_id={chat_id} messages=1", flush=True)


def build_application(token: str, store: HistoryStore, lora_bot: LocalLoraBot, history_limit: int, mode: str) -> Application:
    application = Application.builder().token(token).build()
    application.bot_data["history_store"] = store
    application.bot_data["lora_bot"] = lora_bot
    application.bot_data["history_limit"] = history_limit
    application.bot_data["mode"] = mode
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return application


def validate_args(args: argparse.Namespace) -> None:
    if not args.telegram_token:
        raise SystemExit("Missing Telegram token. Pass --telegram-token or set TELEGRAM_BOT_TOKEN.")
    if args.load_in_4bit and args.no_load_in_4bit:
        raise SystemExit("Pass only one of --load-in-4bit or --no-load-in-4bit")


def resolve_system_prompt(mode: str, custom_prompt: str | None) -> str:
    if custom_prompt:
        return custom_prompt
    if mode == "group":
        return DEFAULT_GROUP_SYSTEM_PROMPT
    return DEFAULT_SINGLE_SYSTEM_PROMPT


def main() -> int:
    args = parse_args()
    validate_args(args)

    load_in_4bit = not args.no_load_in_4bit
    if args.load_in_4bit:
        load_in_4bit = True

    system_prompt = resolve_system_prompt(args.mode, args.system_prompt)
    store = HistoryStore(Path(args.db_path))
    lora_bot = LocalLoraBot(
        model_name=args.model,
        adapter_path=args.adapter,
        max_seq_length=args.max_seq_length,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=load_in_4bit,
        system_prompt=system_prompt,
        mode=args.mode,
    )
    application = build_application(args.telegram_token, store, lora_bot, args.history_limit, args.mode)

    print(f"[telegram_bot] starting long-polling mode={args.mode}", flush=True)
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
