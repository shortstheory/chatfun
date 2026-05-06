# ChatFun Data Prep

This README captures the exact commands used to prepare training data from WhatsApp exports in this repo.

## 1. Parse raw WhatsApp text exports

```bash
python3 scripts/parse_whatsapp.py \
  --in data \
  --out processed/turns.jsonl \
  --drop-system \
  --drop-media-placeholders \
  --strip-edit-marker
```

Optional minimal output (speaker + text only):

```bash
python3 scripts/parse_whatsapp.py \
  --in data \
  --out processed/turns.min.jsonl \
  --drop-system \
  --drop-media-placeholders \
  --strip-edit-marker \
  --minimal
```

## 2. Build chunked train/val/test datasets (speaker control)

```bash
python3 scripts/build_dataset.py \
  --in processed/turns.jsonl \
  --out datasets \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --chunk-turns 32 \
  --prompt-turns 16 \
  --stride 8 \
  --speaker-control \
  --include-metadata
```

## 3. Verify generated files

```bash
wc -l processed/turns.jsonl
wc -l datasets/train.jsonl datasets/val.jsonl datasets/test.jsonl
cat datasets/split_stats.json
```

Current expected outputs:
- `processed/turns.jsonl`
- `datasets/train.jsonl`
- `datasets/val.jsonl`
- `datasets/test.jsonl`
- `datasets/split_stats.json`

## 4. Train QLoRA with Unsloth (uv)

Install deps:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements-train.txt
```

Run training:

```bash
uv run python scripts/train_lora_unsloth.py \
  --train datasets/train.jsonl \
  --val datasets/val.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir models/qwen25-7b-groupchat-lora \
  --max-seq-length 4096 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8
```

## 5. Train with W&B logging

One-time login:

```bash
wandb login
```

Training with logging:

```bash
uv run python scripts/train_lora_unsloth.py \
  --train datasets/train.jsonl \
  --val datasets/val.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir models/qwen25-7b-groupchat-lora \
  --max-seq-length 4096 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --wandb \
  --wandb-project chatfun-lora \
  --wandb-run-name qwen25-7b-v1
```

## 6. Evaluate a trained LoRA on the held-out test set

Run evaluation:

```bash
python3 scripts/eval_lora.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter models/qwen25-7b-groupchat-lora/adapter \
  --test datasets/test.jsonl \
  --out processed/eval_predictions.jsonl \
  --load-in-4bit
```

Notes:
- The script generates from each example's `system` + `user` messages and compares the output to the reference `assistant` content.
- It reports exact match, normalized exact match, and token-level F1.
- Add `--limit 20` for a quick smoke test before running the full split.

Evaluate every checkpoint in one training run folder:

```bash
python3 scripts/eval_lora.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-dir models/qwen25-7b-groupchat-lora \
  --test datasets/test.jsonl \
  --out processed/eval_predictions.jsonl \
  --load-in-4bit
```

This evaluates each `checkpoint-*` directory inside the run folder, plus `adapter/` if present, and prints a ranked summary table.

## 7. Run a local Telegram bot from your desktop

Install bot deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-bot.txt
```

Create a Telegram bot with `@BotFather`, then export the token:

```bash
export TELEGRAM_BOT_TOKEN=your_bot_token_here
```

Run the bot with your trained adapter:

```bash
python3 scripts/telegram_bot.py \
  --adapter models/qwen25-7b-groupchat-lora/adapter \
  --model Qwen/Qwen2.5-7B-Instruct \
  --load-in-4bit
```

Notes:
- This uses Telegram long polling, so it can run entirely from your desktop with no public webhook.
- Local chat history is stored in `processed/telegram_bot.sqlite3`.
- Use `/reset` inside Telegram to clear the saved history for a chat.
- Add `--mode group` if you want transcript-style multi-speaker group chat simulation instead of a single-person reply.
