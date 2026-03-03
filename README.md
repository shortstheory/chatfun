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
