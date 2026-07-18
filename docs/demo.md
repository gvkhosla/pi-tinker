# 60-second demo

Use this demo to see what `pi-tinker` creates. It does **not** call the Tinker API or start training.

## 1. Install

```bash
pi install git:github.com/gvkhosla/pi-tinker
pi
```

## 2. Create the example project

Inside Pi:

```text
/tinker demo
```

This creates a small customer-support dataset and these editable files:

```text
data/train.jsonl
train_sft.py
eval.py
eval_checkpoint.py
tinker.yaml
notes/plan.md
```

## 3. Check your progress

```text
/tinker next
/tinker doctor
```

- `/tinker next` tells you what to do next.
- `/tinker doctor` checks Python, packages, your API key, and generated files.

You can stop here without spending anything.

## Try your own data

A simple CSV is enough:

```csv
question,answer
How do I cancel?,Go to Settings → Billing → Cancel subscription.
My order is late,Send us your order number and we will check it.
```

Then run:

```text
/tinker improve data.csv --goal "better support answers" --budget demo
```

The `demo` budget still makes no API calls.

## Continue to a real smoke run

First install Inkling's Python dependencies:

```bash
uv pip install -U 'tinker-cookbook[inkling]'
export TINKER_API_KEY="your-api-key"
```

Then validate and run only two training steps:

```text
/tinker validate data/train.jsonl
/tinker improve data.csv --goal "better support answers" --budget smoke --yes
```

The `smoke` budget uses the Tinker API. Review the command before adding `--yes`.
