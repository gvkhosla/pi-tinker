# 60-second pi-tinker demo

This is the shortest path to see what the package does without bringing your own data.

```bash
pi install git:github.com/gvkhosla/pi-tinker
```

Then in Pi:

```text
/tinker demo
/tinker next
/tinker doctor
```

Expected result:

- A tiny customer-support training dataset is copied into `examples/customer-support/train.jsonl`.
- Editable project files are generated: `train_sft.py`, `eval.py`, `eval_checkpoint.py`, `tinker.yaml`, and `notes/plan.md`.
- `.tinker-pi/state.json` tracks your progress.
- `/tinker next` shows the next safe step.
- `/tinker doctor` tells you what is missing before real Tinker API usage.

With your own data, use:

```text
/tinker new data.csv --goal "better customer support answers"
```

CSV shape:

```csv
question,answer
How do I cancel?,Go to Settings → Billing → Cancel subscription.
My order is late,Sorry — send us your order number and we’ll check it.
```

Real training requires `TINKER_API_KEY` and `tinker-cookbook` installed. The recommended flow is always:

```text
/tinker doctor
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
/tinker eval baseline --model Qwen/Qwen3.5-9B-Base --yes
/tinker smoke train_sft.py --yes
```
