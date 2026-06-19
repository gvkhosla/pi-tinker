# What pi-tinker outputs

This page shows what users and agents should expect before they spend real Tinker API usage.

## No-data demo

```text
/tinker demo
```

Creates a small customer-support project and wizard state:

```text
examples/customer-support/train.jsonl
data/eval.jsonl
eval.py
eval_checkpoint.py
train_sft.py
tinker.yaml
notes/plan.md
.tinker-pi/state.json
```

Then:

```text
/tinker next
```

shows the progress checklist:

```text
Environment ready
Training data selected
Project files created
Data validated
Baseline eval run
2-step smoke test run
Training/checkpoint available
Checkpoint eval run
Before/after compared
Checkpoint registered for chat
```

## Managed no-API setup

```text
/tinker improve data.csv --goal "better support answers" --budget demo
```

Runs the safe setup stages only:

- converts data to chat JSONL when needed,
- writes editable training/eval files,
- runs doctor checks,
- runs lightweight dataset validation,
- stops before API usage.

Use this before `smoke`, `small`, or `real` budgets.

## API-using budgets

After setting `TINKER_API_KEY`, move up gradually:

```text
/tinker improve data.csv --goal "better support answers" --budget smoke --yes
/tinker improve data.csv --goal "better support answers" --budget small --yes
```

The final report should include:

- baseline eval score,
- smoke train status,
- training metrics,
- sampler checkpoint path,
- checkpoint eval score,
- before/after delta,
- wins/regressions,
- what data to add next,
- registered checkpoint alias.

## Deploy snippets

After a checkpoint exists:

```text
/tinker deploy latest
```

Writes:

```text
deploy/<alias>/README.md
deploy/<alias>/.env.example
deploy/<alias>/python_client.py
deploy/<alias>/node_client.mjs
deploy/<alias>/fastapi_app.py
```

These snippets are for quick inspection/internal app testing through Tinker's OpenAI-compatible endpoint, not a full production serving stack.
