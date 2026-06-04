# Agent guide for pi-tinker

This repo is a Pi package that helps people start fine-tuning open-source models with Tinker.

## Product promise

Help a non-expert go from:

```text
data → prepared JSONL → validation → baseline eval → smoke training → checkpoint → before/after comparison → chat in Pi
```

Do not turn this into a separate training framework. Keep Tinker/Tinker Cookbook as the real training layer and generate normal editable Python.

## Best first commands for users

If the user has no data yet:

```text
/tinker demo
/tinker next
/tinker doctor
```

If the user has CSV/JSON/JSONL/docs, prefer the managed operator first:

```text
/tinker improve <input> --goal "what should improve" --budget demo
```

Then, only after the user understands API usage and evals:

```text
/tinker improve <input> --goal "what should improve" --budget smoke --yes
/tinker improve <input> --goal "what should improve" --budget small --yes
/tinker deploy latest
```

Manual path:

```text
/tinker new <input> --goal "what should improve"
/tinker doctor
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
/tinker eval baseline --model Qwen/Qwen3.5-9B-Base --yes
/tinker smoke train_sft.py --yes
/tinker next
```

If the user only wants conversion:

```text
/tinker prepare data.csv --out data/train.jsonl
```

If the user is stuck:

```text
/tinker doctor
/skill:tinker-debug <paste error or report>
```

## What is possible

- Start from a zero-data demo: `/tinker demo`.
- Run a managed improve loop with `/tinker improve`.
- Generate app/client snippets with `/tinker deploy`.
- Convert CSV, JSON, JSONL prompt/response rows, TXT/MD files, or docs directories to chat JSONL.
- Scaffold editable Tinker Cookbook SFT scripts.
- Validate JSONL shape and, when dependencies are installed, renderer/token masks.
- Create and run an eval-first baseline/checkpoint comparison flow.
- Run a 2-step smoke test before spending real compute.
- Monitor logs and discover checkpoints.
- Register a Tinker sampler checkpoint as a Pi model for quick chat inspection.

## What is not solved here

- It cannot create high-quality training data from nothing.
- It is strongest for SFT; advanced RL/DPO is mostly guided by skills, not the extension wizard.
- It does not provide production serving; checkpoint registration is for inspection inside Pi.
- Real training requires `TINKER_API_KEY` and Tinker/Tinker Cookbook dependencies.

## Development checks

Before committing changes:

```bash
npm test
npm pack --dry-run
```

The local integration test intentionally avoids real Tinker API usage.
