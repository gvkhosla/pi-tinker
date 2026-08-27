# Design notes

`pi-tinker` is deliberately thin.

## Why thin?

Tinker already provides the hard abstraction: users write local Python loops and Tinker handles distributed training. Tinker Cookbook already provides higher-level training loops, renderers, datasets, logging, checkpointing, evaluation, and weight export.

Adding another framework inside Pi would make the user experience worse. Pi should help users operate the existing stack, not replace it.

## Responsibilities

`pi-tinker` handles:

- direct Inkling registration through Tinker's Anthropic-compatible endpoint,
- effort mapping and representative-task sweeps for Inkling,
- discovery: thin Pi skills that point at official Cookbook research/debug/inkling skills,
- one beginner front door (`/tinker improve`) plus one state-aware next command,
- setup checks,
- JSONL plus Python-backed renderer/token-mask validation,
- guided project initialization,
- eval-first baseline/checkpoint comparison,
- scaffolding editable Tinker Cookbook scripts,
- smoke-test execution,
- live monitoring/log summarization,
- checkpoint discovery and registration for interactive inspection in Pi,
- `/tinker deploy` artifacts: Tinker API clients, Cookbook weight-export scripts, and a serving decision (not a serving runtime).

Tinker/Tinker Cookbook handle:

- training APIs,
- renderers and tokenization,
- real dataset building,
- SFT/RL/DPO/distillation loops,
- checkpointing,
- evals,
- weight export.

## Non-goals

- No hidden training service.
- No custom TypeScript Tinker SDK wrapper.
- No replacement training config format.
- No new eval or logging format.
- No production inference abstraction. Generate Cookbook export/Modal commands and an HTDYM pointer; do not wrap vLLM, SGLang, Modal, or HTDYM as a Pi runtime.

## Ideal user flow

```text
/tinker improve data.csv --goal "what should improve" --budget demo
# Review data/train.jsonl, data/eval.jsonl, train_sft.py, and eval.py.
/tinker next
/tinker improve data.csv --goal "what should improve" --budget smoke --eval-reviewed --yes
/tinker next
/tinker improve data.csv --goal "what should improve" --budget small --yes
/tinker deploy latest
```

Managed runs hash their source/train/eval/code/model/effort provenance. A checkpoint is only approved when it beats the matching baseline; `deploy latest` never resolves a rejected candidate. Inkling training and both evals use the same persisted effort.

The user can always open and edit the generated Python.
