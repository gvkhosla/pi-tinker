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
- a beginner step-by-step wizard (`/tinker start`, `/tinker next`, `/tinker reset`),
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
/tinker inkling
/tinker inkling sweep --prompt "representative task" --efforts low,medium,high,xhigh --yes
/tinker start data/train.jsonl --model thinkingmachines/Inkling
/tinker next
/tinker setup
/tinker init data/train.jsonl --model thinkingmachines/Inkling
/tinker validate data/train.jsonl --model thinkingmachines/Inkling
/tinker eval init
/tinker eval baseline --model thinkingmachines/Inkling --effort 0.9
/skill:tinker-research plan my eval before scaling
/tinker smoke train_sft.py
/tinker monitor logs/run
/tinker checkpoints logs/run
```

The user can always open and edit the generated Python.
