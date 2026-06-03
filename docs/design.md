# Design notes

`pi-tinker` is deliberately thin.

## Why thin?

Tinker already provides the hard abstraction: users write local Python loops and Tinker handles distributed training. Tinker Cookbook already provides higher-level training loops, renderers, datasets, logging, checkpointing, evaluation, and weight export.

Adding another framework inside Pi would make the user experience worse. Pi should help users operate the existing stack, not replace it.

## Responsibilities

`pi-tinker` handles:

- discovery: skills that tell Pi when and how to use Tinker,
- setup checks,
- JSONL plus Python-backed renderer/token-mask validation,
- guided project initialization,
- eval-first baseline/checkpoint comparison,
- scaffolding editable Tinker Cookbook scripts,
- smoke-test execution,
- live monitoring/log summarization,
- checkpoint discovery and registration for interactive inspection in Pi.

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
- No production inference abstraction.

## Ideal user flow

```text
/tinker setup
/tinker init data/train.jsonl
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
/tinker eval init
/tinker eval baseline --model Qwen/Qwen3.5-9B-Base
/skill:tinker-research plan my eval before scaling
/tinker smoke train_sft.py
/tinker monitor logs/run
/tinker checkpoints logs/run
```

The user can always open and edit the generated Python.
