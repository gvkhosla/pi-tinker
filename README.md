# pi-tinker

Fine-tune an open-source model with [Tinker](https://thinkingmachines.ai/tinker/) from inside [Pi](https://pi.dev) without learning the whole post-training stack first.

`pi-tinker` is intentionally **not** another training framework. It is a beginner-friendly operator around Tinker/Tinker Cookbook: bring CSV/JSON/JSONL/docs, convert and validate data, create editable Python, run baseline evals, smoke-test training, monitor checkpoints, compare before/after quality, generate app snippets, and chat with the trained checkpoint in Pi.

## Who this is for

Builders with examples who want to fine-tune an open model but do **not** yet know the Tinker/post-training workflow. If you have support tickets, prompt/completion pairs, extraction examples, writing examples, or task-specific eval cases, this helps you get to a useful first run faster.

## 30-second demo, no data required

```bash
pi install git:github.com/gvkhosla/pi-tinker
```

```text
/tinker demo
/tinker next
/tinker doctor
```

`/tinker demo` creates a small customer-support fine-tuning project so you can see the whole flow before bringing your own data. See [`docs/demo.md`](docs/demo.md) for the full terminal demo transcript.

## 10-minute path with your own data

Create a tiny CSV:

```csv
question,answer
How do I cancel?,Go to Settings → Billing → Cancel subscription.
My order is late,Sorry — send us your order number and we’ll check it.
The app crashes after login,Update the app and restart. If it still crashes, send your device type and app version.
```

Then run the managed operator:

```text
/tinker improve data.csv --goal "better customer support answers" --budget demo
```

When the demo/no-API setup looks good, move through safer budgets:

```text
/tinker improve data.csv --goal "better customer support answers" --budget smoke --yes
/tinker improve data.csv --goal "better customer support answers" --budget small --yes
/tinker deploy latest
```

Or drive the steps manually:

```text
/tinker new data.csv --goal "better customer support answers"
/tinker doctor
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
/tinker eval baseline --model Qwen/Qwen3.5-9B-Base --yes
/tinker smoke train_sft.py --yes
/tinker next
```

The extension keeps showing the next safe step: validate → baseline eval → 2-step smoke test → train → compare → deploy/use checkpoint.

If you try it and get stuck, please open an issue with the command you ran and the `/tinker doctor` output.

## For coding agents

If you are using an agent to drive this repo, see [`AGENTS.md`](AGENTS.md). The intended first command is usually:

```text
/tinker demo
```

or, with user data:

```text
/tinker new <csv|json|jsonl|docs-dir> --goal "what should improve"
```

You can also ask Pi:

```text
Use pi-tinker to help me fine-tune a model on this CSV. Start with /tinker doctor, validate the data, run a baseline eval and 2-step smoke test, and do not scale training until eval is defined.
```

## What it does right now

### 1. Adds Tinker skills to Pi

- `/skill:tinker-research` — plan and run Tinker experiments: SFT, RL, DPO, distillation, evaluation, hyperparameters, checkpoints, weight export, and experiment notes.
- `/skill:tinker-debug` — diagnose slow training, hangs, renderer/tokenization issues, checkpoint/export mismatches, opaque errors, and service-vs-user-code problems.

### 2. Adds a `/tinker` command

```text
/tinker improve data.csv --goal "better support answers" --budget demo
/tinker deploy latest
/tinker demo
/tinker new data.csv --goal "better support answers"
/tinker new --example customer-support
/tinker prepare data.csv --out data/train.jsonl
/tinker recommend --goal "structured extraction"
/tinker doctor
/tinker examples list
/tinker start data/train.jsonl
/tinker next
/tinker reset
/tinker setup
/tinker init data/train.jsonl
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
/tinker eval init
/tinker eval baseline --model Qwen/Qwen3.5-9B-Base
/tinker sft data/train.jsonl --model Qwen/Qwen3.5-9B-Base --steps 20
/tinker smoke train_sft.py
/tinker monitor logs/my-run
/tinker checkpoints logs/my-run
/tinker status [log_dir]
/tinker use tinker://.../sampler_weights/... [alias]
/tinker use --list
/tinker use --remove <alias-or-tinker-path>
```

### 3. Adds a managed improvement operator

`/tinker improve` is the highest-level workflow. It can prepare data, scaffold files, run doctor/validation, ensure evals exist, run baseline evals, run 2-step smoke tests, scale to a short/real run with confirmation, evaluate checkpoints, compare wins/regressions, register the checkpoint in Pi, and suggest what data to add next.

Budgets:

- `demo` — no API usage; setup + doctor + lightweight validation.
- `smoke` — baseline eval + 2-step smoke train, then stop.
- `small` — short training run after smoke.
- `real` — larger run, still confirmation-first.

### 4. Adds app/deploy snippets

`/tinker deploy <checkpoint-or-alias>` generates a deploy folder with `.env.example`, Python client, Node client, FastAPI wrapper, and README for using a Tinker sampler checkpoint through the OpenAI-compatible endpoint.

### 5. Adds a real golden path for beginners

`/tinker new` starts from an existing chat JSONL file, converts CSV/JSON/docs into chat JSONL, or copies a built-in example. It scaffolds the project, recommends starter settings, creates wizard state, and shows the next command.

`/tinker start` remains the step-by-step wizard for people who already have training JSONL. It tracks progress in `.tinker-pi/state.json` and always shows the next recommended action: environment, data, validation, baseline eval, smoke test, training, checkpoint eval, comparison, and chatting with the checkpoint.

### 6. Adds data preparation, recommendations, doctor, and examples

- `/tinker prepare` converts CSV, JSON, JSONL prompt/response rows, TXT/MD files, or docs directories into Tinker chat JSONL.
- `/tinker recommend` picks a starter model/method/settings from the user's goal.
- `/tinker doctor` checks API key, Python, Tinker imports, generated scripts, selected data, and next step.
- `/tinker examples` provides concrete starter tasks: customer support, structured extraction, and concise writing.

### 7. Adds eval-first before training

`/tinker eval init` creates an editable `eval.py` and `data/eval.jsonl`. Users can run a baseline eval before training, evaluate a checkpoint after training, and compare wins/regressions.

### 8. Scaffolds editable SFT projects

`/tinker init ...` or `/tinker sft ...` writes:

```text
train_sft.py          # editable tinker-cookbook SFT script
eval_checkpoint.py    # quick sampling smoke test for a checkpoint
tinker.yaml           # human-readable run summary
notes/plan.md         # experiment plan template
```

### 9. Registers trained checkpoints as Pi models

`/tinker use tinker://.../sampler_weights/... my-model` registers a checkpoint through Tinker's beta OpenAI-compatible inference endpoint. You can then select it with `/model` and quickly poke at the fine-tuned model inside Pi.

> Tinker's OpenAI-compatible endpoint is currently best for checkpoint inspection and internal testing, not production serving.

## Install

### From a local checkout

```bash
pi install /path/to/pi-tinker
```

For one run only:

```bash
pi -e /path/to/pi-tinker
```

### From GitHub

```bash
pi install git:github.com/gvkhosla/pi-tinker
```

## Requirements

- Pi installed.
- Python 3.11+ recommended for `tinker-cookbook`.
- Tinker API key:

```bash
export TINKER_API_KEY="your-api-key"
```

- For training scripts generated by this package:

```bash
uv pip install tinker-cookbook
# or
python3 -m pip install tinker-cookbook
```

## Quickstart

Fastest path with a CSV/JSONL/docs input:

```text
/tinker new data.csv --goal "better customer support answers"
/tinker doctor
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
/tinker eval baseline --model Qwen/Qwen3.5-9B-Base --yes
/tinker smoke train_sft.py --yes
/tinker monitor logs/<run-dir>
```

No data yet? Try a concrete example:

```text
/tinker demo
/tinker next
```

Then run the generated smoke test:

```bash
python train_sft.py max_steps=2
```

If the smoke test looks good, scale the run:

```bash
python train_sft.py max_steps=100
```

Monitor from Pi:

```text
/tinker status logs/<run-dir>
```

Register a sampler checkpoint:

```text
/tinker use tinker://.../sampler_weights/... my-sft
/model
```

## Data format for `/tinker sft`

JSONL, one conversation per line:

```json
{"messages":[{"role":"user","content":"How do I reset my password?"},{"role":"assistant","content":"Go to Settings → Security → Reset password."}]}
```

`/tinker validate` runs basic JSONL/message-shape checks plus a Python-backed Tinker Cookbook renderer/token-mask audit when dependencies are installed. It returns a clear `READY`, `SMOKE ONLY`, or `FIX DATA FIRST` recommendation with token histograms, truncation risk, zero-trainable rows, longest examples, decoded previews, and trainable-token snippets.

## Command reference

See [`docs/commands.md`](docs/commands.md).

## Design principles

- Keep Tinker and Tinker Cookbook as the real training layer.
- Generate normal, editable Python instead of hidden framework state.
- Encourage baseline evals and data inspection before spending compute.
- Make common workflows discoverable in Pi.
- Make trained checkpoints easy to inspect inside Pi.

## Package structure

```text
extensions/tinker.ts       # Pi extension with /tinker command and provider registration
skills/tinker-research/    # Pi skill adapted from tinker-cookbook research skill
skills/tinker-debug/       # Pi skill adapted from tinker-cookbook debug skill
docs/                      # user/developer docs
examples/customer-support.csv # copy-paste starter CSV
AGENTS.md                  # quickstart and boundaries for coding agents
scripts/smoke-test.mjs     # lightweight repository checks
```

## Development

```bash
npm test
pi -e . --list-models
```

## License and attribution

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

This package includes Pi-specific adaptations of skill content from Thinking Machines Lab's `tinker-cookbook` repository.
