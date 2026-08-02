# pi-tinker

Use and fine-tune [Inkling](https://thinkingmachines.ai/inkling/) with [Tinker](https://thinkingmachines.ai/tinker/) inside [Pi](https://pi.dev).

`pi-tinker` helps you turn examples into a trained model without hiding the real work. It prepares data, writes editable Python, checks your setup, runs evals, starts small training runs, and lets you chat with checkpoints in Pi.

> **New here?** Install the package and run `/tinker demo`. The demo does not use the Tinker API.

## Pick what you want to do

### 1. Chat with Inkling in Pi

Install the package:

```bash
pi install git:github.com/gvkhosla/pi-tinker
```

Set your Tinker API key and open Pi:

```bash
export TINKER_API_KEY="your-api-key"
pi
```

Then:

```text
/tinker inkling
/model
```

Choose Inkling or Inkling-Small, in either the 64K or 256K context variant, from `/model`. Both support tools, images, streamed thinking, and adjustable reasoning effort.

### 2. Try the no-cost demo

Inside Pi:

```text
/tinker demo
/tinker next
/tinker doctor
```

This creates a tiny customer-support project with editable training data, Python scripts, and an eval. It stops before any API usage.

### 3. Fine-tune Inkling on your data

Start with CSV, JSON, JSONL, text files, Markdown, or a docs directory:

```text
/tinker improve data.csv --goal "better customer support answers" --budget demo
```

The `demo` budget prepares everything without calling the API. Review the generated files, then run a small smoke test:

```text
/tinker improve data.csv --goal "better customer support answers" --budget smoke --yes
```

If the smoke run and eval look good:

```text
/tinker improve data.csv --goal "better customer support answers" --budget small --yes
```

`pi-tinker` guides you through:

```text
data → validation → baseline eval → 2-step smoke run → training → checkpoint eval → chat in Pi
```

## Setup for fine-tuning

Chatting with Inkling only needs Pi and `TINKER_API_KEY`. Fine-tuning also needs Python 3.11+ and Inkling's renderer stack:

```bash
uv pip install -U 'tinker-cookbook[inkling]'
```

This installs compatible versions of the Tinker SDK, Tinker Cookbook, and `tml-renderers`.

Check everything with:

```text
/tinker doctor
```

## What files does it create?

The important output is normal code and data you can inspect and edit:

```text
data/train.jsonl      training conversations
data/eval.jsonl       held-out evaluation examples
train_sft.py          Tinker Cookbook training script
eval.py               baseline/checkpoint eval script
tinker.yaml           readable run settings
notes/plan.md         experiment notes
```

Tinker and Tinker Cookbook remain the training layer. `pi-tinker` is an operator around them, not a separate framework.

## A tiny data example

CSV:

```csv
question,answer
How do I cancel?,Go to Settings → Billing → Cancel subscription.
My order is late,Send us your order number and we will check it.
```

Or chat JSONL, one conversation per line:

```json
{"messages":[{"role":"user","content":"How do I reset my password?"},{"role":"assistant","content":"Go to Settings → Security → Reset password."}]}
```

## Reasoning effort, simply explained

Inkling can spend different amounts of compute thinking. In Pi, Shift+Tab changes the thinking level.

| Pi level | Inkling effort |
|---|---:|
| low | 0.2 |
| medium | 0.7 |
| high | 0.9 |
| xhigh | 0.99 |

The default fine-tuning and eval effort is `0.9` (`high`). Always compare the base model and trained checkpoint at the same effort.

To see how effort changes a representative answer:

```text
/tinker inkling sweep --prompt "Solve a task like the ones in my eval" --efforts low,medium,high,xhigh --yes
```

## The four training budgets

| Budget | Uses API? | What it does |
|---|---:|---|
| `demo` | No | Prepares files, checks setup, and validates data |
| `smoke` | Yes | Runs a baseline eval and two training steps |
| `small` | Yes | Runs a short training and compares the checkpoint |
| `real` | Yes | Runs a larger confirmed experiment |

API-using stages require confirmation. In non-interactive commands, `--yes` is the confirmation.

## Useful commands

```text
/tinker inkling                         Inkling setup and model information
/tinker demo                            Create a no-cost example project
/tinker improve <data> --goal "..."     Run the guided workflow
/tinker doctor                          Check your environment
/tinker validate data/train.jsonl       Inspect data and token masks
/tinker next                            Show the next safe step
/tinker monitor logs/<run>              Watch training metrics
/tinker checkpoints logs/<run>          Find and register checkpoints
/model                                  Choose Inkling or a trained checkpoint
```

See the full [`/tinker` command reference](docs/commands.md).

## Other coding agents

Pi is the main interface, but Claude Code, Codex, Cursor, Copilot, Gemini CLI, and other shell-capable agents can use the same workflow.

From a repository checkout:

```bash
node scripts/agent-cli.mjs inkling
node scripts/agent-cli.mjs doctor
node scripts/agent-cli.mjs improve data.csv --goal "better answers" --budget demo
```

Ask your agent:

```text
Read AGENTS.md. Use pi-tinker to prepare and validate this data for Inkling.
Do not call the API or start training until I approve.
```

Generated Python and JSONL stay editable regardless of which agent you use. See [`docs/coding-agents.md`](docs/coding-agents.md).

## Troubleshooting

Run:

```text
/tinker doctor
```

If you are in Pi, you can also use:

```text
/skill:tinker-debug <paste the error>
```

When opening a GitHub issue, include the command you ran and the `/tinker doctor` output. Do not include your API key.

## More documentation

- [60-second demo](docs/demo.md)
- [Command reference](docs/commands.md)
- [Using other coding agents](docs/coding-agents.md)
- [Real training runbook](docs/real-training-runbook.md)
- [Design principles](docs/design.md)

## Development

```bash
npm test
npm pack --dry-run
```

## License and attribution

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

This package includes Pi-specific adaptations of skill content from Thinking Machines Lab's `tinker-cookbook` repository.
