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

Choose an Inkling model from `/model`. **Inkling-Small is the default** (cheaper, same renderer and effort interface). Full Inkling and the 256K variant are there if you need them. All three support tools, images, streamed thinking, and adjustable reasoning effort.

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

The `demo` budget deterministically holds out task-relevant eval rows, excludes them from training, and prepares everything without calling the API. Review `data/eval.jsonl`, then acknowledge that review when you run a small smoke test:

```text
/tinker improve data.csv --goal "better customer support answers" --budget smoke --eval-reviewed --yes
```

If the smoke run and eval look good:

```text
/tinker improve data.csv --goal "better customer support answers" --budget small --yes
```

`pi-tinker` guides you through:

```text
data → held-out eval → validation → baseline → smoke → training → checkpoint eval → approval → chat/deploy
```

It hashes the source, train/eval data, eval code, model, and effort so stale results are never reused. Candidate checkpoints stay separate from approved checkpoints; `deploy latest` only resolves a checkpoint that beat its matching baseline.

## Setup for fine-tuning

Chatting with Inkling only needs Pi and `TINKER_API_KEY`. Fine-tuning also needs Python 3.11+, Tinker SDK 0.23+, and PyTorch 2.10+ (`tml-renderers` is a default Cookbook dependency, not an extra):

```bash
uv pip install -U tinker-cookbook
```

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
deploy/<alias>/       Tinker API snippets, EXPORT.md, SERVING.md, export.py
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

When managed improve first uses the API, it scores the held-out eval at several efforts and selects the lowest effort tied for best accuracy. That effort is persisted and used for SFT rendering, baseline evaluation, and checkpoint evaluation. Pass `--effort` to pin one explicitly or `--no-sweep` to keep the current/default value.

To see how effort changes a representative answer:

```text
/tinker inkling sweep --prompt "Solve a task like the ones in my eval" --efforts low,medium,high,xhigh --yes
```

## The four training budgets

| Budget | Uses API? | What it does |
|---|---:|---|
| `demo` | No | Prepares files, checks setup, and validates data |
| `smoke` | Yes | Scores effort, runs baseline + two training steps, then evaluates the smoke checkpoint |
| `small` | Yes | Scales only after smoke wins, then evaluates and approves/rejects the checkpoint |
| `real` | Yes | Runs a larger confirmed experiment |

API-using stages require confirmation. In non-interactive commands, `--yes` is the confirmation.

## Useful commands

```text
/tinker demo                            Create a no-cost example project
/tinker improve <data> --goal "..."     The front door (demo → smoke → small)
/tinker next                            The one next command, filled in
/tinker doctor                          Check your environment
/tinker inkling                         Inkling-Small vs Inkling, effort
/tinker monitor logs/<run>              Watch training metrics
/tinker deploy latest                   Tinker API snippets + export plan
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
