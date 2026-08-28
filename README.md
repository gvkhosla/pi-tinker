# pi-tinker

Fine-tune models on [Tinker](https://thinkingmachines.ai/tinker/) from [Pi](https://pi.dev).

pi-tinker prepares your data, writes editable Python, runs small training jobs, and compares the trained checkpoint with the original model. It uses Tinker and Tinker Cookbook underneath; it is not a separate training framework.

## Quick start

Install pi-tinker:

```bash
pi install git:github.com/gvkhosla/pi-tinker
```

Set your Tinker API key and open Pi:

```bash
export TINKER_API_KEY="your-api-key"
pi
```

Try the free local demo:

```text
/tinker demo
/tinker next
```

The demo creates sample data, an eval, and Python scripts. It does not call the Tinker API.

## Fine-tune your data

Start with CSV, JSON, JSONL, Markdown, text files, or a directory of documents:

```text
/tinker improve data.csv --goal "better customer support answers" --budget demo
```

This prepares the project without using the API. Review these files:

```text
data/train.jsonl
data/eval.jsonl
train_sft.py
eval.py
```

Then run a two-step smoke test:

```text
/tinker improve --budget smoke --eval-reviewed --yes
```

If the checkpoint beats the original model, run a small training job:

```text
/tinker improve --budget small --yes
```

Use `/tinker next` at any time to get one filled-in next command.

## Which model should I use?

pi-tinker can fine-tune any active model supported by Tinker. It cannot upload an arbitrary Hugging Face model that Tinker does not provide.

**If you are unsure, use `thinkingmachines/Inkling-Small`.** It is the default and the best-supported path in pi-tinker.

Choose something else when you have a clear reason:

| Need | Start with |
|---|---|
| Simple default, coding, grading, images, or audio | `thinkingmachines/Inkling-Small` |
| Better quality than Inkling-Small on your eval | `thinkingmachines/Inkling` |
| Small model or easier self-hosting | `Qwen/Qwen3.5-4B` or `Qwen/Qwen3.5-9B` |
| A less opinionated base model | `Qwen/Qwen3.5-9B-Base` |
| Low-cost reasoning | `openai/gpt-oss-20b` |
| More than 64K context | A matching `:peft:262144` model |

Pass the model ID to `improve`:

```text
/tinker improve data.csv --goal "better extraction" --model Qwen/Qwen3.5-9B-Base --budget demo
```

A good rule is:

1. Start with the smallest model that might work.
2. Evaluate it on real held-out examples.
3. Try a larger model only if the smaller one misses your quality target.
4. Fine-tune the cheapest model that passes.

If you need to self-host, prefer an exportable Qwen or Nemotron model. Inkling stays on Tinker.

Model availability changes. Check the [current Tinker model list](https://tinker-docs.thinkingmachines.ai/tinker/models/) before starting a large run.

## Inkling

pi-tinker registers four Inkling models in Pi:

- Inkling-Small, 64K context (default)
- Inkling-Small, 256K context
- Inkling, 64K context
- Inkling, 256K context

Set `TINKER_API_KEY`, then use:

```text
/tinker inkling
/model
```

Inkling supports tools, images, streamed thinking, and reasoning effort. Inkling-Small and full Inkling use the same renderer and effort interface.

### Reasoning effort

| Pi level | Inkling effort |
|---|---:|
| low | 0.2 |
| medium | 0.7 |
| high | 0.9 |
| xhigh | 0.99 |

Managed training tests several effort values on your eval and selects the lowest effort tied for the best score. It uses that value for training, baseline evaluation, and checkpoint evaluation.

To sample one prompt at several effort levels:

```text
/tinker inkling sweep --prompt "Solve a task like the ones in my eval" --efforts low,medium,high,xhigh --yes
```

## Safety defaults

pi-tinker is conservative about API usage and checkpoints:

- `demo` makes no API calls.
- API commands require confirmation; `--yes` is explicit approval.
- Automatically held-out evals must be reviewed before training.
- Training only scales when the checkpoint beats the matching baseline.
- `deploy latest` only uses an approved checkpoint.
- Changes to data, eval code, model, or effort invalidate old results.
- `--force` only overwrites generated files; it does not bypass safety checks.

Advanced overrides are documented in [the command reference](docs/commands.md).

## Training budgets

| Budget | API use | What it does |
|---|---:|---|
| `demo` | No | Prepares and validates the project |
| `smoke` | Yes | Runs the baseline, two training steps, and checkpoint eval |
| `small` | Yes | Runs a short training job after smoke passes |
| `real` | Yes | Runs a larger confirmed experiment |

## Setup

Chatting with Inkling only needs Pi and `TINKER_API_KEY`.

Fine-tuning also needs Python 3.11+, Tinker SDK 0.23+, PyTorch 2.10+, and Tinker Cookbook:

```bash
uv pip install -U tinker-cookbook
```

Check your setup:

```text
/tinker doctor
```

## Data format

A simple CSV works:

```csv
question,answer
How do I cancel?,Go to Settings → Billing → Cancel subscription.
My order is late,Send us your order number and we will check it.
```

You can also use chat JSONL, one conversation per line:

```json
{"messages":[{"role":"user","content":"How do I reset my password?"},{"role":"assistant","content":"Go to Settings → Security → Reset password."}]}
```

## Generated files

pi-tinker writes normal files that you can inspect and edit:

```text
data/train.jsonl      training examples
data/eval.jsonl       held-out examples
train_sft.py          Tinker Cookbook SFT script
eval.py               baseline and checkpoint eval
tinker.yaml           run settings
notes/plan.md         experiment notes
deploy/<alias>/       API, export, and serving guidance
```

## Useful commands

```text
/tinker demo                            Create a local example
/tinker improve <data> --goal "..."     Prepare, train, and evaluate
/tinker next                            Show one next command
/tinker doctor                          Check the environment
/tinker inkling                         Explain Inkling and effort
/tinker monitor logs/<run>              Watch training metrics
/tinker deploy latest                   Generate deployment files
/model                                  Select Inkling or a trained checkpoint
```

See the full [command reference](docs/commands.md).

## Other coding agents

Claude Code, Codex, Cursor, Copilot, Gemini CLI, and other shell-capable agents can use the same workflow:

```bash
node scripts/agent-cli.mjs doctor
node scripts/agent-cli.mjs improve data.csv --goal "better answers" --budget demo
```

Ask the agent to read `AGENTS.md` and not use the API until you approve. See [Using other coding agents](docs/coding-agents.md).

## Troubleshooting

Run:

```text
/tinker doctor
```

In Pi, you can also use:

```text
/skill:tinker-debug <paste the error>
```

Do not include your API key in bug reports.

## Documentation

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

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
