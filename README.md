# pi-tinker

Fine-tune [Inkling](https://thinkingmachines.ai/inkling/) **or any active model supported by [Tinker](https://thinkingmachines.ai/tinker/)** inside [Pi](https://pi.dev).

`pi-tinker` helps you turn examples into a trained model without hiding the real work. It prepares data, writes editable Python, checks your setup, runs evals, starts small training runs, and lets you chat with checkpoints in Pi. Inkling has the richest first-class integration, but the generated Tinker Cookbook SFT workflow is not limited to Inkling.

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

Choose an Inkling model from `/model`. **Inkling-Small 64K is the default** (cheaper, same renderer and effort interface). Inkling-Small and full Inkling each have a 256K long-context option. All four support tools, images, streamed thinking, and adjustable reasoning effort.

### 2. Try the no-cost demo

Inside Pi:

```text
/tinker demo
/tinker next
/tinker doctor
```

This creates a tiny customer-support project with editable training data, Python scripts, and an eval. It stops before any API usage.

### 3. Fine-tune a Tinker model on your data

Start with CSV, JSON, JSONL, text files, Markdown, or a docs directory. Omit `--model` to use Inkling-Small, or pass any active [Tinker model ID](https://tinker-docs.thinkingmachines.ai/tinker/models/):

```text
/tinker improve data.csv --goal "better customer support answers" --budget demo
/tinker improve data.csv --goal "better extraction" --model Qwen/Qwen3.5-9B-Base --budget demo
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

It hashes the source, train/eval data, eval code, model, and effort so stale results are never reused. Eval comparisons fail closed on malformed scores, count mismatches, or effort/model mismatches. Candidate checkpoints stay separate from approved checkpoints; `deploy latest` only resolves a checkpoint that beat its matching baseline. `--force` only regenerates files—it cannot bypass quality or provenance checks. Dangerous overrides have explicit names such as `--accept-regression` and `--allow-unapproved`.

## Which model should I fine-tune?

### The 30-second answer

**If you are unsure, start with `thinkingmachines/Inkling-Small`.** It is the pi-tinker default because it is capable, multimodal, cheaper than full Inkling, and uses the same renderer and reasoning-effort interface. Move to another model only when your eval, context, or deployment requirements give you a reason.

| What you need | Good first candidate | Why |
|---|---|---|
| Easiest strong default; coding, grading, synthetic data, images/audio | `thinkingmachines/Inkling-Small` | Best-supported pi-tinker path; 64K context and automatic effort selection |
| Inkling quality is not enough on your eval | `thinkingmachines/Inkling` | Larger and more expensive; use only when it measurably beats Small |
| More than 64K context | The matching `:peft:262144` variant | 256K context costs more, so choose it only when examples actually need it |
| Compact, lower-cost, exportable model | `Qwen/Qwen3.5-4B` or `Qwen/Qwen3.5-9B` | Smaller dense models; useful when latency, cost, or self-hosting matters |
| Train from a less opinionated base model | `Qwen/Qwen3.5-9B-Base` | More control, but less chat-ready and usually needs stronger data/evals |
| Low-cost reasoning model | `openai/gpt-oss-20b` | Small reasoning option with a 32K context; export requires merged weights |
| Efficient medium model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | Low active parameter count; useful cost/quality candidate |
| Self-host after training | Prefer Qwen or Nemotron | pi-tinker can generate PEFT/export guidance; Inkling stays on Tinker |

Model IDs and availability change. The [live Tinker model list](https://tinker-docs.thinkingmachines.ai/tinker/models/) is the source of truth; pi-tinker refuses known retired IDs unless you explicitly acknowledge that risk.

### Choose by constraints, in this order

1. **Serving:** If you must self-host, do not choose Inkling. Start with an exportable Qwen or Nemotron model.
2. **Inputs:** For audio, choose Inkling. For images, choose a model marked Vision or Audio + Vision in Tinker's catalog.
3. **Context:** Use 64K or less unless your real examples require more. Long-context variants cost more.
4. **Budget and latency:** Start with the smallest plausible model. A well-chosen smaller model is easier to iterate on.
5. **Your held-out eval:** Compare candidates on the same reviewed eval and pick the cheapest model that meets your quality bar—not the model with the largest parameter count.

A practical comparison after the `demo` setup:

```text
/tinker eval baseline --model thinkingmachines/Inkling-Small --effort 0.9 --out eval_results/inkling-small.json --yes
/tinker eval baseline --model Qwen/Qwen3.5-9B --out eval_results/qwen-9b.json --yes
```

Inspect both summaries using the same `data/eval.jsonl`, then run managed fine-tuning with the winner:

```text
/tinker improve data.csv --goal "better customer support answers" --model thinkingmachines/Inkling-Small --budget smoke --eval-reviewed --yes
```

### What “any Tinker model” means

- The model must be on Tinker's current supported-model list; pi-tinker cannot upload an arbitrary Hugging Face model to Tinker.
- pi-tinker is strongest for editable Cookbook **SFT**. Learning rate, batch size, and other settings may need calibration for each model.
- Inkling gets automatic base-model chat registration and reasoning-effort selection. Other models still use the managed data, training, eval, checkpoint, and approval workflow.
- Export options depend on architecture: Inkling stays on Tinker; Qwen/Nemotron generally support PEFT workflows; some families require merged weights.

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
Read AGENTS.md. Use pi-tinker to prepare and validate this data for a Tinker-supported model.
Start with Inkling-Small unless my eval, context, budget, or deployment requirements suggest a better candidate.
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
