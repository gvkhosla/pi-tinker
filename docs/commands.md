# `/tinker` command reference

## `/tinker setup`

Checks local prerequisites:

- `TINKER_API_KEY`
- `python3`
- `uv`
- `tinker` CLI
- Python imports for `tinker` and `tinker_cookbook`

It does not install anything automatically. If packages are missing, install:

```bash
uv pip install tinker-cookbook
```

## `/tinker init [jsonl] [options]`

Guided golden-path setup for a chat SFT project. In interactive Pi, it asks for missing values; in print/non-interactive mode, pass them as arguments.

Example:

```text
/tinker init data/train.jsonl --model Qwen/Qwen3.5-9B-Base --metric "held-out support quality"
```

Generated files:

- `README.md`
- `train_sft.py`
- `eval_checkpoint.py`
- `tinker.yaml`
- `notes/plan.md`

Options are the same as `/tinker sft`, plus:

| Option | Default | Meaning |
|---|---:|---|
| `--metric` | prompt/TODO | What should improve before scale-up |

## `/tinker validate <jsonl> [options]`

Runs two validation layers:

1. lightweight JSONL/message-shape checks in TypeScript,
2. Python-backed Tinker Cookbook renderer/token-mask validation when dependencies are installed.

The Python-backed check loads the recommended renderer for the model, tokenizes examples, reports token-length stats, reports trainable assistant-token stats, and decodes previews.

Example:

```text
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `Qwen/Qwen3.5-9B-Base` | Model for renderer/tokenizer validation |
| `--examples` | `50` | Number of examples to tokenize/check |
| `--quick` | false | Only run lightweight JSONL checks |

## `/tinker sft <jsonl> [options]`

Scaffolds an editable supervised fine-tuning project using `tinker-cookbook` without the guided wizard.

Example:

```text
/tinker sft data/train.jsonl --model Qwen/Qwen3.5-9B-Base --steps 20
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `Qwen/Qwen3.5-9B-Base` | Tinker base model ID |
| `--steps` / `--max_steps` | `20` | Generated `max_steps` |
| `--batch-size` | `8` | Chat examples per batch |
| `--lr` / `--learning-rate` | `2e-4` | Learning rate |
| `--test-size` | `0` | Held-out examples for NLL eval |
| `--max-length` | `32768` | Max tokenized sequence length |
| `--log` | `logs/sft-<timestamp>` | Log/checkpoint directory |
| `--force` | false | Overwrite existing scaffold files |

## `/tinker smoke [script] [--yes]`

Runs a 2-step smoke test:

```bash
python3 train_sft.py max_steps=2
```

It asks for confirmation in interactive mode because this can create a real Tinker training client and incur small API usage. Pass `--yes` to skip confirmation.

Example:

```text
/tinker smoke train_sft.py --yes
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--timeout` | `1800000` | Timeout in milliseconds |

## `/tinker monitor <log_dir>`

Pins a live metrics widget above the editor and updates it every 5 seconds from:

- `metrics.jsonl`
- `checkpoints.jsonl`

Example:

```text
/tinker monitor logs/sft-2026-06-03T18-08-42
```

Stop monitoring:

```text
/tinker monitor --stop
```

## `/tinker checkpoints <log_dir>`

Reads `checkpoints.jsonl`, lists state/sampler checkpoints, and in interactive Pi lets you select a sampler checkpoint to register as a Pi model.

Example:

```text
/tinker checkpoints logs/sft-2026-06-03T18-08-42
```

## `/tinker status [log_dir]`

Shows recent Tinker runs via:

```bash
tinker run list --limit 10
```

If `log_dir` is provided, also reads `metrics.jsonl` and displays a compact view of the latest numeric metrics.

## `/tinker use <checkpoint> [alias]`

Registers a Tinker sampler checkpoint as a Pi model using Tinker's OpenAI-compatible inference endpoint.

Example:

```text
/tinker use tinker://0034...:train:0/sampler_weights/000080 my-sft
```

Then select it with:

```text
/model
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--alias` | `tinker-N` | Display name in Pi |
| `--context` | `32768` | Context window to advertise to Pi |
| `--max-tokens` | `4096` | Max output tokens to advertise to Pi |

Registrations are saved to:

```text
~/.pi/agent/tinker-checkpoints.json
```

## `/tinker use --list`

Lists registered checkpoint models.

## `/tinker use --remove <alias-or-tinker-path>`

Removes a registered checkpoint model.
