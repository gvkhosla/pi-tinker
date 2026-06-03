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

## `/tinker validate <jsonl>`

Runs a lightweight validation pass over a chat JSONL file.

Checks:

- file exists,
- each non-empty line is valid JSON,
- each row has `messages: [...]`,
- each message has an expected role,
- content is either a string or content-part array,
- basic user/assistant counts.

It also previews the first few conversations.

Example:

```text
/tinker validate data/train.jsonl
```

## `/tinker sft <jsonl> [options]`

Scaffolds an editable supervised fine-tuning project using `tinker-cookbook`.

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

Generated files:

- `train_sft.py`
- `eval_checkpoint.py`
- `tinker.yaml`
- `notes/plan.md`

Run the smoke test first:

```bash
python train_sft.py max_steps=2
```

Then inspect metrics and decoded samples before scaling.

## `/tinker status [log_dir]`

Shows recent Tinker runs via:

```bash
tinker run list --limit 10
```

If `log_dir` is provided, also reads `metrics.jsonl` and displays a compact view of the latest numeric metrics.

Example:

```text
/tinker status logs/sft-2026-06-03T18-08-42
```

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
