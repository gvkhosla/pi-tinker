# `/tinker` command reference

## `/tinker new [input] [options]`

The easiest entrypoint. Starts the golden path from CSV, JSON, JSONL, TXT/MD, a docs directory, or a built-in example.

```text
/tinker new data.csv --goal "better customer support answers"
/tinker new data/train.jsonl --goal "structured extraction accuracy"
/tinker new --example customer-support
```

If the input is not chat JSONL, `/tinker new` converts it to `data/train.jsonl`, scaffolds editable Python/eval files, creates `.tinker-pi/state.json`, recommends starter settings, and shows the next wizard step.

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--goal` / `--metric` | prompt/TODO | What should improve |
| `--model` | `Qwen/Qwen3.5-9B-Base` | Starter model |
| `--example` | false | Built-in example slug, e.g. `customer-support` |
| `--out` | `data/train.jsonl` | Output path when converting data |
| `--prepare` | false | Force conversion even for JSONL |
| `--force` | false | Overwrite generated files |

## `/tinker prepare <input> [options]`

Converts common data into Tinker chat JSONL.

Supported inputs:

- CSV with `question`/`answer`, `prompt`/`completion`, `input`/`output`, or `messages` columns.
- JSON array, or object with `examples[]`/`data[]`.
- JSONL with `messages[]` or prompt/response-style fields.
- TXT/MD files or directories for starter document-summary examples.

```text
/tinker prepare support.csv --out data/train.jsonl
```

## `/tinker recommend [goal] [options]`

Gives a simple starter model/method/settings recommendation and the next commands.

```text
/tinker recommend --goal "valid JSON extraction" --data data/train.jsonl
```

## `/tinker doctor [jsonl]`

Diagnoses local setup and project readiness: API key, Python, `uv`, Tinker CLI, Python imports, generated files, script compilation, selected data, and next wizard step.

```text
/tinker doctor data/train.jsonl
```

## `/tinker examples list|copy [slug]`

Lists or copies concrete starter examples.

```text
/tinker examples list
/tinker examples copy customer-support
/tinker new --example structured-extraction
```

Current examples:

- `customer-support`
- `structured-extraction`
- `concise-writing`

## `/tinker start [jsonl] [options]`

Beginner step-by-step fine-tuning wizard. This is the simplest entrypoint for people who have examples and want to improve an open model without knowing the Tinker internals.

```text
/tinker start data/train.jsonl --model Qwen/Qwen3.5-9B-Base --metric "support answer quality"
```

It creates project files, stores progress in `.tinker-pi/state.json`, and guides the user through:

1. environment setup,
2. data selection,
3. file creation,
4. validation,
5. baseline eval,
6. smoke test,
7. training/checkpoint discovery,
8. checkpoint eval,
9. before/after comparison,
10. registering the checkpoint for chat in Pi.

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | prompt / `Qwen/Qwen3.5-9B-Base` | Starter model |
| `--metric` | prompt | What should improve |
| `--log` | `logs/sft-<timestamp>` | Training log path |
| `--force` | false | Overwrite generated files |

## `/tinker next`

Shows wizard progress and the next recommended action.

## `/tinker reset`

Deletes `.tinker-pi/` wizard state for this project. It does not delete training data, generated scripts, logs, or checkpoints.

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

The Python-backed check loads the recommended renderer for the model and produces a data-readiness report: `READY`, `SMOKE ONLY`, or `FIX DATA FIRST`. It tokenizes examples, checks trainable assistant-token masks, reports token and trainable-token histograms, identifies zero-trainable rows, flags truncation risk, lists longest examples, estimates token volume per epoch, and shows decoded previews plus trainable-token snippets.

Example:

```text
/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `Qwen/Qwen3.5-9B-Base` | Model for renderer/tokenizer validation |
| `--examples` | `200` | Number of examples to tokenize/check |
| `--max-length` | `32768` | Length threshold for truncation-risk warnings |
| `--quick` | false | Only run lightweight JSONL checks |

## `/tinker eval init`

Creates a minimal editable eval harness:

- `eval.py`
- `data/eval.jsonl`

The eval JSONL format is:

```json
{"messages":[{"role":"user","content":"..."}],"expected":"...","match":"contains"}
```

Supported match modes:

- `contains`
- `exact`
- `prefix`

## `/tinker eval baseline [options]`

Runs `eval.py` against a base model and writes `eval_results/baseline.json` by default.

```text
/tinker eval baseline --model Qwen/Qwen3.5-9B-Base --yes
```

## `/tinker eval checkpoint <tinker-path> [options]`

Runs the same eval against a Tinker sampler checkpoint.

```text
/tinker eval checkpoint tinker://.../sampler_weights/... --model Qwen/Qwen3.5-9B-Base --yes
```

For both baseline and checkpoint:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `Qwen/Qwen3.5-9B-Base` | Base model / renderer model |
| `--data` | `data/eval.jsonl` | Eval JSONL |
| `--out` | `eval_results/baseline.json` or checkpoint-based name | Output JSON |
| `--limit` | all | Limit examples |
| `--max-tokens` | `128` | Generation limit |
| `--temperature` | `0.0` | Sampling temperature |
| `--yes` | false | Skip confirmation |

## `/tinker eval compare <baseline.json> <candidate.json>`

Compares two eval result files and reports accuracy delta, wins, and regressions.

```text
/tinker eval compare eval_results/baseline.json eval_results/step-20.json
```

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
