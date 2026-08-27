# `/tinker` command reference

Pi users run these as `/tinker ...`. Other coding agents replace `/tinker` with `node scripts/agent-cli.mjs`; see [`coding-agents.md`](coding-agents.md).

## Start here

You only need a few commands for the normal workflow:

| Goal | Command |
|---|---|
| See a free local demo | `/tinker demo` |
| Prepare your own data without API usage | `/tinker improve <data> --goal "..." --budget demo` |
| The one next command | `/tinker next` |
| Check your setup | `/tinker doctor` |
| Learn about Inkling | `/tinker inkling` |
| Choose Inkling or a checkpoint | `/model` |

`new`, `start`, `init`, and `finetune` are aliases for `improve --budget demo`. Managed improve hashes data/code/model/effort provenance. Smoke/small refuse to scale or approve a checkpoint that does not beat the matching baseline unless you pass `--force`; `deploy latest` resolves only an approved checkpoint.

The `demo` budget makes no API calls. Commands using `--yes` may sample or train through the Tinker API.

## `/tinker inkling [info|sweep]`

Shows Inkling setup and registers Inkling-Small (default) and full Inkling in both 64K and 256K variants. After setting `TINKER_API_KEY`, pick one with `/model`.

```text
/tinker inkling
/tinker inkling sweep --prompt "A representative task" --efforts low,medium,high,xhigh --yes
```

The sweep calls Tinker's official `sample_reasoning` script. Presets map to `low=0.2`, `medium=0.7`, `high=0.9`, and `xhigh=0.99`; raw floats in `[0, 1)` also work. Options: `--max-tokens`, `--temperature`, `--timeout`, and `--yes`.

Install Cookbook (includes `tml-renderers` and requires `torch>=2.10`):

```bash
uv pip install -U tinker-cookbook
```

There is no `[inkling]` extra anymore.

## `/tinker improve [input] [options]`

The easiest way to fine-tune. It prepares data, checks it, runs a small test, trains, evaluates the checkpoint, and registers it in Pi.

```text
/tinker improve data.csv --goal "better support answers" --budget demo
/tinker improve data.csv --goal "better support answers" --budget smoke --eval-reviewed --yes
/tinker improve data.csv --goal "better support answers" --budget small --yes
```

In order, it does this:

1. normalizes source data and deterministically holds out task-relevant eval rows,
2. excludes held-out rows from training and asks you to review the editable eval,
3. hashes source/train/eval/code/model provenance and invalidates stale results,
4. scores Inkling effort on the held-out eval and pins one value everywhere,
5. measures the base model and runs only two training steps first,
6. scales only after the checkpoint beats its matching baseline,
7. keeps rejected candidates separate and approves/registers only winners.

Budgets:

| Budget | API usage | Meaning |
|---|---:|---|
| `demo` | no | Setup, project files, doctor, lightweight validation. Stops before API calls. |
| `smoke` | yes | Baseline eval + 2-step smoke training. |
| `small` | yes | Smoke, then short training run, checkpoint eval, comparison, registration. |
| `real` | yes | Larger managed run. Still confirmation-first unless `--yes`. |

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--goal` / `--metric` | TODO | What should improve |
| `--budget` | `demo` | `demo`, `smoke`, `small`, or `real` |
| `--model` | `thinkingmachines/Inkling-Small` | Base model / renderer model |
| `--eval <jsonl>` | automatic holdout | Use a separate reviewed eval; all source rows remain in training |
| `--eval-reviewed` | false | Acknowledge review of an automatic `data/eval.jsonl` before API usage |
| `--effort` | selected / `0.9` | Pin Inkling training + baseline + checkpoint effort |
| `--efforts` | low,medium,high,xhigh | Efforts scored when selecting from eval behavior |
| `--no-sweep` | false | Skip automatic effort scoring and keep the pinned/default effort |
| `--steps` | budget default | Override scale-up max steps |
| `--yes` | false | Confirm API-using stages |
| `--force` | false | Overwrite generated files, rerun, or explicitly override a quality gate |
| `--alias` | generated | Alias for an approved registered checkpoint |
| `--register` | true | Register an approved checkpoint for chat |

## `/tinker deploy <checkpoint-or-alias> [alias] [options]`

Generates Tinker API clients **and** a serving decision. It does not stand up GPUs. `latest` means the latest checkpoint approved by managed evaluation, never merely the newest candidate. An explicit rejected/unevaluated candidate is blocked unless `--force` is supplied.

```text
/tinker deploy latest
/tinker deploy my-sft
/tinker deploy tinker://.../sampler_weights/... support-sft --model Qwen/Qwen3.8-27B
```

Writes by default:

```text
deploy/<alias>/README.md
deploy/<alias>/.env.example
deploy/<alias>/python_client.py
deploy/<alias>/node_client.mjs
deploy/<alias>/fastapi_app.py
deploy/<alias>/EXPORT.md
deploy/<alias>/export.py
deploy/<alias>/SERVING.md
```

- **Inkling / Inkling-Small:** stay on Tinker's endpoint. `export.py` exits with that message.
- **Qwen / Nemotron / gpt-oss / Kimi / DeepSeek:** `EXPORT.md` + `export.py` wrap Cookbook `weights.download` / `build_hf_model` / `build_lora_adapter`, plus optional Modal SGLang commands.
- **SERVING.md:** stay-on-Tinker vs self-host. If the base model is in HTDYM (Qwen3.8-27B, Qwen3.6-35B-A3B, gpt-oss 20B/120B, Kimi K2.6), it points at https://htdym.sailresearch.com with the matching preset. It does not run the estimator.

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--out` | `deploy/<alias>` | Output directory |
| `--alias` | positional/generated | Friendly name |
| `--model` / `--base-model` | checkpoint/wizard base | Architecture used for export + HTDYM mapping |
| `--force` | false | Overwrite existing files |

## `/tinker demo`

Zero-data demo. Copies the built-in `customer-support` example, creates a guided project, writes editable training/eval files, and shows the next step.

```text
/tinker demo
/tinker next
/tinker doctor
```

This is the best first command for someone who wants to see what `pi-tinker` can do before bringing real data.

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
| `--model` | `thinkingmachines/Inkling` | Starter model |
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
/tinker demo
/tinker new --example structured-extraction
```

Current examples:

- `customer-support`
- `structured-extraction`
- `concise-writing`

## `/tinker start [jsonl] [options]`

Beginner step-by-step fine-tuning wizard. This is the simplest entrypoint for people who have examples and want to improve an open model without knowing the Tinker internals.

```text
/tinker start data/train.jsonl --model thinkingmachines/Inkling --metric "support answer quality"
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
| `--model` | prompt / `thinkingmachines/Inkling` | Starter model |
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
- Python imports for `tinker`, `tinker_cookbook`, and `tml_renderers`
- Python 3.11+ for Inkling

It does not install anything automatically. If packages are missing, install:

```bash
uv pip install -U tinker-cookbook
```

## `/tinker init [jsonl] [options]`

Guided golden-path setup for a chat SFT project. In interactive Pi, it asks for missing values; in print/non-interactive mode, pass them as arguments.

Example:

```text
/tinker init data/train.jsonl --model thinkingmachines/Inkling --metric "held-out support quality"
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
/tinker validate data/train.jsonl --model thinkingmachines/Inkling
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `thinkingmachines/Inkling` | Model for renderer/tokenizer validation |
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
/tinker eval baseline --model thinkingmachines/Inkling --effort 0.9 --yes
```

## `/tinker eval checkpoint <tinker-path> [options]`

Runs the same eval against a Tinker sampler checkpoint.

```text
/tinker eval checkpoint tinker://.../sampler_weights/... --model thinkingmachines/Inkling --effort 0.9 --yes
```

For both baseline and checkpoint:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `thinkingmachines/Inkling` | Base model / renderer model |
| `--effort` | `0.9` | Inkling reasoning effort in `[0, 1)`; keep identical before/after |
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
/tinker sft data/train.jsonl --model thinkingmachines/Inkling --steps 20
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `thinkingmachines/Inkling` | Tinker base model ID |
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

Registers a Tinker sampler checkpoint as a Pi model using Tinker's Anthropic-compatible inference endpoint. Supply the base model so Inkling checkpoints retain reasoning, image, and effort capabilities.

Example:

```text
/tinker use tinker://0034...:train:0/sampler_weights/000080 my-sft --base-model thinkingmachines/Inkling
```

Then select it with:

```text
/model
```

Options:

| Option | Default | Meaning |
|---|---:|---|
| `--alias` | `tinker-N` | Display name in Pi |
| `--base-model` | unknown | Set to `thinkingmachines/Inkling` for an Inkling checkpoint |
| `--context` | `65536` for Inkling, otherwise `32768` | Context window to advertise to Pi |
| `--max-tokens` | `16384` for Inkling, otherwise `4096` | Max output tokens to advertise to Pi |

Registrations are saved to:

```text
~/.pi/agent/tinker-checkpoints.json
```

## `/tinker use --list`

Lists registered checkpoint models.

## `/tinker use --remove <alias-or-tinker-path>`

Removes a registered checkpoint model.
