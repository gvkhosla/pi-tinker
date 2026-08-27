# Changelog

## 0.9.3

Inkling-Small model metadata (adapted from [Cameron's PR #1](https://github.com/gvkhosla/pi-tinker/pull/1)):

- Registered the official `thinkingmachines/Inkling-Small:peft:262144` long-context model alongside the 64K default.
- Updated Inkling-Small's limited-time discounted pricing to $0.58/$1.44 (64K) and $1.16/$2.89 (256K).
- Preserve `:peft:<length>` context windows when registering trained checkpoints.
- Test exact model IDs so substring matches cannot hide a missing variant.

## 0.9.2

Trust patch:

- Added SHA-256 provenance for normalized source data, train/eval data, eval code, model, and Inkling effort. Stale baselines, candidates, and approvals are invalidated automatically.
- Managed improve now creates a deterministic task-relevant holdout, excludes those rows from training, detects duplicates, and requires `--eval-reviewed` before API usage. `--eval` accepts a separate reviewed eval.
- Inkling effort is selected by scoring the held-out eval at each effort (lowest effort tied for best accuracy), then pinned in generated SFT rendering, baseline eval, checkpoint eval, state, and `tinker.yaml`.
- Candidate and approved checkpoints are separate. `deploy latest` resolves only an approved checkpoint; rejected/unevaluated candidates require an explicit URI plus `--force`.
- Expanded local integration coverage for provenance invalidation, held-out generation, effort-capable Python, and approved-only deployment.

## 0.9.1

One front door, fail closed:

- `new`, `start`, `init`, and `finetune` now run `improve --budget demo`. Help is `demo` / `improve` / `next` / `doctor` / `inkling` / `monitor` / `deploy`.
- `/tinker next` prints one filled-in `improve` or `deploy` command.
- Smoke saves a checkpoint (`save_every=1`) and evals it. If it does not beat baseline, improve refuses to scale or register unless `--force`.
- Inkling effort is pinned (default 0.9) on API budgets. First run sweeps one eval prompt unless `--effort` or `--no-sweep` is set. Retired model ids are refused without `--force`.

## 0.9.0

Cookbook catch-up and serving decision:

- Dropped the removed `tinker-cookbook[inkling]` extra. Install is `uv pip install -U tinker-cookbook` (`tml-renderers` + `torch>=2.10` are default deps).
- Registered `thinkingmachines/Inkling-Small` and made it the default model. Full Inkling remains available.
- Synced the packaged model snapshot with Cookbook's live lineup; doctor warns on retired ids (including Llama 3.x).
- Doctor now checks PyTorch >= 2.10 and always imports `tml_renderers`.
- `/tinker deploy` still writes Tinker API clients, and now also writes `EXPORT.md`, `export.py`, and `SERVING.md` (Cookbook `weights.*`, optional Modal SGLang commands, HTDYM pointer for overlapping presets). Inkling stays on Tinker.
- Stopped vendoring Cookbook research/debug reference files. Packaged skills are thin Pi overlays that point at `/plugin marketplace add thinking-machines-lab/tinker-cookbook`. Added `tinker-inkling` for operator rules.

## 0.8.0

Inkling support:

- Registered `thinkingmachines/Inkling` (64K) and its 256K Tinker variant as Pi models.
- Switched Pi model chat to Tinker's Anthropic-compatible endpoint for tool use, image input, streamed thinking, and checkpoint support.
- Added Inkling reasoning-effort mappings and `/tinker inkling sweep` using Tinker's official Cookbook script.
- Made Inkling the default model for new, improve, validation, and eval workflows while retaining Qwen alternatives.
- Added Inkling-aware checkpoint metadata, dependency diagnostics, TMLv0 validation, and effort-matched before/after evals.
- Updated generated projects and documentation for Python 3.11+, Tinker 0.23+, `tml-renderers`, and `tinker-cookbook[inkling]`.
- Added the `pi-tinker-agent` non-interactive shell adapter plus repository guidance for Claude Code, Codex, Cursor, Copilot, Gemini CLI, and other coding agents.
- Restored and updated the packaged Tinker research/debug skills so full package tests and published skill discovery work again.
- Reworked the README, demo, agent guide, command reference, and in-Pi help around three simple paths: chat, free demo, and fine-tune.

## 0.7.0

Managed improvement and deployment:

- Added `/tinker improve` as a managed fine-tuning operator with `demo`, `smoke`, `small`, and `real` budgets.
- The improve operator prepares data, scaffolds project files, runs doctor/validation, runs baseline evals, smoke tests, scaled training, checkpoint evals, before/after comparisons, checkpoint registration, and data-improvement suggestions.
- Added `/tinker deploy` to generate `.env.example`, Python client, Node client, FastAPI wrapper, and README for a Tinker sampler checkpoint.
- Updated README, command docs, agent guidance, and local integration tests for the managed workflow.

## 0.6.2

Launch polish:

- Added GitHub issue templates for onboarding friction, dataset conversion problems, and training errors.
- Added `docs/demo.md` with the shortest no-data demo and expected outputs.
- Added a copy-paste Pi agent prompt to the README.
- Prepared repo metadata/release polish for public discovery.

## 0.6.1

Onboarding clarity pass:

- Added `/tinker demo` as the simplest zero-data first command.
- Added README sections for who this is for, 30-second demo, 10-minute path, copy-paste CSV, and agent guidance.
- Added `AGENTS.md` so coding agents know the intended workflows and boundaries.
- Added `examples/customer-support.csv` as a copy-paste starter dataset.

## 0.6.0

Golden-path onboarding upgrade:

- Added `/tinker new` / `/tinker finetune` as the fastest zero-to-first-run entrypoint.
- Added `/tinker prepare` to convert CSV, JSON, JSONL prompt/response rows, TXT/MD files, or docs directories into chat JSONL.
- Added `/tinker recommend` for beginner-friendly model/method/settings suggestions from a goal.
- Added `/tinker doctor` for setup and project readiness diagnostics.
- Added `/tinker examples` with concrete customer-support, structured-extraction, and concise-writing starter tasks.
- Updated README/docs around a 10-minute fine-tuning golden path.
- Added `npm test` and expanded local integration coverage for onboarding commands.

## 0.5.0

Beginner wizard:

- Added `/tinker start` as the step-by-step fine-tuning flow for non-experts.
- Added `/tinker next` to show current progress and the next recommended action.
- Added `/tinker reset` to clear wizard state for a project.
- Wizard state is stored in `.tinker-pi/state.json` and tracks data, model, metric, validation, eval, smoke test, checkpoint, comparison, and chat registration progress.
- Local integration tests now cover wizard start/next/reset.

## 0.4.0

Eval-first workflow:

- Added `/tinker eval init` to create an editable exact/contains-match eval harness.
- Added `/tinker eval baseline` to evaluate the base model before training.
- Added `/tinker eval checkpoint` to evaluate a Tinker sampler checkpoint with the same eval set.
- Added `/tinker eval compare` to show baseline vs checkpoint delta plus wins/regressions.
- Local integration tests now cover eval scaffolding and comparison.

## 0.3.0

Validation quality upgrade:

- `/tinker validate` now produces a clear data-readiness report: `READY`, `SMOKE ONLY`, or `FIX DATA FIRST`.
- Added richer renderer/token-mask checks using Tinker Cookbook.
- Added token-length stats, trainable-token stats, trainable-ratio stats, histograms, longest examples, over-max-length warnings, zero-trainable detection, empty-assistant detection, and token-volume estimates.
- Added decoded input previews and trainable assistant-token snippets so users can inspect exactly what the model will learn.

## 0.2.0

Golden-path usability improvements:

- Added `/tinker init` guided SFT project setup.
- Upgraded `/tinker validate` with Python-backed Tinker Cookbook renderer/token-mask checks.
- Added `/tinker smoke` for 2-step training smoke tests.
- Added `/tinker monitor` live metrics widget.
- Added `/tinker checkpoints` discovery and interactive checkpoint registration.
- Expanded command docs and design notes.

## 0.1.0

Initial shareable package:

- Pi package manifest.
- `/tinker` extension command.
- Tinker setup checks.
- Chat JSONL validation.
- Editable SFT scaffold generation.
- Tinker run/metrics status helper.
- Tinker sampler checkpoint registration as Pi models.
- Pi-adapted `tinker-research` and `tinker-debug` skills.
