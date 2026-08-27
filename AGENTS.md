# Agent guide for pi-tinker

This repo is a Pi package that helps people run and fine-tune Inkling and other open-weight models with Tinker.

## Product promise

Help a non-expert go from:

```text
data → prepared JSONL → validation → baseline eval → smoke training → checkpoint → before/after comparison → chat in Pi
```

Do not turn this into a separate training framework. Keep Tinker/Tinker Cookbook as the real training layer and generate normal editable Python.

## Agent compatibility

This `AGENTS.md` is the canonical guide for every coding agent. Pi is the primary and richest interface, but Claude Code, Codex, Cursor, Copilot, Gemini CLI, and other shell-capable agents can run the same operator non-interactively:

```bash
node scripts/agent-cli.mjs inkling
node scripts/agent-cli.mjs doctor
node scripts/agent-cli.mjs validate data/train.jsonl --model thinkingmachines/Inkling-Small
```

When installed from npm, use `pi-tinker-agent` instead. Read `docs/coding-agents.md` for details. Other agents should inspect and edit the generated Python directly; they must not imitate hidden Pi UI state or trigger API-using stages without explicit user approval. `--force` is file-only; never add `--accept-regression`, `--allow-retired-model`, or `--allow-unapproved` unless the user explicitly requests that exact safety override.

## Best first commands for users

For Inkling itself:

```text
/tinker inkling
/tinker inkling sweep --prompt "a representative task" --efforts low,medium,high,xhigh --yes
/model
```

If the user has no data yet:

```text
/tinker demo
/tinker next
```

If the user has CSV/JSON/JSONL/docs, prefer the managed operator first:

```text
/tinker improve <input> --goal "what should improve" --budget demo
```

Then, only after the user understands API usage and evals:

```text
/tinker improve <input> --goal "what should improve" --budget smoke --eval-reviewed --yes
/tinker improve <input> --goal "what should improve" --budget small --yes
/tinker deploy latest
```

Manual path:

```text
/tinker improve <input> --model thinkingmachines/Inkling-Small --goal "what should improve" --budget demo
/tinker doctor
/tinker validate data/train.jsonl --model thinkingmachines/Inkling-Small
/tinker eval baseline --model thinkingmachines/Inkling-Small --effort 0.9 --yes
/tinker smoke train_sft.py --yes
/tinker next
```

If the user only wants conversion:

```text
/tinker prepare data.csv --out data/train.jsonl
```

If the user is stuck:

```text
/tinker doctor
/skill:tinker-debug <paste error or report>
```

## What is possible

- Start from a zero-data demo: `/tinker demo`.
- Run a managed improve loop with `/tinker improve`.
- Generate Tinker API snippets, Cookbook export files, and a serving decision with `/tinker deploy`.
- Convert CSV, JSON, JSONL prompt/response rows, TXT/MD files, or docs directories to chat JSONL.
- Scaffold editable Tinker Cookbook SFT scripts.
- Validate JSONL shape and, when dependencies are installed, renderer/token masks.
- Create and run an eval-first baseline/checkpoint comparison flow.
- Run a 2-step smoke test before spending real compute.
- Monitor logs and discover checkpoints.
- Register Inkling-Small (default) and full Inkling in 64K/256K variants, plus Tinker sampler checkpoints, as Pi models with tool use, vision, streamed thinking, and effort controls.
- Expose the same `/tinker` operator to other coding agents through the `pi-tinker-agent` shell adapter.

## What is not solved here

- It cannot create high-quality training data from nothing.
- It is strongest for SFT; advanced RL/DPO is mostly guided by skills, not the extension wizard.
- It does not run a production serving stack. `/tinker deploy` writes Tinker API clients plus Cookbook `weights.*` / Modal commands and an HTDYM pointer when the base model is in that catalog. Inkling stays on Tinker.
- Real training requires `TINKER_API_KEY`, Python 3.11+, Tinker SDK 0.23+, torch>=2.10, and `tinker-cookbook` (tml-renderers is a default dep).
- Prefer `thinkingmachines/Inkling-Small` unless the user asks for full Inkling. Do not train retired Tinker models; check https://tinker-docs.thinkingmachines.ai/tinker/models/.
- Do not vendor Cookbook research/debug skills. Point agents at `/plugin marketplace add thinking-machines-lab/tinker-cookbook`.
- Managed Inkling improve scores the held-out eval across efforts and pins one effort in training rendering, baseline eval, and checkpoint eval. Do not compare mismatched efforts.

## Development checks

Before committing changes:

```bash
npm test
npm pack --dry-run
```

The local integration test intentionally avoids real Tinker API usage.
