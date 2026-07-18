# Using pi-tinker with coding agents

Pi is the primary interface because `pi-tinker` can register Inkling as a model, expose `/tinker` commands, render reports, monitor runs, and switch directly into trained checkpoints. The generated training and evaluation code is ordinary Python, so the workflow is not locked to Pi.

## Pi (recommended)

Install the package and use it interactively:

```bash
pi install git:github.com/gvkhosla/pi-tinker
pi
```

```text
/tinker inkling
/tinker doctor
/tinker new data.csv --model thinkingmachines/Inkling --goal "better answers"
```

## Claude Code, Codex, Cursor, Copilot, Gemini CLI, and shell-capable agents

Give the agent this instruction:

> Read `AGENTS.md` and `docs/coding-agents.md`. Use `pi-tinker-agent` for `/tinker` workflows. Inspect and edit the generated Python directly. Never run an API-using stage without explicit approval, and keep baseline/checkpoint Inkling effort identical.

When this repository is checked out, invoke the adapter directly:

```bash
node scripts/agent-cli.mjs inkling
node scripts/agent-cli.mjs doctor
node scripts/agent-cli.mjs validate data/train.jsonl --model thinkingmachines/Inkling
node scripts/agent-cli.mjs new data.csv --goal "better support answers"
```

When installed from npm, use its binary:

```bash
pi-tinker-agent inkling
pi-tinker-agent doctor
pi-tinker-agent --cwd /path/to/project new data.csv --goal "better support answers"
```

The adapter runs the same extension in Pi's non-interactive print mode. It does not create a chat session. Commands that can use the Tinker API remain confirmation-first and require `--yes` in non-interactive use.

## Agent-safe operating loop

1. Run `pi-tinker-agent inkling` and `pi-tinker-agent doctor`.
2. Prepare or inspect data without API usage.
3. Define and inspect held-out eval rows.
4. Validate renderer/token masks.
5. Ask for approval before baseline sampling or training.
6. Run a two-step smoke test before scaling.
7. Compare baseline and checkpoint at the same Inkling effort.
8. Leave generated `train_sft.py`, `eval.py`, JSONL, and YAML editable and visible.

## Repository instruction files

- `AGENTS.md` is the canonical tool-agnostic agent guide.
- `CLAUDE.md` and `GEMINI.md` direct those agents to the canonical guide.
- `.github/copilot-instructions.md` provides GitHub Copilot instructions.
- `.cursor/rules/pi-tinker.mdc` provides Cursor rules.

Do not duplicate the training framework inside an agent integration. Tinker and Tinker Cookbook remain the implementation layer; agents should operate the generated files and existing APIs.
