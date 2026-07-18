# Using pi-tinker with coding agents

Pi is the main interface. Other coding agents can run the same `/tinker` workflows through a small shell adapter.

## Pi

```bash
pi install git:github.com/gvkhosla/pi-tinker
pi
```

Then use `/tinker` commands directly:

```text
/tinker demo
/tinker doctor
/tinker improve data.csv --goal "better answers" --budget demo
```

Pi also lets you select Inkling or a trained checkpoint with `/model`.

## Claude Code, Codex, Cursor, Copilot, or Gemini CLI

From this repository, replace `/tinker` with `node scripts/agent-cli.mjs`:

| In Pi | In another coding agent's shell |
|---|---|
| `/tinker demo` | `node scripts/agent-cli.mjs demo` |
| `/tinker doctor` | `node scripts/agent-cli.mjs doctor` |
| `/tinker validate data/train.jsonl` | `node scripts/agent-cli.mjs validate data/train.jsonl` |
| `/tinker improve data.csv --budget demo` | `node scripts/agent-cli.mjs improve data.csv --budget demo` |

For example:

```bash
node scripts/agent-cli.mjs doctor
node scripts/agent-cli.mjs improve data.csv --goal "better support answers" --budget demo
```

The adapter prints the same report that Pi displays. Generated Python and JSONL are identical.

## Prompt to give your agent

```text
Read AGENTS.md. Use pi-tinker to prepare and validate my data for Inkling.
Show me the generated files. Do not call the API or start training until I approve.
Keep the baseline and checkpoint eval at the same reasoning effort.
```

## Safe workflow

1. Run `doctor`.
2. Prepare and inspect the data.
3. Create and inspect held-out eval examples.
4. Validate the renderer and token masks.
5. Ask before any command that uses the API.
6. Run a two-step smoke test before a larger run.
7. Compare the base model and checkpoint at the same Inkling effort.

The `demo` budget makes no API calls. The `smoke`, `small`, and `real` budgets do.

## Agent instruction files

The repository includes instructions for common agents:

- `AGENTS.md` — canonical guide for all agents
- `CLAUDE.md` — Claude Code
- `GEMINI.md` — Gemini CLI
- `.github/copilot-instructions.md` — GitHub Copilot
- `.cursor/rules/pi-tinker.mdc` — Cursor

All agents should use official Tinker and Tinker Cookbook APIs and keep the generated code visible and editable.
