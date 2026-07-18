# Gemini CLI instructions

Read and follow [`AGENTS.md`](AGENTS.md) as the canonical repository guide, then read [`docs/coding-agents.md`](docs/coding-agents.md).

Pi is the primary interactive runtime. From Gemini CLI, invoke pi-tinker through:

```bash
node scripts/agent-cli.mjs <tinker-subcommand> [options]
```

Keep all generated Python editable, use official Tinker/Cookbook APIs and Inkling renderers, and require explicit approval before any API-using eval or training stage.
