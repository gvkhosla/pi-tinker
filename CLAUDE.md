# Claude Code instructions

Read and follow [`AGENTS.md`](AGENTS.md) as the canonical repository guide, then read [`docs/coding-agents.md`](docs/coding-agents.md).

Pi is the primary interactive runtime. From Claude Code, run the same safe workflows through:

```bash
node scripts/agent-cli.mjs <tinker-subcommand> [options]
```

Keep Tinker/Tinker Cookbook as the training layer, edit generated Python rather than hiding behavior, and never start API-using eval or training stages without explicit user approval.
