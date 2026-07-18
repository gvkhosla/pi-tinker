# pi-tinker repository instructions

- Read `AGENTS.md` and `docs/coding-agents.md` before changing workflows.
- Pi is the primary UI; shell-based agents can run `node scripts/agent-cli.mjs <subcommand>`.
- Use official Tinker SDK, Tinker Cookbook, and `tml-renderers`; do not create a parallel training framework.
- Keep generated Python and data files normal, editable, and inspectable.
- Default to `thinkingmachines/Inkling`; preserve support for explicitly selected alternative Tinker models.
- For Inkling, use Python 3.11+, Tinker SDK 0.23+, and `tinker-cookbook[inkling]`.
- Match baseline and checkpoint reasoning effort. Default Inkling SFT/eval effort is `0.9` unless the user deliberately changes it.
- Never trigger API usage or training without explicit approval. Smoke-test for two steps before scaling.
- Use `@trigger.dev/sdk` v4 patterns if unrelated Trigger.dev code is encountered; never use deprecated `client.defineJob`.
