# Changelog

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
