---
name: tinker-debug
description: Pi overlay for diagnosing Tinker training, renderer, and deploy issues. Use when training is slow, hung, mismatched vs vLLM/SGLang, or an error is opaque. Canonical triage lives in Tinker Cookbook.
---

# Tinker debug (Pi overlay)

Do not duplicate Tinker Cookbook's debug skill. Start here, then load upstream.

## First in Pi

```text
/tinker doctor
```

That checks `TINKER_API_KEY`, Python 3.11+, Tinker SDK 0.23+, `torch>=2.10`, `tml_renderers`, Cookbook imports, retired model ids, and Inkling effort notes.

Install/upgrade:

```bash
uv pip install -U tinker-cookbook
```

There is no `[inkling]` extra. `tml-renderers` is a default Cookbook dependency.

## Canonical triage

```text
/plugin marketplace add thinking-machines-lab/tinker-cookbook
```

Then `/tinker:debug`. If the plugin is missing, read `skills/debug/SKILL.md` in `thinking-machines-lab/tinker-cookbook`.

## Pi-specific tells

| Symptom | Check |
|---|---|
| `No matching distribution for tinker-cookbook[inkling]` | Extra removed. Install `tinker-cookbook`. |
| `ModuleNotFoundError: tml_renderers` | Reinstall Cookbook; needs Python 3.11+. |
| `TmlV0Renderer requires PyTorch 2.10` | `pip install "torch>=2.10"`. |
| `get_lr(...Inkling) NotImplementedError` | Expected. Calibrate LR; Cookbook publishes no default. |
| Train vs eval mismatch on Inkling | Effort not pinned. Same float for data, baseline, checkpoint. |
| Deployed Inkling to vLLM | Wrong. Stay on Tinker. See `SERVING.md`. |
| Export of DeepSeek/Kimi/gpt-oss as PEFT then vLLM LoRA | Cookbook merge path, not adapter serving. |

Inspect generated Python, `metrics.jsonl`, and `checkpoints.jsonl`. Those are the source of truth, not Pi UI state.
