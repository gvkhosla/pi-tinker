---
name: tinker-inkling
description: Sample, evaluate, and post-train Inkling and Inkling-Small from Pi. Use when the user mentions Inkling, tml-renderers, TMLv0, thinking effort, Inkling-Small, or Inkling audio/images. Load before writing Inkling training or eval code.
---

# Inkling in Pi

Thinking Machines' models for Tinker. Same renderer (`tml_v0`), tokenizer, and effort interface.

| Id | Size | When |
|---|---|---|
| `thinkingmachines/Inkling-Small` | 276B / 12B active | **Default.** Coding, grading, synthetic data, cheaper runs. |
| `thinkingmachines/Inkling` | 975B / 41B active | Only if the user asks for full Inkling. |
| `thinkingmachines/Inkling:peft:262144` | full, 256K ctx | Long-context inference/training variant. |

Both are post-trained starting points, not base models.

## Setup

```bash
uv pip install -U tinker-cookbook
```

Needs Python 3.11+, Tinker SDK 0.23+, torch>=2.10. `tml-renderers` is included. There is no `[inkling]` extra.

```text
/tinker inkling
/model
/tinker inkling sweep --prompt "representative task" --efforts low,medium,high,xhigh --yes
```

Never `tokenizer.encode()` a chat prompt. Never hardcode the renderer name — use `model_info.get_recommended_renderer_name`.

## Effort

Finite scalar in `[0.0, 1.0)`. Renderer inserts the effort system message. **Set it at sampling and when building training data.** Default if omitted is `0.9` (high) — make that deliberate.

| none | minimal | low | medium | high | xhigh |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.1 | 0.2 | 0.7 | 0.9 | 0.99 |

- Same effort for train data, baseline eval, and checkpoint eval.
- Effort and `max_tokens` are independent. High effort can need 16k+.
- Eval at `temperature=1.0`. Lowering temperature is not a substitute for lowering effort.
- `effort=0.0` conditions toward no reasoning; it is not a hard off switch.

## Serving

Stay on Tinker. Cookbook has no Inkling merge/PEFT path. `/tinker deploy` writes API clients and a `SERVING.md` that says so. Do not send Inkling to HTDYM or vLLM.

## Training

Cookbook `get_lr("thinkingmachines/Inkling")` raises `NotImplementedError`. Calibrate LR. Watch entropy. Prefer Small unless quality on the user's eval requires full Inkling.

Canonical long-form: Cookbook `/tinker:inkling` or https://tinker-docs.thinkingmachines.ai/cookbook/inkling/
