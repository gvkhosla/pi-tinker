# Model lineup (snapshot)

Canonical live list: https://tinker-docs.thinkingmachines.ai/tinker/models/

Deprecations: https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/

This file is a snapshot of Tinker Cookbook's `skills/research/references/models.md` for offline Pi use. Prefer the docs URL when they disagree.

## Thinking Machines family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `thinkingmachines/Inkling` | Hybrid + Audio + Vision | MoE, 975B / 41B active | Large |
| `thinkingmachines/Inkling-Small` | Hybrid + Audio + Vision | MoE, 276B / 12B active | Medium |
| `thinkingmachines/Inkling:peft:262144` | Long-context Inkling | MoE | Large |

These render through `tml-renderers` (a **default** Cookbook dependency, not an `[inkling]` extra). Require an explicit thinking-effort value. Prefer Inkling-Small unless the user asks for full Inkling. Use the `tinker-inkling` skill.

## Qwen family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `Qwen/Qwen3.8-27B` | Hybrid + Vision | Dense | Medium |
| `Qwen/Qwen3.6-35B-A3B` | Hybrid + Vision | MoE | Medium |
| `Qwen/Qwen3.6-27B` | Hybrid + Vision | Dense | Medium |
| `Qwen/Qwen3.5-397B-A17B` | Hybrid + Vision | MoE | Large |
| `Qwen/Qwen3.5-35B-A3B-Base` | Base | MoE | Medium |
| `Qwen/Qwen3.5-9B` | Hybrid + Vision | Dense | Small |
| `Qwen/Qwen3.5-9B-Base` | Base | Dense | Small |
| `Qwen/Qwen3.5-4B` | Hybrid + Vision | Dense | Compact |
| `Qwen/Qwen3-8B` | Hybrid | Dense | Small |

Use the `_disable_thinking` renderer variant when you want direct instruction-following from a hybrid Qwen model.

## Nemotron family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | Hybrid | MoE | Medium |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | Hybrid | MoE | Large |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | Hybrid | MoE | Large |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Hybrid | MoE | Medium |

## Other families

| Model | Type | Arch | Size |
|-------|------|------|------|
| `openai/gpt-oss-120b` | Reasoning | MoE | Medium |
| `openai/gpt-oss-20b` | Reasoning | MoE | Small |
| `deepseek-ai/DeepSeek-V3.1` | Hybrid | MoE | Large |
| `moonshotai/Kimi-K2.6` | Hybrid + Vision | MoE | Large |

## HTDYM overlap (self-host pricing)

Only these Tinker ids currently map to an HTDYM preset. `/tinker deploy` writes that mapping into `SERVING.md`.

| Tinker id | HTDYM preset |
|---|---|
| `Qwen/Qwen3.8-27B` | Qwen3.8 27B |
| `Qwen/Qwen3.6-35B-A3B` | Qwen3.6 35B A3B |
| `openai/gpt-oss-120b` | gpt-oss-120b MXFP4/BF16 |
| `openai/gpt-oss-20b` | gpt-oss-20b MXFP4/BF16 |
| `moonshotai/Kimi-K2.6` | Kimi K2.6 INT4/BF16 |

Inkling is not in HTDYM. Llama 3.x is not on Tinker's current lineup.

## Renderer matching

```python
from tinker_cookbook import model_info
renderer_name = model_info.get_recommended_renderer_name(model_name)
```

Never hardcode renderer names.
