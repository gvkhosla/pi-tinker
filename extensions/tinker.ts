import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Box, Text } from "@earendil-works/pi-tui";
import { execFile as execFileCb } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFile = promisify(execFileCb);

const TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1";
const TINKER_ANTHROPIC_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api";
const INKLING_MODEL = "thinkingmachines/Inkling";
const INKLING_SMALL_MODEL = "thinkingmachines/Inkling-Small";
const INKLING_256K_MODEL = "thinkingmachines/Inkling:peft:262144";
const DEFAULT_MODEL = INKLING_SMALL_MODEL;
const COOKBOOK_INSTALL = "uv pip install -U tinker-cookbook";
const MODELS_DOCS_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models/";
const DEPRECATIONS_URL = "https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/";
const HTDYM_URL = "https://htdym.sailresearch.com";
const STATE_PATH = path.join(os.homedir(), ".pi", "agent", "tinker-checkpoints.json");
const MESSAGE_TYPE = "tinker-report";

type ReportLevel = "info" | "success" | "warning" | "error";

type CheckpointModel = {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  addedAt: number;
  baseModel?: string;
  reasoning?: boolean;
  vision?: boolean;
};

type TinkerState = {
  checkpoints: CheckpointModel[];
};

function isInklingModel(model?: string): boolean {
  return Boolean(model?.startsWith("thinkingmachines/Inkling"));
}

function starterModelChoices(): string[] {
  return [
    `${INKLING_SMALL_MODEL} — default; cheaper Inkling sibling (276B / 12B active)`,
    `${INKLING_MODEL} — full Inkling (975B / 41B active)`,
    "Qwen/Qwen3.5-9B-Base — cheaper small base model",
    "Qwen/Qwen3.5-35B-A3B-Base — stronger Qwen MoE base",
    "Qwen/Qwen3.8-27B — dense hybrid; exportable to self-host",
    "custom",
  ];
}

const RETIRED_MODEL_IDS = new Set([
  "meta-llama/Llama-3.3-70B-Instruct",
  "meta-llama/Llama-3.1-70B",
  "meta-llama/Llama-3.1-8B",
  "meta-llama/Llama-3.1-8B-Instruct",
  "meta-llama/Llama-3.2-3B",
  "meta-llama/Llama-3.2-1B",
  "moonshotai/Kimi-K2-Thinking",
  "moonshotai/Kimi-K2.5",
  "deepseek-ai/DeepSeek-V3.1-Base",
  "Qwen/Qwen3.5-27B",
  "Qwen/Qwen3.5-35B-A3B",
  "Qwen/Qwen3-32B",
  "Qwen/Qwen3-8B-Base",
  "Qwen/Qwen3-4B-Instruct-2507",
  "Qwen/Qwen3-235B-A22B-Instruct-2507",
  "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "Qwen/Qwen3-30B-A3B",
  "Qwen/Qwen3-30B-A3B-Base",
  "Qwen/Qwen3-VL-235B-A22B-Instruct",
  "Qwen/Qwen3-VL-30B-A3B-Instruct",
]);

function isRetiredModel(model?: string): boolean {
  if (!model) return false;
  if (model.startsWith("meta-llama/")) return true;
  return RETIRED_MODEL_IDS.has(model);
}

type ExportKind = "tinker-only" | "peft" | "merge-only" | "unknown";

function exportKindFor(baseModel?: string): ExportKind {
  if (!baseModel) return "unknown";
  if (isInklingModel(baseModel)) return "tinker-only";
  const id = baseModel.toLowerCase();
  // Cookbook: DeepSeek LoRA serving blocked; Kimi/gpt-oss conversion works but vLLM LoRA serving does not.
  if (id.includes("deepseek") || id.includes("kimi") || id.includes("gpt-oss")) return "merge-only";
  if (id.includes("qwen") || id.includes("nemotron")) return "peft";
  return "unknown";
}

const HTDYM_PRESETS: Record<string, string> = {
  "Qwen/Qwen3.8-27B": "Qwen3.8 27B",
  "Qwen/Qwen3.6-35B-A3B": "Qwen3.6 35B A3B",
  "openai/gpt-oss-120b": "gpt-oss-120b MXFP4/BF16",
  "openai/gpt-oss-20b": "gpt-oss-20b MXFP4/BF16",
  "moonshotai/Kimi-K2.6": "Kimi K2.6 INT4/BF16",
};

function htdymPreset(baseModel?: string): string | undefined {
  if (!baseModel) return undefined;
  return HTDYM_PRESETS[baseModel];
}

function inklingCompat() {
  return {
    supportsEagerToolInputStreaming: false,
    supportsLongCacheRetention: false,
    supportsCacheControlOnTools: false,
    forceAdaptiveThinking: true,
    allowEmptySignature: true,
    supportsToolReferences: false,
  } as const;
}

function inklingThinkingLevels() {
  return {
    minimal: null,
    low: "low",
    medium: "medium",
    high: "high",
    xhigh: "xhigh",
    max: "max",
  } as const;
}

function checkpointRegistration(options: {
  id: string;
  name: string;
  baseModel?: string;
  contextWindow?: number;
  maxTokens?: number;
}): CheckpointModel {
  const inkling = isInklingModel(options.baseModel);
  return {
    id: options.id,
    name: options.name,
    baseModel: options.baseModel,
    reasoning: inkling,
    vision: inkling,
    contextWindow: options.contextWindow ?? (inkling ? 65_536 : 32_768),
    maxTokens: options.maxTokens ?? (inkling ? 16_384 : 4_096),
    addedAt: Date.now(),
  };
}

function shellSplit(input: string): string[] {
  const out: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;
  let escaping = false;

  for (const ch of input.trim()) {
    if (escaping) {
      current += ch;
      escaping = false;
      continue;
    }
    if (ch === "\\" && quote !== "'") {
      escaping = true;
      continue;
    }
    if ((ch === "'" || ch === '"') && !quote) {
      quote = ch;
      continue;
    }
    if (ch === quote) {
      quote = undefined;
      continue;
    }
    if (/\s/.test(ch) && !quote) {
      if (current) out.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  if (current) out.push(current);
  return out;
}

function parseOptions(tokens: string[]): { positional: string[]; options: Record<string, string | boolean> } {
  const positional: string[] = [];
  const options: Record<string, string | boolean> = {};
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]!;
    if (!token.startsWith("--")) {
      positional.push(token);
      continue;
    }
    const raw = token.slice(2);
    const eq = raw.indexOf("=");
    if (eq >= 0) {
      options[raw.slice(0, eq)] = raw.slice(eq + 1);
      continue;
    }
    const next = tokens[i + 1];
    if (next && !next.startsWith("--")) {
      options[raw] = next;
      i++;
    } else {
      options[raw] = true;
    }
  }
  return { positional, options };
}

async function loadState(): Promise<TinkerState> {
  try {
    const raw = await readFile(STATE_PATH, "utf8");
    const parsed = JSON.parse(raw) as Partial<TinkerState>;
    return { checkpoints: Array.isArray(parsed.checkpoints) ? parsed.checkpoints : [] };
  } catch {
    return { checkpoints: [] };
  }
}

async function saveState(state: TinkerState): Promise<void> {
  await mkdir(path.dirname(STATE_PATH), { recursive: true });
  await writeFile(STATE_PATH, JSON.stringify(state, null, 2) + "\n");
}

function readJsonl(filePath: string): Record<string, unknown>[] {
  if (!existsSync(filePath)) return [];
  return readFileSync(filePath, "utf8")
    .split(/\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function compactMetrics(row: Record<string, unknown>, limit = 28): string {
  const interesting = Object.entries(row)
    .filter(([k, v]) =>
      typeof v === "number" &&
      (k.includes("loss") ||
        k.includes("score") ||
        k.includes("reward") ||
        k.includes("kl") ||
        k.includes("nll") ||
        k.startsWith("progress/") ||
        k.startsWith("time/total"))
    )
    .slice(0, limit);
  if (interesting.length === 0) return JSON.stringify(row, null, 2).slice(0, 2000);
  return interesting.map(([k, v]) => `- ${k}: ${typeof v === "number" ? Number(v.toFixed(6)) : v}`).join("\n");
}

function latestMetrics(metricsPath: string): string | undefined {
  if (!existsSync(metricsPath)) return undefined;
  try {
    const rows = readJsonl(metricsPath);
    const last = rows.slice(-1)[0];
    if (!last) return undefined;
    return compactMetrics(last);
  } catch (error) {
    return `Could not parse ${metricsPath}: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function monitorSummary(logDir: string): string[] {
  const metricsPath = path.join(logDir, "metrics.jsonl");
  const lines: string[] = [`Tinker monitor: ${path.basename(logDir)}`];
  try {
    const rows = readJsonl(metricsPath);
    const last = rows.slice(-1)[0];
    if (last) {
      const step = last["progress/batch"] ?? last["batch"] ?? rows.length;
      lines.push(`step: ${step}`);
      const lossEntry = Object.entries(last).find(([k, v]) => typeof v === "number" && k.includes("loss"));
      const rewardEntry = Object.entries(last).find(([k, v]) => typeof v === "number" && k.includes("reward"));
      const scoreEntry = Object.entries(last).find(([k, v]) => typeof v === "number" && k.includes("score"));
      if (lossEntry) lines.push(`${lossEntry[0]}: ${lossEntry[1]}`);
      if (rewardEntry) lines.push(`${rewardEntry[0]}: ${rewardEntry[1]}`);
      if (scoreEntry) lines.push(`${scoreEntry[0]}: ${scoreEntry[1]}`);
    } else {
      lines.push("metrics: waiting");
    }
  } catch (error) {
    lines.push(`metrics error: ${error instanceof Error ? error.message : String(error)}`);
  }
  try {
    const checkpoints = readCheckpoints(logDir);
    const last = checkpoints.filter((c) => c.sampler_path || c.state_path).slice(-1)[0];
    if (last) lines.push(`checkpoint: ${last.name}${last.final ? " (final)" : ""}`);
  } catch {}
  return lines.slice(0, 8);
}

type CheckpointRecord = {
  name?: string;
  batch?: number;
  epoch?: number;
  final?: boolean;
  state_path?: string;
  sampler_path?: string;
};

function readCheckpoints(logDir: string): CheckpointRecord[] {
  return readJsonl(path.join(logDir, "checkpoints.jsonl")) as CheckpointRecord[];
}

async function commandExists(cmd: string): Promise<boolean> {
  try {
    await execFile("which", [cmd], { timeout: 10_000 });
    return true;
  } catch {
    return false;
  }
}

function validateDataset(filePath: string): string {
  if (!existsSync(filePath)) return `❌ File not found: ${filePath}`;
  const lines = readFileSync(filePath, "utf8").split(/\n/).filter((line: string) => line.trim());
  let ok = 0;
  let assistantMessages = 0;
  let userMessages = 0;
  const issues: string[] = [];
  const samples: string[] = [];

  lines.forEach((line: string, idx: number) => {
    try {
      const obj = JSON.parse(line);
      if (!Array.isArray(obj.messages)) {
        issues.push(`line ${idx + 1}: missing messages[]`);
        return;
      }
      if (obj.messages.length < 2) issues.push(`line ${idx + 1}: fewer than 2 messages`);
      for (const [j, msg] of obj.messages.entries()) {
        if (!msg || typeof msg !== "object") {
          issues.push(`line ${idx + 1} message ${j}: not an object`);
          continue;
        }
        if (!["system", "user", "assistant", "tool"].includes(msg.role)) {
          issues.push(`line ${idx + 1} message ${j}: unexpected role ${JSON.stringify(msg.role)}`);
        }
        if (msg.role === "assistant") assistantMessages++;
        if (msg.role === "user") userMessages++;
        if (!(typeof msg.content === "string" || Array.isArray(msg.content))) {
          issues.push(`line ${idx + 1} message ${j}: content should be string or content-part array`);
        }
      }
      ok++;
      if (samples.length < 3) {
        const preview = obj.messages
          .map((m: any) => `${m.role}: ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`)
          .join("\n")
          .slice(0, 1200);
        samples.push(`### Example ${idx + 1}\n\n\`\`\`text\n${preview}\n\`\`\``);
      }
    } catch (error) {
      issues.push(`line ${idx + 1}: invalid JSON (${error instanceof Error ? error.message : String(error)})`);
    }
  });

  const warnings: string[] = [];
  if (assistantMessages === 0) warnings.push("No assistant messages found; SFT will have nothing obvious to train on.");
  if (userMessages === 0) warnings.push("No user messages found; expected chat JSONL may be malformed.");
  if (lines.length < 20) warnings.push("Very small dataset; use this only for smoke tests.");

  return [
    `# Dataset validation: ${filePath}`,
    `- JSONL rows: ${lines.length}`,
    `- Valid rows: ${ok}`,
    `- User messages: ${userMessages}`,
    `- Assistant messages: ${assistantMessages}`,
    issues.length ? `\n## Issues\n${issues.slice(0, 50).map((x) => `- ${x}`).join("\n")}${issues.length > 50 ? `\n- … ${issues.length - 50} more` : ""}` : "\n## Issues\nNone found by the lightweight validator.",
    warnings.length ? `\n## Warnings\n${warnings.map((x) => `- ${x}`).join("\n")}` : "",
    `\n## Samples\n${samples.join("\n\n")}`,
    "\nNext: run the Python-backed validator (`/tinker validate ... --model ...`) before spending real compute.",
  ].filter(Boolean).join("\n");
}

async function validateDatasetWithPython(filePath: string, model: string, maxExamples: number, maxLength: number): Promise<string> {
  const code = String.raw`
import json, sys, statistics
file_path, model_name, max_examples_s, max_length_s = sys.argv[1:5]
max_examples = int(max_examples_s)
max_length = int(max_length_s)

try:
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
except Exception as e:
    print(json.dumps({"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}))
    raise SystemExit(0)

try:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
except Exception as e:
    print(json.dumps({"ok": False, "stage": "renderer", "error": f"{type(e).__name__}: {e}"}))
    raise SystemExit(0)

BUCKETS = [512, 2048, 8192, 16384, 32768, 65536]

def flatten_values(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flatten_values(item))
        return out
    try:
        return [float(x)]
    except Exception:
        return []

def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""

def stats(xs):
    if not xs:
        return {"count": 0}
    xs_sorted = sorted(xs)
    return {
        "count": len(xs),
        "min": xs_sorted[0],
        "p25": xs_sorted[max(0, int((len(xs_sorted)-1)*0.25))],
        "median": statistics.median(xs_sorted),
        "p90": xs_sorted[max(0, int((len(xs_sorted)-1)*0.90))],
        "p95": xs_sorted[max(0, int((len(xs_sorted)-1)*0.95))],
        "max": xs_sorted[-1],
        "mean": sum(xs_sorted) / len(xs_sorted),
    }

def histogram(xs):
    counts = {f"<= {b}": 0 for b in BUCKETS}
    counts[f"> {BUCKETS[-1]}"] = 0
    for x in xs:
        placed = False
        for b in BUCKETS:
            if x <= b:
                counts[f"<= {b}"] += 1
                placed = True
                break
        if not placed:
            counts[f"> {BUCKETS[-1]}"] += 1
    return counts

def contiguous_ranges(indices):
    if not indices:
        return []
    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev + 1))
            start = prev = idx
    ranges.append((start, prev + 1))
    return ranges

rows = 0
valid_rows = 0
roles = {}
bad = []
no_user = []
no_assistant = []
empty_assistant = []
assistant_char_lengths = []

checked = 0
token_lengths = []
trainable_tokens = []
trainable_ratios = []
zero_trainable = []
over_max_length = []
top_longest = []
previews = []
trainable_previews = []
length_mismatch = []

try:
    with open(file_path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                bad.append({"line": line_number, "issue": f"invalid JSON: {type(e).__name__}: {e}"})
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list):
                bad.append({"line": line_number, "issue": "missing messages[]"})
                continue

            valid_rows += 1
            seen_user = False
            seen_assistant = False
            assistant_chars_this_row = 0

            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    bad.append({"line": line_number, "issue": f"message {j} is not an object"})
                    continue
                role = msg.get("role", "<missing>")
                roles[role] = roles.get(role, 0) + 1
                if role not in {"system", "user", "assistant", "tool"}:
                    bad.append({"line": line_number, "issue": f"message {j} has unexpected role {role!r}"})
                if role == "user":
                    seen_user = True
                if role == "assistant":
                    seen_assistant = True
                    txt = content_text(msg.get("content"))
                    assistant_chars_this_row += len(txt.strip())
                    if not txt.strip():
                        empty_assistant.append(line_number)
                content = msg.get("content")
                if not isinstance(content, (str, list)):
                    bad.append({"line": line_number, "issue": f"message {j} content should be string or content-part array"})

            if not seen_user:
                no_user.append(line_number)
            if not seen_assistant:
                no_assistant.append(line_number)
            assistant_char_lengths.append(assistant_chars_this_row)

            if checked >= max_examples:
                continue

            try:
                model_input, weights = renderer.build_supervised_example(
                    messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
                )
                toks = model_input.to_ints() if hasattr(model_input, "to_ints") else []
                w = flatten_values(weights)
                positive = [i for i, weight in enumerate(w) if weight > 0]
                checked += 1
                token_lengths.append(len(toks))
                trainable_tokens.append(len(positive))
                trainable_ratios.append((len(positive) / len(toks)) if toks else 0.0)

                if len(w) not in {len(toks), max(0, len(toks) - 1)}:
                    length_mismatch.append({"line": line_number, "tokens": len(toks), "weights": len(w)})
                if len(positive) == 0:
                    zero_trainable.append(line_number)
                if len(toks) > max_length:
                    over_max_length.append({"line": line_number, "tokens": len(toks)})

                top_longest.append({"line": line_number, "tokens": len(toks), "trainable": len(positive)})
                top_longest = sorted(top_longest, key=lambda x: x["tokens"], reverse=True)[:8]

                if len(previews) < 3:
                    previews.append({"line": line_number, "text": tokenizer.decode(toks[: min(len(toks), 500)])})
                if len(trainable_previews) < 5 and positive:
                    snippets = []
                    for start, end in contiguous_ranges(positive)[:4]:
                        snippet_tokens = toks[start:min(end, start + 160)]
                        snippets.append({"range": [start, end], "text": tokenizer.decode(snippet_tokens)})
                    trainable_previews.append({"line": line_number, "snippets": snippets})
            except Exception as e:
                bad.append({"line": line_number, "issue": f"renderer/tokenization failed: {type(e).__name__}: {e}"})
except Exception as e:
    print(json.dumps({"ok": False, "stage": "file", "error": f"{type(e).__name__}: {e}"}))
    raise SystemExit(0)

serious = []
if bad:
    serious.append(f"{len(bad)} data/renderer issue(s)")
if no_assistant:
    serious.append(f"{len(no_assistant)} row(s) with no assistant message")
if empty_assistant:
    serious.append(f"{len(empty_assistant)} empty assistant message(s)")
if zero_trainable:
    serious.append(f"{len(zero_trainable)} checked row(s) with zero trainable tokens")
if length_mismatch:
    serious.append(f"{len(length_mismatch)} checked row(s) with token/weight length mismatch")

warnings = []
if rows < 20:
    warnings.append("very small dataset; good for smoke tests only")
if over_max_length:
    warnings.append(f"{len(over_max_length)} checked row(s) exceed max_length={max_length}")
if checked < min(max_examples, valid_rows):
    warnings.append("not all rows were tokenized; increase --examples for a fuller audit")
if token_lengths and max(token_lengths) > max_length * 0.9:
    warnings.append("some examples are close to the max length; inspect truncation behavior")
if assistant_char_lengths and statistics.median(assistant_char_lengths) < 20:
    warnings.append("median assistant completion is very short; confirm this is intentional")

if serious:
    readiness = "FIX DATA FIRST"
elif warnings:
    readiness = "SMOKE ONLY"
else:
    readiness = "READY"

mean_tokens = (sum(token_lengths) / len(token_lengths)) if token_lengths else 0
mean_trainable = (sum(trainable_tokens) / len(trainable_tokens)) if trainable_tokens else 0
estimate = {
    "estimated_total_input_tokens_per_epoch": int(mean_tokens * valid_rows) if checked else 0,
    "estimated_trainable_tokens_per_epoch": int(mean_trainable * valid_rows) if checked else 0,
    "basis": f"estimated from {checked} checked row(s)",
}

print(json.dumps({
    "ok": not serious,
    "stage": "done",
    "readiness": readiness,
    "serious": serious,
    "warnings": warnings,
    "model": model_name,
    "renderer": renderer_name,
    "rows": rows,
    "valid_rows": valid_rows,
    "checked_examples": checked,
    "roles": roles,
    "token_lengths": stats(token_lengths),
    "trainable_tokens": stats(trainable_tokens),
    "trainable_ratios": stats(trainable_ratios),
    "assistant_chars": stats(assistant_char_lengths),
    "token_length_histogram": histogram(token_lengths),
    "trainable_token_histogram": histogram(trainable_tokens),
    "estimate": estimate,
    "bad": bad[:80],
    "bad_count": len(bad),
    "no_user": no_user[:20],
    "no_user_count": len(no_user),
    "no_assistant": no_assistant[:20],
    "no_assistant_count": len(no_assistant),
    "empty_assistant": empty_assistant[:20],
    "empty_assistant_count": len(empty_assistant),
    "zero_trainable": zero_trainable[:20],
    "zero_trainable_count": len(zero_trainable),
    "over_max_length": over_max_length[:20],
    "over_max_length_count": len(over_max_length),
    "length_mismatch": length_mismatch[:20],
    "length_mismatch_count": len(length_mismatch),
    "top_longest": top_longest,
    "previews": previews,
    "trainable_previews": trainable_previews,
}, ensure_ascii=False))
`;
  const { stdout } = await execFile("python3", ["-c", code, filePath, model, String(maxExamples), String(maxLength)], {
    timeout: 240_000,
    maxBuffer: 12 * 1024 * 1024,
  });
  const result = JSON.parse(stdout.trim().split(/\n/).slice(-1)[0] ?? "{}");
  if (!result.ok && result.stage !== "done") {
    const installCommand = COOKBOOK_INSTALL;
    return [
      `# Python-backed validation could not run`,
      `- Stage: ${result.stage ?? "unknown"}`,
      `- Error: ${result.error ?? "unknown"}`,
      "",
      "Install/upgrade dependencies with:",
      "```bash",
      installCommand,
      "```",
      "",
      "Falling back to lightweight JSONL checks is still useful, but renderer/token-mask validation requires Python dependencies.",
    ].join("\n");
  }

  const readiness = String(result.readiness ?? "UNKNOWN");
  const icon = readiness === "READY" ? "✅" : readiness === "SMOKE ONLY" ? "⚠️" : "❌";
  const fmtStats = (s: any) => s?.count ? `count=${s.count}, min=${s.min}, p50=${s.median}, p90=${s.p90}, p95=${s.p95}, max=${s.max}, mean=${Number(s.mean).toFixed(1)}` : "n/a";
  const fmtHist = (h: Record<string, number> | undefined) => h ? Object.entries(h).filter(([, v]) => v > 0).map(([k, v]) => `- ${k}: ${v}`).join("\n") || "- empty" : "- n/a";
  const issueLines = [
    ...(result.bad ?? []).map((x: any) => `- line ${x.line}: ${x.issue}`),
    result.no_assistant_count ? `- rows with no assistant message: ${result.no_assistant.join(", ")}${result.no_assistant_count > result.no_assistant.length ? " …" : ""}` : "",
    result.empty_assistant_count ? `- empty assistant rows: ${result.empty_assistant.join(", ")}${result.empty_assistant_count > result.empty_assistant.length ? " …" : ""}` : "",
    result.zero_trainable_count ? `- zero-trainable checked rows: ${result.zero_trainable.join(", ")}${result.zero_trainable_count > result.zero_trainable.length ? " …" : ""}` : "",
    result.length_mismatch_count ? `- token/weight length mismatches: ${(result.length_mismatch ?? []).map((x: any) => `line ${x.line} (${x.tokens} tokens/${x.weights} weights)`).join(", ")}` : "",
  ].filter(Boolean);
  const topLongest = (result.top_longest ?? []).map((x: any) => `- line ${x.line}: ${x.tokens} tokens, ${x.trainable} trainable`).join("\n") || "- n/a";
  const trainablePreviews = (result.trainable_previews ?? []).map((p: any) => {
    const snippets = (p.snippets ?? []).map((s: any) => `  - tokens ${s.range[0]}–${s.range[1]}:\n\n    ${String(s.text).replace(/\n/g, "\n    ").slice(0, 1000)}`).join("\n");
    return `### Line ${p.line}\n${snippets}`;
  }).join("\n\n") || "No trainable token previews available.";
  const decodedPreviews = (result.previews ?? []).map((p: any) => `### Line ${p.line}\n\n\`\`\`text\n${String(p.text).slice(0, 1600)}\n\`\`\``).join("\n\n") || "No decoded previews available.";

  const recommendation = readiness === "READY"
    ? "Run `/tinker smoke train_sft.py --yes`, inspect metrics, then scale only after defining an eval."
    : readiness === "SMOKE ONLY"
      ? "Run only a tiny smoke test for now. Inspect warnings, decoded previews, and trainable snippets before larger runs."
      : "Fix the data issues above before creating a training run.";

  return [
    `# Tinker data readiness report`,
    `## ${icon} ${readiness}`,
    `- File: \`${filePath}\``,
    `- Model: \`${result.model}\``,
    `- Recommended renderer: \`${result.renderer}\``,
    `- Rows: ${result.rows} (${result.valid_rows} valid JSONL conversation rows)`,
    `- Checked with tokenizer/renderer: ${result.checked_examples}`,
    `- Roles: ${Object.entries(result.roles ?? {}).map(([k, v]) => `${k}=${v}`).join(", ") || "none"}`,
    "",
    `## Recommendation\n${recommendation}`,
    (result.serious ?? []).length ? `\n## Must fix\n${(result.serious ?? []).map((x: string) => `- ${x}`).join("\n")}` : "",
    (result.warnings ?? []).length ? `\n## Warnings\n${(result.warnings ?? []).map((x: string) => `- ${x}`).join("\n")}` : "",
    `\n## Token stats\n- Input tokens: ${fmtStats(result.token_lengths)}\n- Trainable assistant tokens: ${fmtStats(result.trainable_tokens)}\n- Trainable ratio: ${fmtStats(result.trainable_ratios)}\n- Assistant text chars: ${fmtStats(result.assistant_chars)}`,
    `\n## Token length histogram\n${fmtHist(result.token_length_histogram)}`,
    `\n## Trainable-token histogram\n${fmtHist(result.trainable_token_histogram)}`,
    `\n## Token-volume estimate\n- Total input tokens / epoch: ~${result.estimate?.estimated_total_input_tokens_per_epoch ?? 0}\n- Trainable tokens / epoch: ~${result.estimate?.estimated_trainable_tokens_per_epoch ?? 0}\n- Basis: ${result.estimate?.basis ?? "n/a"}`,
    issueLines.length ? `\n## Detailed issues\n${issueLines.slice(0, 100).join("\n")}` : "\n## Detailed issues\nNo serious data/rendering issues found in checked examples.",
    `\n## Longest checked examples\n${topLongest}`,
    (result.over_max_length_count ?? 0) > 0 ? `\n## Over max length (${maxLength})\n${(result.over_max_length ?? []).map((x: any) => `- line ${x.line}: ${x.tokens} tokens`).join("\n")}` : "",
    `\n## Trainable token snippets\n${trainablePreviews}`,
    `\n## Decoded input previews\n${decodedPreviews}`,
  ].filter(Boolean).join("\n");
}

function makeSftScript(options: {
  dataFile: string;
  model: string;
  logPath: string;
  maxSteps: string;
  batchSize: string;
  learningRate: string;
  testSize: string;
  maxLength: string;
}) {
  return `import asyncio
import sys

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


class _EffortRenderer:
    """Pins Inkling SFT rendering to the same effort used by evals."""

    def __init__(self, renderer, effort: float):
        self._renderer = renderer
        self._effort = effort

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def build_supervised_example(self, messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES):
        return self._renderer.build_supervised_example(
            messages, train_on_what=train_on_what, effort=self._effort
        )


@chz.chz
class EffortConversationFileBuilder(FromConversationFileBuilder):
    effort: float = 0.9

    @property
    def renderer(self):
        renderer = super().renderer
        if self.common_config.model_name_for_tokenizer.startswith("thinkingmachines/Inkling"):
            return _EffortRenderer(renderer, self.effort)
        return renderer


def _training_effort(argv: list[str]) -> tuple[float, list[str]]:
    effort = 0.9
    remaining = []
    for arg in argv:
        if arg.startswith("effort="):
            effort = float(arg.split("=", 1)[1])
        else:
            remaining.append(arg)
    if not 0.0 <= effort < 1.0:
        raise SystemExit("effort must be in [0, 1)")
    return effort, remaining


def build_config_blueprint(effort: float = 0.9) -> chz.Blueprint[train.Config]:
    model_name = ${JSON.stringify(options.model)}
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=${options.maxLength},
        batch_size=${options.batchSize},
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = EffortConversationFileBuilder(
        common_config=common_config,
        file_path=${JSON.stringify(options.dataFile)},
        test_size=${options.testSize},
        shuffle_seed=0,
        effort=effort,
    )

    return chz.Blueprint(train.Config).apply({
        "log_path": ${JSON.stringify(options.logPath)},
        "model_name": model_name,
        "recipe_name": "pi_tinker_sft",
        "renderer_name": renderer_name,
        "dataset_builder": dataset,
        "learning_rate": ${options.learningRate},
        "lr_schedule": "linear",
        "num_epochs": 1,
        "lora_rank": 32,
        "save_every": 20,
        "eval_every": 10,
        "max_steps": ${options.maxSteps},
    })


def main(config: train.Config):
    print("Resolved Tinker SFT config:")
    print(config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    effort, argv = _training_effort(sys.argv[1:])
    print(f"Pinned training effort: {effort}")
    blueprint = build_config_blueprint(effort)
    blueprint.make_from_argv(argv)
    main(blueprint.make())
`;
}

function makeEvalScript() {
  return `"""\nOptional quick checkpoint smoke test.\n\nUsage:\n  python eval_checkpoint.py 'tinker://.../sampler_weights/...'\n"""\nimport asyncio\nimport sys\n\nimport tinker\n\n\nasync def main(model_path: str):\n    svc = tinker.ServiceClient()\n    sc = await svc.create_sampling_client_async(model_path=model_path)\n    tok = sc.get_tokenizer()\n    prompt = tinker.ModelInput.from_ints(tok.encode("The best way to test a fine-tuned model is"))\n    result = await sc.sample_async(\n        prompt=prompt,\n        num_samples=1,\n        sampling_params=tinker.SamplingParams(max_tokens=80, temperature=0.7),\n    )\n    seq = result.sequences[0] if hasattr(result, "sequences") else result.samples[0]\n    print(tok.decode(seq.tokens))\n\n\nif __name__ == "__main__":\n    if len(sys.argv) != 2:\n        raise SystemExit("Usage: python eval_checkpoint.py 'tinker://.../sampler_weights/...'" )\n    asyncio.run(main(sys.argv[1]))\n`;
}

function makeProjectReadme(options: { model: string; dataFile: string; logPath: string; successMetric: string }) {
  return `# Tinker fine-tuning project\n\nThis project was scaffolded by \`pi-tinker\`. The important files are normal editable Python, not hidden framework state.\n\n## Goal\n\nFine-tune \`${options.model}\` on:\n\n\`${options.dataFile}\`\n\n## Success metric\n\n${options.successMetric || "Define this before scaling beyond a smoke test."}\n\n## Smoke test\n\n\`\`\`bash\n${COOKBOOK_INSTALL}\npython train_sft.py max_steps=2\n\`\`\`\n\n${isInklingModel(options.model) ? "Inkling uses its recommended TMLv0 renderer automatically. This scaffold accepts `effort=<float>` and renders SFT examples at that effort; managed improve pins the same value for training, baseline, and checkpoint evals.\n\n" : ""}## Monitor\n\nInside Pi:\n\n\`\`\`text\n/tinker monitor ${options.logPath}\n/tinker status ${options.logPath}\n/tinker checkpoints ${options.logPath}\n\`\`\`\n\n## Scale up\n\nOnly scale after checking:\n\n- JSONL validation passed\n- renderer/token validation passed\n- smoke test produced metrics\n- decoded examples look correct\n- success metric/eval is defined\n\n\`\`\`bash\npython train_sft.py max_steps=100\n\`\`\`\n\n## Chat with a checkpoint in Pi\n\nAfter a sampler checkpoint appears in \`checkpoints.jsonl\`:\n\n\`\`\`text\n/tinker checkpoints ${options.logPath}\n/model\n\`\`\`\n`;
}

function makeExampleEvalJsonl() {
  return [
    {
      messages: [{ role: "user", content: "Rewrite this to be concise: We are extremely sorry for the inconvenience and are investigating." }],
      expected: "Sorry for the inconvenience — we’re investigating.",
      match: "contains",
      notes: "Replace with examples from your real target distribution.",
    },
    {
      messages: [{ role: "user", content: "Classify sentiment: The setup was fast and the docs were clear." }],
      expected: "positive",
      match: "contains",
    },
  ].map((row) => JSON.stringify(row)).join("\n") + "\n";
}


type PreparedDataset = {
  inputPath: string;
  outFile: string;
  format: string;
  rows: number;
  warnings: string[];
  samples: string[];
};

type ExampleTemplate = {
  slug: string;
  title: string;
  goal: string;
  trainRows: Array<Record<string, unknown>>;
  evalRows: Array<Record<string, unknown>>;
};

const EXAMPLE_TEMPLATES: ExampleTemplate[] = [
  {
    slug: "customer-support",
    title: "Customer support tone + answer quality",
    goal: "More accurate, concise, empathetic customer support answers",
    trainRows: [
      { messages: [{ role: "user", content: "My package says delivered but it isn't here. What do I do?" }, { role: "assistant", content: "I’m sorry — that’s frustrating. Please check nearby doors or your building mail area first. If it still hasn’t arrived, send us your order number and we’ll open a carrier trace or arrange a replacement." }] },
      { messages: [{ role: "user", content: "Can I cancel my subscription?" }, { role: "assistant", content: "Yes. Go to Settings → Billing → Cancel subscription. Your access stays active until the end of the current billing period." }] },
      { messages: [{ role: "user", content: "The app keeps crashing after login." }, { role: "assistant", content: "Sorry about that. Please update to the latest version, restart the app, and try again. If it still crashes, send us your device type and app version so we can investigate." }] },
    ],
    evalRows: [
      { messages: [{ role: "user", content: "I was charged twice this month." }], expected: "sorry", match: "contains" },
      { messages: [{ role: "user", content: "Where do I cancel?" }], expected: "Settings", match: "contains" },
    ],
  },
  {
    slug: "structured-extraction",
    title: "Structured JSON extraction",
    goal: "Higher valid-JSON extraction accuracy on domain text",
    trainRows: [
      { messages: [{ role: "user", content: "Extract JSON: Jane Doe works at Acme as VP Sales." }, { role: "assistant", content: '{"name":"Jane Doe","company":"Acme","title":"VP Sales"}' }] },
      { messages: [{ role: "user", content: "Extract JSON: Omar Khan, CTO, Northstar Labs." }, { role: "assistant", content: '{"name":"Omar Khan","company":"Northstar Labs","title":"CTO"}' }] },
      { messages: [{ role: "user", content: "Extract JSON: No company listed for Priya Shah." }, { role: "assistant", content: '{"name":"Priya Shah","company":null,"title":null}' }] },
    ],
    evalRows: [
      { messages: [{ role: "user", content: "Extract JSON: Alex Lee is Head of Ops at Meridian." }], expected: '"company":"Meridian"', match: "contains" },
      { messages: [{ role: "user", content: "Extract JSON: Taylor Smith, no role, Orbit." }], expected: '"company":"Orbit"', match: "contains" },
    ],
  },
  {
    slug: "concise-writing",
    title: "Concise rewriting",
    goal: "Shorter, clearer rewrites that preserve meaning",
    trainRows: [
      { messages: [{ role: "user", content: "Rewrite concisely: We are reaching out today in order to inform you that your requested refund has now been successfully processed." }, { role: "assistant", content: "Your refund has been processed." }] },
      { messages: [{ role: "user", content: "Rewrite concisely: Due to the fact that the meeting was moved, we will need to make an adjustment to the current schedule." }, { role: "assistant", content: "Because the meeting moved, we need to adjust the schedule." }] },
      { messages: [{ role: "user", content: "Rewrite concisely: At this point in time, we do not have sufficient information to make a final decision." }, { role: "assistant", content: "We don’t have enough information to decide yet." }] },
    ],
    evalRows: [
      { messages: [{ role: "user", content: "Rewrite concisely: We wanted to let you know that your account has been approved." }], expected: "approved", match: "contains" },
      { messages: [{ role: "user", content: "Rewrite concisely: In the event that you need assistance, please contact support." }], expected: "support", match: "contains" },
    ],
  },
];

function jsonl(rows: Array<Record<string, unknown>>): string {
  return rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    const next = text[i + 1];
    if (quoted && ch === '"' && next === '"') {
      field += '"';
      i++;
      continue;
    }
    if (ch === '"') {
      quoted = !quoted;
      continue;
    }
    if (ch === "," && !quoted) {
      row.push(field);
      field = "";
      continue;
    }
    if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && next === "\n") i++;
      row.push(field);
      if (row.some((x) => x.trim())) rows.push(row);
      row = [];
      field = "";
      continue;
    }
    field += ch;
  }
  row.push(field);
  if (row.some((x) => x.trim())) rows.push(row);
  return rows;
}

function normalizeHeader(header: string): string {
  return header.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function pickColumn(headers: string[], candidates: string[]): number {
  const normalized = headers.map(normalizeHeader);
  for (const candidate of candidates) {
    const idx = normalized.indexOf(candidate);
    if (idx >= 0) return idx;
  }
  for (const candidate of candidates) {
    const idx = normalized.findIndex((h) => h.includes(candidate));
    if (idx >= 0) return idx;
  }
  return -1;
}

function rowToTrainingExample(row: Record<string, unknown>): Record<string, unknown> | undefined {
  if (Array.isArray(row.messages)) return { messages: row.messages };
  const system = typeof row.system === "string" ? row.system : typeof row.instruction === "string" ? row.instruction : undefined;
  const input = String(row.user ?? row.question ?? row.prompt ?? row.input ?? row.query ?? "").trim();
  const output = String(row.assistant ?? row.answer ?? row.completion ?? row.output ?? row.response ?? "").trim();
  if (!input || !output) return undefined;
  const messages: Array<Record<string, string>> = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: input });
  messages.push({ role: "assistant", content: output });
  return { messages };
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function hashFile(filePath?: string): string | undefined {
  return filePath && existsSync(filePath) ? sha256(readFileSync(filePath)) : undefined;
}

function normalizedJsonlHash(filePath: string): string {
  const rows = readFileSync(filePath, "utf8").split(/\n/).filter((line) => line.trim()).map((line) => JSON.parse(line) as Record<string, unknown>);
  return sha256(jsonl(rows));
}

function provenanceFingerprint(parts: Record<string, unknown>): string {
  return sha256(JSON.stringify(parts, Object.keys(parts).sort()));
}

function messageText(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content.map((part) => part && typeof part === "object" && "text" in part ? String((part as { text?: unknown }).text ?? "") : "").join("\n").trim();
  }
  return "";
}

function trainingRowToEval(row: Record<string, unknown>): Record<string, unknown> | undefined {
  if (!Array.isArray(row.messages)) return undefined;
  const messages = row.messages as Array<Record<string, unknown>>;
  let assistantIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "assistant" && messageText(messages[i]?.content)) {
      assistantIndex = i;
      break;
    }
  }
  if (assistantIndex <= 0) return undefined;
  const expected = messageText(messages[assistantIndex]?.content);
  return {
    messages: messages.slice(0, assistantIndex),
    expected,
    match: "exact",
    notes: "Auto-held-out target. Review the prompt, expected answer, and matcher before API usage.",
  };
}

type ManagedDataResult = {
  sourceFile: string;
  trainFile: string;
  evalFile: string;
  sourceHash: string;
  trainingHash: string;
  evalHash: string;
  sourceRows: number;
  trainRows: number;
  evalRows: number;
  evalGenerated: boolean;
  warnings: string[];
};

async function writeManagedDataFiles(options: {
  cwd: string;
  normalizedSource: string;
  evalOverride?: string;
  preserveGeneratedEval?: boolean;
}): Promise<ManagedDataResult> {
  const warnings: string[] = [];
  const rows = readFileSync(options.normalizedSource, "utf8").split(/\n/).filter((line) => line.trim()).map((line) => JSON.parse(line) as Record<string, unknown>);
  if (!rows.length) throw new Error("No usable source rows found.");
  const sourceFile = path.join(options.cwd, ".tinker-pi", "source.jsonl");
  const trainFile = path.join(options.cwd, "data", "train.jsonl");
  const evalFile = path.join(options.cwd, "data", "eval.jsonl");
  await mkdir(path.dirname(sourceFile), { recursive: true });
  await mkdir(path.dirname(trainFile), { recursive: true });
  await writeFile(sourceFile, jsonl(rows));

  let trainRows = rows;
  let evalRows: Array<Record<string, unknown>> = [];
  let evalGenerated = false;
  if (options.evalOverride) {
    const evalText = readFileSync(options.evalOverride, "utf8");
    evalRows = evalText.split(/\n/).filter((line) => line.trim()).map((line) => JSON.parse(line) as Record<string, unknown>);
    await writeFile(evalFile, jsonl(evalRows));
  } else {
    evalGenerated = true;
    const ranked = rows.map((row, index) => ({ row, index, rank: sha256(JSON.stringify(row)) })).sort((a, b) => a.rank.localeCompare(b.rank));
    const evalCount = rows.length >= 10 ? Math.min(50, Math.max(2, Math.floor(rows.length * 0.2))) : rows.length >= 4 ? 2 : rows.length >= 2 ? 1 : 0;
    const heldoutIndexes = new Set(ranked.slice(0, evalCount).map((item) => item.index));
    trainRows = rows.filter((_row, index) => !heldoutIndexes.has(index));
    evalRows = ranked.slice(0, evalCount).map((item) => trainingRowToEval(item.row)).filter((row): row is Record<string, unknown> => Boolean(row));
    if (options.preserveGeneratedEval && existsSync(evalFile)) {
      evalRows = readFileSync(evalFile, "utf8").split(/\n/).filter((line) => line.trim()).map((line) => JSON.parse(line) as Record<string, unknown>);
    } else {
      await writeFile(evalFile, evalRows.length ? jsonl(evalRows) : "");
    }
    warnings.push(`Held out ${evalRows.length} of ${rows.length} rows deterministically; held-out rows are excluded from training.`);
    warnings.push("Auto-generated evals use exact matching. Review `data/eval.jsonl` and acknowledge it with `--eval-reviewed` before API usage.");
  }
  await writeFile(trainFile, jsonl(trainRows));
  const duplicateCount = rows.length - new Set(rows.map((row) => sha256(JSON.stringify(row)))).size;
  if (duplicateCount) warnings.push(`${duplicateCount} duplicate source row(s) detected; remove them before a serious run.`);
  if (options.evalOverride) {
    const trainPrompts = new Set(trainRows.map((row) => JSON.stringify(Array.isArray(row.messages) ? row.messages.filter((message: any) => message?.role !== "assistant") : [])));
    const leaked = evalRows.filter((row) => trainPrompts.has(JSON.stringify(Array.isArray(row.messages) ? row.messages.filter((message: any) => message?.role !== "assistant") : []))).length;
    if (leaked) warnings.push(`${leaked} eval prompt(s) also appear in training data. Remove leakage before trusting the comparison.`);
  }
  if (evalRows.length < 2) warnings.push("Fewer than 2 eval rows: demo setup is allowed, but API-using improve stages will stop.");
  return {
    sourceFile,
    trainFile,
    evalFile,
    sourceHash: hashFile(sourceFile)!,
    trainingHash: hashFile(trainFile)!,
    evalHash: hashFile(evalFile)!,
    sourceRows: rows.length,
    trainRows: trainRows.length,
    evalRows: evalRows.length,
    evalGenerated,
    warnings,
  };
}

function collectTextFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const target = path.join(dir, name);
    const stat = statSync(target);
    if (stat.isDirectory()) out.push(...collectTextFiles(target));
    else if (/\.(md|mdx|txt)$/i.test(name)) out.push(target);
  }
  return out;
}

async function prepareDataset(inputPath: string, outFile: string): Promise<PreparedDataset> {
  const warnings: string[] = [];
  const rows: Array<Record<string, unknown>> = [];
  const stat = statSync(inputPath);
  let format = "unknown";

  if (stat.isDirectory()) {
    format = "docs-directory";
    for (const file of collectTextFiles(inputPath).slice(0, 500)) {
      const text = readFileSync(file, "utf8").replace(/\s+/g, " ").trim();
      if (!text) continue;
      const chunks = text.match(/.{1,3500}(?:\s|$)/g) ?? [text.slice(0, 3500)];
      for (const [idx, chunk] of chunks.slice(0, 20).entries()) {
        rows.push({
          messages: [
            { role: "user", content: `Summarize the useful facts from ${path.basename(file)}${chunks.length > 1 ? ` part ${idx + 1}` : ""}:\n\n${chunk.trim()}` },
            { role: "assistant", content: chunk.trim().slice(0, 900) },
          ],
          source: path.relative(path.dirname(inputPath), file),
        });
      }
    }
    warnings.push("Docs were converted into starter summarization examples. For best results, replace assistant outputs with the exact behavior you want the model to learn.");
  } else if (/\.csv$/i.test(inputPath)) {
    format = "csv";
    const parsed = parseCsv(readFileSync(inputPath, "utf8"));
    if (parsed.length < 2) throw new Error("CSV needs a header row and at least one data row.");
    const headers = parsed[0]!.map((h) => h.trim());
    const userIdx = pickColumn(headers, ["user", "question", "prompt", "input", "query"]);
    const assistantIdx = pickColumn(headers, ["assistant", "answer", "completion", "output", "response"]);
    const systemIdx = pickColumn(headers, ["system", "instruction"]);
    const messagesIdx = pickColumn(headers, ["messages"]);
    for (const values of parsed.slice(1)) {
      if (messagesIdx >= 0) {
        try {
          const messages = JSON.parse(values[messagesIdx] ?? "");
          if (Array.isArray(messages)) rows.push({ messages });
          else warnings.push("Skipped a CSV row whose messages column was not a JSON array.");
        } catch {
          warnings.push("Skipped a CSV row whose messages column was not valid JSON.");
        }
        continue;
      }
      if (userIdx < 0 || assistantIdx < 0) throw new Error(`CSV needs columns like question/prompt/input and answer/response/output. Found: ${headers.join(", ")}`);
      const messages: Array<Record<string, string>> = [];
      const system = systemIdx >= 0 ? String(values[systemIdx] ?? "").trim() : "";
      const user = String(values[userIdx] ?? "").trim();
      const assistant = String(values[assistantIdx] ?? "").trim();
      if (!user || !assistant) continue;
      if (system) messages.push({ role: "system", content: system });
      messages.push({ role: "user", content: user });
      messages.push({ role: "assistant", content: assistant });
      rows.push({ messages });
    }
  } else if (/\.jsonl$/i.test(inputPath)) {
    format = "jsonl";
    for (const line of readFileSync(inputPath, "utf8").split(/\n/).filter((x) => x.trim())) {
      const converted = rowToTrainingExample(JSON.parse(line));
      if (converted) rows.push(converted);
      else warnings.push("Skipped a JSONL row that did not contain messages[] or recognizable prompt/response fields.");
    }
  } else if (/\.json$/i.test(inputPath)) {
    format = "json";
    const parsed = JSON.parse(readFileSync(inputPath, "utf8"));
    const items = Array.isArray(parsed) ? parsed : Array.isArray(parsed.examples) ? parsed.examples : Array.isArray(parsed.data) ? parsed.data : [];
    if (!Array.isArray(items) || items.length === 0) throw new Error("JSON should be an array, or contain examples[]/data[].");
    for (const item of items) {
      const converted = item && typeof item === "object" ? rowToTrainingExample(item as Record<string, unknown>) : undefined;
      if (converted) rows.push(converted);
      else warnings.push("Skipped a JSON item that did not contain messages[] or recognizable prompt/response fields.");
    }
  } else if (/\.(md|mdx|txt)$/i.test(inputPath)) {
    format = "document";
    const text = readFileSync(inputPath, "utf8").replace(/\s+/g, " ").trim();
    for (const chunk of (text.match(/.{1,3500}(?:\s|$)/g) ?? [text]).slice(0, 100)) {
      rows.push({ messages: [{ role: "user", content: `Summarize the useful facts from this document:\n\n${chunk.trim()}` }, { role: "assistant", content: chunk.trim().slice(0, 900) }] });
    }
    warnings.push("A raw document creates weak starter examples. Convert real question/answer or input/output pairs when possible.");
  } else {
    throw new Error("Unsupported input. Use CSV, JSON, JSONL, TXT/MD, or a directory of TXT/MD files.");
  }

  if (rows.length === 0) throw new Error("No usable training rows found.");
  await mkdir(path.dirname(outFile), { recursive: true });
  await writeFile(outFile, jsonl(rows));
  return {
    inputPath,
    outFile,
    format,
    rows: rows.length,
    warnings,
    samples: rows.slice(0, 3).map((row) => JSON.stringify(row).slice(0, 1000)),
  };
}

function recommendPlan(goal: string, dataRows?: number): string {
  const g = goal.toLowerCase();
  const structured = /json|extract|classif|label|schema|sql|regex|parse/.test(g);
  const reasoning = /math|reason|proof|solve|logic|agent|tool/.test(g);
  const writing = /write|tone|support|summar|rewrite|style|email|copy/.test(g);
  const model = DEFAULT_MODEL;
  const examples = dataRows ? `${dataRows} detected rows` : "unknown row count";
  const rowsAdvice = !dataRows ? "Start with 20–100 high-quality examples for a smoke test, then scale." : dataRows < 20 ? "Good for a smoke test only; add more examples before expecting durable quality gains." : dataRows < 200 ? "Enough for an early SFT run if examples are high quality." : "Enough to try a serious SFT run after baseline eval passes.";
  const task = structured ? "SFT with strict eval for valid outputs" : reasoning ? "SFT first; consider RL only after strong evals exist" : writing ? "SFT with before/after human or rubric eval" : "SFT golden path";
  return [
    `# Tinker recommendation`,
    `- Goal: ${goal || "not provided"}`,
    `- Data: ${examples}`,
    `- Recommended first model: \`${model}\``,
    `- First training method: ${task}`,
    `- Starter settings: \`max_steps=20\`, \`batch_size=8\`, \`lr=2e-4\`, 2-step smoke test before scale-up`,
    `- Data advice: ${rowsAdvice}`,
    "",
    "## Fastest path",
    "```text",
    "/tinker new data/train.jsonl --goal \"" + (goal || "what should improve") + "\" --model " + model,
    "/tinker doctor",
    "/tinker validate data/train.jsonl --model " + model,
    "/tinker eval baseline --model " + model + " --yes",
    "/tinker smoke train_sft.py --yes",
    "```",
  ].join("\n");
}

function templateBySlug(slug: string): ExampleTemplate | undefined {
  return EXAMPLE_TEMPLATES.find((t) => t.slug === slug);
}

async function writeExampleTemplate(cwd: string, slug: string, force = false): Promise<{ template: ExampleTemplate; files: string[] }> {
  const template = templateBySlug(slug);
  if (!template) throw new Error(`Unknown example ${slug}. Available: ${EXAMPLE_TEMPLATES.map((t) => t.slug).join(", ")}`);
  const files = [
    { rel: path.join("examples", slug, "train.jsonl"), content: jsonl(template.trainRows) },
    { rel: path.join("examples", slug, "eval.jsonl"), content: jsonl(template.evalRows) },
    { rel: path.join("examples", slug, "README.md"), content: `# ${template.title}\n\nGoal: ${template.goal}\n\nTry:\n\n\`\`\`text\n/tinker improve examples/${slug}/train.jsonl --goal "${template.goal}" --budget demo --force\n/tinker validate examples/${slug}/train.jsonl --quick\n\`\`\`\n` },
  ];
  const written: string[] = [];
  for (const file of files) {
    const target = path.join(cwd, file.rel);
    if (existsSync(target) && !force) continue;
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, file.content);
    written.push(file.rel);
  }
  return { template, files: written };
}

async function buildDoctorReport(cwd: string, dataFileArg?: string): Promise<string> {
  const wizard = await readWizardState(cwd);
  const dataFile = dataFileArg ? path.resolve(cwd, dataFileArg) : wizard?.dataFile;
  const model = wizard?.model ?? DEFAULT_MODEL;
  const checks: string[] = [];
  checks.push(process.env.TINKER_API_KEY ? "✅ TINKER_API_KEY is set" : "❌ TINKER_API_KEY is missing");
  checks.push((await commandExists("python3")) ? "✅ python3 found" : "❌ python3 not found");
  checks.push((await commandExists("uv")) ? "✅ uv found" : "⚠️ uv not found; pip fallback is okay");
  checks.push((await commandExists("tinker")) ? "✅ tinker CLI found" : "⚠️ tinker CLI not found");
  if (isInklingModel(model)) {
    try {
      const { stdout } = await execFile("python3", ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], { timeout: 20_000 });
      const [major, minor] = stdout.trim().split(".").map(Number);
      checks.push(major > 3 || (major === 3 && minor >= 11) ? `✅ Python ${stdout.trim()} supports Inkling` : `❌ Inkling requires Python 3.11+ (found ${stdout.trim()})`);
    } catch {}
  }
  const modules = ["tinker", "tinker_cookbook", "chz", "tml_renderers"];
  for (const mod of modules) {
    try {
      await execFile("python3", ["-c", `import ${mod}; print('ok')`], { cwd, timeout: 20_000 });
      checks.push(`✅ Python import ${mod}`);
    } catch {
      checks.push(`⚠️ Python cannot import ${mod}`);
    }
  }
  try {
    const { stdout } = await execFile("python3", ["-c", "import torch; print(torch.__version__)"], { timeout: 20_000 });
    const ver = stdout.trim();
    const [major, minor] = ver.split(".").map((p) => Number.parseInt(p, 10));
    checks.push(major > 2 || (major === 2 && minor >= 10) ? `✅ PyTorch ${ver} (>= 2.10)` : `❌ PyTorch ${ver} is too old; tml-renderers needs torch>=2.10`);
  } catch {
    checks.push("⚠️ Python cannot import torch; reinstall with `" + COOKBOOK_INSTALL + "`");
  }
  if (isInklingModel(model)) {
    try {
      const { stdout } = await execFile("python3", ["-c", "import importlib.metadata as m; print(m.version('tinker'))"], { timeout: 20_000 });
      const [major, minor] = stdout.trim().split(".").map(Number);
      checks.push(major > 0 || minor >= 23 ? `✅ Tinker SDK ${stdout.trim()} supports Inkling` : `❌ Inkling requires Tinker SDK 0.23+ (found ${stdout.trim()})`);
    } catch {}
  }
  if (isRetiredModel(model)) {
    checks.push(`⚠️ \`${model}\` is not on Tinker's current lineup. Check ${MODELS_DOCS_URL} and ${DEPRECATIONS_URL} before training.`);
  }
  checks.push(existsSync(path.join(cwd, "train_sft.py")) ? "✅ train_sft.py exists" : "⬜ train_sft.py missing; run /tinker new or /tinker init");
  checks.push(existsSync(path.join(cwd, "eval.py")) ? "✅ eval.py exists" : "⬜ eval.py missing; run /tinker eval init");
  if (dataFile) checks.push(existsSync(dataFile) ? `✅ data file exists: ${rel(cwd, dataFile)}` : `❌ data file missing: ${dataFile}`);
  else checks.push("⬜ no data file selected yet");
  if (existsSync(path.join(cwd, "train_sft.py"))) {
    try {
      await execFile("python3", ["-m", "py_compile", "train_sft.py"], { cwd, timeout: 20_000 });
      checks.push("✅ train_sft.py compiles");
    } catch (error: any) {
      checks.push(`❌ train_sft.py does not compile: ${error?.message ?? String(error)}`);
    }
  }
  if (existsSync(path.join(cwd, "eval.py"))) {
    try {
      await execFile("python3", ["-m", "py_compile", "eval.py"], { cwd, timeout: 20_000 });
      checks.push("✅ eval.py compiles");
    } catch (error: any) {
      checks.push(`❌ eval.py does not compile: ${error?.message ?? String(error)}`);
    }
  }
  if (isInklingModel(model)) {
    checks.push("ℹ️ Inkling family uses TMLv0 via tml-renderers (default cookbook dep) and SFT effort 0.9 unless you set otherwise");
    checks.push(model === INKLING_SMALL_MODEL || model.startsWith(INKLING_SMALL_MODEL)
      ? "ℹ️ Inkling-Small is the default: same renderer/effort interface as full Inkling, cheaper"
      : "ℹ️ Full Inkling is 975B/41B active. Prefer Inkling-Small unless you need the larger model");
    checks.push("ℹ️ Pin the same effort for training data, baseline eval, and checkpoint eval. Sweep with `/tinker inkling sweep`");
  }
  const next = wizard ? wizardSteps(cwd, wizard).find((step) => !step.done)?.nextCommand : "/tinker new";
  return [
    "# Tinker doctor",
    checks.map((c) => `- ${c}`).join("\n"),
    "",
    "## Next recommended command",
    "```text",
    next ?? "/tinker next",
    "```",
    "",
    `Install/upgrade Cookbook with \`${COOKBOOK_INSTALL}\`. Live model list: ${MODELS_DOCS_URL}`,
    "Research/debug methodology lives in the official Cookbook plugin: `/plugin marketplace add thinking-machines-lab/tinker-cookbook`.",
    "If anything is confusing, run `/skill:tinker-debug` with this report.",
  ].join("\n");
}


type ImproveBudget = "demo" | "smoke" | "small" | "real";

function parseImproveBudget(raw: unknown): ImproveBudget {
  const budget = String(raw ?? "demo").toLowerCase();
  if (["demo", "smoke", "small", "real"].includes(budget)) return budget as ImproveBudget;
  throw new Error("--budget must be one of: demo, smoke, small, real");
}

function budgetMaxSteps(budget: ImproveBudget, override?: unknown): number {
  if (override !== undefined) return Number(override);
  if (budget === "smoke") return 2;
  if (budget === "small") return 20;
  if (budget === "real") return 100;
  return 0;
}

const INKLING_EFFORT_PRESETS: Record<string, number> = {
  none: 0,
  minimal: 0.1,
  low: 0.2,
  medium: 0.7,
  high: 0.9,
  xhigh: 0.99,
};

function parseInklingEfforts(value: unknown): number[] {
  const raw = String(value ?? "low,medium,high,xhigh");
  const efforts = raw.split(",").map((part) => {
    const text = part.trim().toLowerCase();
    return text in INKLING_EFFORT_PRESETS ? INKLING_EFFORT_PRESETS[text]! : Number(text);
  });
  if (efforts.length === 0 || efforts.some((effort) => !Number.isFinite(effort) || effort < 0 || effort >= 1)) {
    throw new Error("--efforts must be comma-separated presets or floats in [0, 1), e.g. low,medium,high,xhigh or 0.2,0.7,0.9,0.99");
  }
  return efforts;
}

function managedRunPlan(budget: ImproveBudget): string {
  const lines = [
    "# Managed fine-tuning operator",
    `- Budget: \`${budget}\``,
    "- Principle: never scale until data, eval, and smoke test are sane.",
    "",
    "## Loop",
    "1. Prepare/locate data",
    "2. Scaffold editable Tinker Cookbook files",
    "3. Run project doctor",
    "4. Validate data/rendering",
    "5. Ensure eval exists",
    "6. Run baseline eval before training",
    "7. Run 2-step smoke training",
    "8. Scale training only with confirmation",
    "9. Evaluate checkpoint on the same eval",
    "10. Compare wins/regressions and register the checkpoint for chat",
  ];
  if (budget === "demo") lines.push("", "Demo budget stops before real Tinker API usage.");
  if (budget === "smoke") lines.push("", "Smoke budget runs baseline eval + 2-step smoke training, then stops.");
  if (budget === "small") lines.push("", "Small budget runs a short training job after smoke passes.");
  if (budget === "real") lines.push("", "Real budget still asks for confirmation before scale-up unless `--yes` is passed.");
  return lines.join("\n");
}

function suggestDataImprovementsFromEval(filePath?: string): string {
  if (!filePath || !existsSync(filePath)) {
    return [
      "## What data to add next",
      "After you have eval failures, add training examples that look like the failures — same input style, corrected ideal output, and edge cases.",
      "Prefer 20 excellent examples over 200 noisy examples.",
    ].join("\n");
  }
  try {
    const result = readEvalSummary(filePath);
    const failures = (result.results ?? []).filter((r) => !r.correct).slice(0, 8);
    if (failures.length === 0) {
      return "## What data to add next\nNo eval failures found in the sample. Add harder held-out eval cases before scaling further.";
    }
    return [
      "## What data to add next",
      "Add corrected training examples similar to these failures:",
      ...failures.map((f) => `- Eval #${f.index}: expected ${JSON.stringify(f.expected)} but got ${JSON.stringify(String(f.output).slice(0, 180))}`),
      "",
      "For each failure, add 2–5 training rows that show the exact desired behavior and one eval row that stays held out.",
    ].join("\n");
  } catch {
    return "## What data to add next\nCould not read eval failures. Add examples for the cases your current model gets wrong, then re-run baseline/checkpoint comparison.";
  }
}

function sanitizeName(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "tinker-model";
}

function makeExportMarkdown(options: { checkpoint: string; alias: string; baseModel?: string }): string {
  const { checkpoint, alias, baseModel } = options;
  const kind = exportKindFor(baseModel);
  const model = baseModel ?? "<base-model>";
  if (kind === "tinker-only") {
    return `# Export ${alias}

\`${model}\` should stay on Tinker's sampling endpoint. Cookbook has no merge/PEFT path for Inkling or Inkling-Small.

Use the Tinker API snippets in this folder (\`python_client.py\`, \`fastapi_app.py\`). Do not run \`export.py\`.
`;
  }
  const peftBlock = kind === "peft"
    ? `## PEFT adapter (vLLM / SGLang)

\`\`\`bash
${COOKBOOK_INSTALL}
python3 export.py --peft
# then, for example:
# vllm serve ${model} --lora-modules ${alias}=./peft_adapter
\`\`\`

`
    : `## PEFT adapter

Not recommended for \`${model}\`. Cookbook can convert some of these families, but vLLM/SGLang LoRA serving is blocked or unverified. Merge instead.

`;
  const modalBlock = kind === "unknown"
    ? ""
    : `## Optional: serve on Modal

Cookbook's Modal recipe merges the checkpoint and serves it with SGLang. This is a generated command, not a Pi runtime.

\`\`\`bash
pip install "tinker-cookbook[modal]"
modal setup
export TINKER_API_KEY=...
# optional, gated base models only:
export HF_TOKEN=...

modal run -m tinker_cookbook.inference.modal.prepare \\
  --tinker-path '${checkpoint}' \\
  --base-model ${model} --name ${alias}

FINETUNE=${alias} MODEL=${model} modal deploy -m tinker_cookbook.inference.modal.serve
\`\`\`

Compare Tinker sampling vs the Modal endpoint:

\`\`\`bash
FINETUNE=${alias} MODEL=${model} \\
  modal run -m tinker_cookbook.inference.modal.compare \\
  --tinker-path '${checkpoint}' --url $URL
\`\`\`
`;
  return `# Export ${alias}

Download the Tinker sampler checkpoint and turn it into local weights with Cookbook's \`tinker_cookbook.weights\` helpers. pi-tinker does not run a serving stack.

- Checkpoint: \`${checkpoint}\`
- Base model: \`${model}\`

Install Cookbook, then:

\`\`\`bash
${COOKBOOK_INSTALL}
python3 export.py --merge
\`\`\`

${peftBlock}## Merge to a full Hugging Face model

\`\`\`bash
python3 export.py --merge
# optional:
python3 export.py --merge --publish your-hf-user/${alias}
\`\`\`

${modalBlock}Live Cookbook weight docs: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/weights/README.md
`;
}

function makeExportScript(options: { checkpoint: string; alias: string; baseModel?: string }): string {
  const { checkpoint, alias, baseModel } = options;
  const kind = exportKindFor(baseModel);
  const model = baseModel ?? "";
  if (kind === "tinker-only" || !model) {
    return `"""Inkling checkpoints stay on Tinker. Do not export."""
import sys
raise SystemExit(
    "${model || "This Inkling checkpoint"} should be served through Tinker's API. "
    "Use python_client.py / fastapi_app.py in this folder."
)
`;
  }
  return `"""Export a Tinker sampler checkpoint with tinker_cookbook.weights.

Usage:
  python3 export.py --merge
  python3 export.py --peft
  python3 export.py --merge --publish user/${alias}
"""
from __future__ import annotations

import argparse
from tinker_cookbook import weights

CHECKPOINT = ${JSON.stringify(checkpoint)}
BASE_MODEL = ${JSON.stringify(model)}
ALIAS = ${JSON.stringify(alias)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Tinker sampler checkpoint.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--merge", action="store_true", help="Merge LoRA into a full HF model (default)")
    mode.add_argument("--peft", action="store_true", help="Write a PEFT adapter for vLLM/SGLang")
    parser.add_argument("--publish", help="Hugging Face repo id, e.g. user/${alias}")
    args = parser.parse_args()
    peft = args.peft
    adapter = weights.download(tinker_path=CHECKPOINT, output_dir="./adapter")
    if peft:
        weights.build_lora_adapter(base_model=BASE_MODEL, adapter_path=adapter, output_path="./peft_adapter")
        print(f"PEFT adapter written to ./peft_adapter")
        print(f"Example: vllm serve {BASE_MODEL} --lora-modules {ALIAS}=./peft_adapter")
        return
    weights.build_hf_model(base_model=BASE_MODEL, adapter_path=adapter, output_path="./model")
    print("Merged model written to ./model")
    if args.publish:
        weights.publish_to_hf_hub(model_path="./model", repo_id=args.publish, private=True)
        print(f"Published to {args.publish}")


if __name__ == "__main__":
    main()
`;
}

function makeServingMarkdown(options: { alias: string; baseModel?: string }): string {
  const { alias, baseModel } = options;
  const model = baseModel ?? "unknown base model";
  const kind = exportKindFor(baseModel);
  const preset = htdymPreset(baseModel);
  if (kind === "tinker-only") {
    return `# Serving ${alias}

Stay on Tinker. \`${model}\` is not in a self-host catalog we can price, and Cookbook does not export Inkling weights for vLLM/SGLang.

Use the OpenAI-compatible snippets in this folder against Tinker's endpoint. That is the product path for Inkling and Inkling-Small.
`;
  }
  const htdymBlock = preset
    ? `## Self-host cost (HTDYM)

This base model maps to the HTDYM preset **${preset}**.

HTDYM is a static inference cost estimator, not a deployer. Open it with these four inputs:

1. Model: \`${preset}\`
2. Workload: start at prefill 4096 / decode 1024 (edit if your eval is different)
3. Hardware: H100 8x, H200, B200, or 4090 depending on what you can rent
4. Target: e.g. 20 tok/s/user

Explorer: ${HTDYM_URL}

Use it to decide whether self-host is even a conversation versus staying on Tinker. pi-tinker does not run the estimator or stand up a cluster.
`
    : `## Self-host cost (HTDYM)

\`${model}\` is not in the current HTDYM preset list, so we cannot quote tok/s or relative $ here.

Explorer: ${HTDYM_URL}

If you add the architecture later, compare Tinker sampling vs a box you actually have. Do not invent a GPU table.
`;
  return `# Serving ${alias}

Default: keep sampling on Tinker until eval is green.

- Checkpoint serving via Tinker: \`python_client.py\` / \`fastapi_app.py\`
- Export weights: see \`EXPORT.md\` and \`export.py\`
- Base model: \`${model}\`

${htdymBlock}Llama 3.x is not a Tinker live-lineup default anymore. Do not plan a self-host one-pager around a retired Tinker model without checking ${MODELS_DOCS_URL}.
`;
}

function makeDeployFiles(options: { checkpoint: string; alias: string; outDir: string; baseModel?: string }) {
  const model = options.checkpoint;
  const alias = sanitizeName(options.alias);
  const baseModel = options.baseModel;
  return [
    {
      rel: "README.md",
      content: `# Deploy ${alias}\n\nThis folder was generated by \`/tinker deploy\`.\n\n1. **Tinker API** — copy-paste clients against Tinker's OpenAI-compatible endpoint (good for Inkling and for inspection).\n2. **Export** — \`EXPORT.md\` + \`export.py\` use Cookbook \`weights.*\` when the base model can leave Tinker.\n3. **Serving decision** — \`SERVING.md\` says stay-on-Tinker vs self-host, with an HTDYM pointer when the architecture is in that catalog.\n\n## Environment\n\n\`\`\`bash\nexport TINKER_API_KEY=...\nexport TINKER_MODEL='${model}'\nexport TINKER_BASE_URL='${TINKER_OAI_BASE_URL}'\n\`\`\`\n\n## Python\n\n\`\`\`bash\npython3 -m pip install openai\npython3 python_client.py\n\`\`\`\n\n## Node\n\n\`\`\`bash\nnpm install openai\nnode node_client.mjs\n\`\`\`\n\n## FastAPI wrapper\n\n\`\`\`bash\npython3 -m pip install fastapi uvicorn openai\nuvicorn fastapi_app:app --reload\n\`\`\`\n\nRead \`EXPORT.md\` before standing up GPUs. Inkling stays on Tinker.\n`,
    },
    { rel: "EXPORT.md", content: makeExportMarkdown({ checkpoint: options.checkpoint, alias, baseModel }) },
    { rel: "export.py", content: makeExportScript({ checkpoint: options.checkpoint, alias, baseModel }) },
    { rel: "SERVING.md", content: makeServingMarkdown({ alias, baseModel }) },
    {
      rel: ".env.example",
      content: `TINKER_API_KEY=your-api-key\nTINKER_BASE_URL=${TINKER_OAI_BASE_URL}\nTINKER_MODEL=${model}\n`,
    },
    {
      rel: "python_client.py",
      content: `import os\nfrom openai import OpenAI\n\nclient = OpenAI(\n    api_key=os.environ["TINKER_API_KEY"],\n    base_url=os.environ.get("TINKER_BASE_URL", "${TINKER_OAI_BASE_URL}"),\n)\n\nmodel = os.environ.get("TINKER_MODEL", "${model}")\n\nresponse = client.chat.completions.create(\n    model=model,\n    messages=[{"role": "user", "content": "Say what you improved in one sentence."}],\n    max_tokens=120,\n    temperature=0.2,\n)\n\nprint(response.choices[0].message.content)\n`,
    },
    {
      rel: "node_client.mjs",
      content: `import OpenAI from "openai";\n\nconst client = new OpenAI({\n  apiKey: process.env.TINKER_API_KEY,\n  baseURL: process.env.TINKER_BASE_URL ?? "${TINKER_OAI_BASE_URL}",\n});\n\nconst model = process.env.TINKER_MODEL ?? "${model}";\n\nconst response = await client.chat.completions.create({\n  model,\n  messages: [{ role: "user", content: "Say what you improved in one sentence." }],\n  max_tokens: 120,\n  temperature: 0.2,\n});\n\nconsole.log(response.choices[0].message.content);\n`,
    },
    {
      rel: "fastapi_app.py",
      content: `import os\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\nfrom openai import OpenAI\n\napp = FastAPI(title="${alias} Tinker checkpoint")\nclient = OpenAI(\n    api_key=os.environ["TINKER_API_KEY"],\n    base_url=os.environ.get("TINKER_BASE_URL", "${TINKER_OAI_BASE_URL}"),\n)\nMODEL = os.environ.get("TINKER_MODEL", "${model}")\n\nclass ChatRequest(BaseModel):\n    message: str\n    max_tokens: int = 256\n    temperature: float = 0.2\n\n@app.post("/chat")\ndef chat(req: ChatRequest):\n    response = client.chat.completions.create(\n        model=MODEL,\n        messages=[{"role": "user", "content": req.message}],\n        max_tokens=req.max_tokens,\n        temperature=req.temperature,\n    )\n    return {"output": response.choices[0].message.content}\n`,
    },
  ];
}

async function writeDeployFiles(cwd: string, options: { checkpoint: string; alias: string; outDir: string; force?: boolean; baseModel?: string }): Promise<string[]> {
  const dir = path.resolve(cwd, options.outDir);
  const written: string[] = [];
  for (const file of makeDeployFiles({ checkpoint: options.checkpoint, alias: options.alias, outDir: options.outDir, baseModel: options.baseModel })) {
    const target = path.join(dir, file.rel);
    if (existsSync(target) && !options.force) continue;
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, file.content);
    written.push(path.relative(cwd, target));
  }
  return written;
}

function resolveCheckpointRef(input: string | undefined, state: TinkerState, wizard?: WizardState): { checkpoint?: string; alias: string; baseModel?: string } {
  if (!input || input === "latest") {
    const latest = wizard ? undefined : state.checkpoints.slice(-1)[0];
    return {
      checkpoint: wizard?.approvedCheckpointPath ?? latest?.id,
      alias: wizard?.registeredModel ?? latest?.name ?? "my-finetune",
      baseModel: wizard?.model ?? latest?.baseModel,
    };
  }
  const byAlias = state.checkpoints.find((m) => m.name === input || m.id === input);
  if (byAlias) return { checkpoint: byAlias.id, alias: byAlias.name, baseModel: byAlias.baseModel ?? wizard?.model };
  if (input.startsWith("tinker://")) return { checkpoint: input, alias: "my-finetune", baseModel: wizard?.model };
  return { checkpoint: undefined, alias: input, baseModel: wizard?.model };
}

function makeExactEvalScript() {
  return `"""Simple editable eval for Tinker checkpoints.

Input JSONL format:
  {"messages": [{"role": "user", "content": "..."}], "expected": "...", "match": "contains"}

Match modes:
  - exact: normalized output equals normalized expected
  - contains: normalized expected appears in normalized output
  - prefix: normalized output starts with normalized expected

Examples:
  python eval.py --base-model thinkingmachines/Inkling-Small --effort 0.9 --data data/eval.jsonl --out eval_results/baseline.json
  python eval.py --model-path 'tinker://.../sampler_weights/...' --renderer-model thinkingmachines/Inkling-Small --effort 0.9 --data data/eval.jsonl --out eval_results/step-20.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer, get_text_content


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\\s+", " ", text)
    text = text.strip(" \\t\\n\\r.,;:!?")
    return text


def score_output(output: str, expected: str, match: str) -> bool:
    out = normalize(output)
    exp = normalize(expected)
    if match == "exact":
        return out == exp
    if match == "prefix":
        return out.startswith(exp)
    if match == "contains":
        return exp in out
    raise ValueError(f"unknown match mode: {match!r}")


def load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list):
                raise ValueError(f"line {line_number}: missing messages[]")
            if "expected" not in row:
                raise ValueError(f"line {line_number}: missing expected")
            rows.append(row)
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base model or Tinker sampler checkpoint.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--base-model", help="Tinker base model id for baseline evaluation")
    target.add_argument("--model-path", help="Tinker sampler checkpoint path")
    parser.add_argument("--renderer-model", help="Model id used to choose renderer/tokenizer; defaults to --base-model or sampler base model")
    parser.add_argument("--data", default="data/eval.jsonl")
    parser.add_argument("--out", default="eval_results/result.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--effort", type=float, default=0.9, help="Inkling effort in [0, 1); ignored by other renderers")
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    args = parser.parse_args()

    if not 0.0 <= args.effort < 1.0:
        raise SystemExit("--effort must be in [0, 1)")

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No eval rows found.")

    service = tinker.ServiceClient()
    if args.model_path:
        sampling_client = await service.create_sampling_client_async(model_path=args.model_path)
        target_name = args.model_path
        renderer_model = args.renderer_model or sampling_client.get_base_model()
    else:
        sampling_client = await service.create_sampling_client_async(base_model=args.base_model)
        target_name = args.base_model
        renderer_model = args.renderer_model or args.base_model

    tokenizer = sampling_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(renderer_model)
    renderer = get_renderer(renderer_name, tokenizer)
    stop = renderer.get_stop_sequences()

    results: list[dict[str, Any]] = []
    correct = 0

    for index, row in enumerate(rows, start=1):
        messages = row["messages"]
        expected = str(row["expected"])
        match = str(row.get("match", "contains"))
        prompt_kwargs = {"role": "assistant"}
        if renderer_model.startswith("thinkingmachines/Inkling"):
            prompt_kwargs["effort"] = args.effort
        prompt = renderer.build_generation_prompt(messages, **prompt_kwargs)
        response = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stop=stop,
            ),
        )
        sequence = response.sequences[0] if hasattr(response, "sequences") else response.samples[0]
        parsed_message, termination = renderer.parse_response(sequence.tokens)
        output = get_text_content(parsed_message)
        is_correct = score_output(output, expected, match)
        correct += int(is_correct)
        results.append({
            "index": index,
            "correct": is_correct,
            "match": match,
            "expected": expected,
            "output": output,
            "termination": getattr(termination, "value", str(termination)),
            "messages": messages,
        })
        print(f"[{index}/{len(rows)}] {'✓' if is_correct else '✗'} expected={expected!r} output={output[:120]!r}")

    summary = {
        "target": target_name,
        "renderer_model": renderer_model,
        "renderer": renderer_name,
        "effort": args.effort if renderer_model.startswith("thinkingmachines/Inkling") else None,
        "data": args.data,
        "num_examples": len(rows),
        "num_correct": correct,
        "accuracy": correct / len(rows),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n")
    print(f"accuracy={summary['accuracy']:.3f} ({correct}/{len(rows)})")
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
`;
}

type EvalSummary = {
  target?: string;
  renderer_model?: string;
  renderer?: string;
  effort?: number | null;
  data?: string;
  num_examples?: number;
  num_correct?: number;
  accuracy?: number;
  results?: Array<{ index: number; correct: boolean; expected: string; output: string }>;
};

function readEvalSummary(filePath: string): EvalSummary {
  return JSON.parse(readFileSync(filePath, "utf8")) as EvalSummary;
}

function formatEvalSummary(filePath: string): string {
  const result = readEvalSummary(filePath);
  const accuracy = typeof result.accuracy === "number" ? `${(result.accuracy * 100).toFixed(1)}%` : "n/a";
  const examples = result.num_examples ?? result.results?.length ?? 0;
  const correct = result.num_correct ?? result.results?.filter((r) => r.correct).length ?? 0;
  const failures = (result.results ?? []).filter((r) => !r.correct).slice(0, 5);
  return [
    `- File: \`${filePath}\``,
    `- Target: \`${result.target ?? "unknown"}\``,
    `- Renderer: \`${result.renderer ?? "unknown"}\`${result.renderer_model ? ` for \`${result.renderer_model}\`` : ""}`,
    result.effort !== null && result.effort !== undefined ? `- Inkling effort: ${result.effort}` : "",
    `- Accuracy: ${accuracy} (${correct}/${examples})`,
    failures.length ? `\n## Sample failures\n${failures.map((f) => `### #${f.index}\n- Expected: ${JSON.stringify(f.expected)}\n- Output: ${JSON.stringify(String(f.output).slice(0, 500))}`).join("\n\n")}` : "\nNo failures recorded.",
  ].join("\n");
}

function compareEvalSummaries(baselinePath: string, candidatePath: string): string {
  const baseline = readEvalSummary(baselinePath);
  const candidate = readEvalSummary(candidatePath);
  const bAcc = baseline.accuracy ?? 0;
  const cAcc = candidate.accuracy ?? 0;
  const delta = cAcc - bAcc;
  const bResults = new Map((baseline.results ?? []).map((r) => [r.index, r]));
  const effortMismatch = baseline.effort !== null && baseline.effort !== undefined && candidate.effort !== null && candidate.effort !== undefined && baseline.effort !== candidate.effort;
  const wins: string[] = [];
  const regressions: string[] = [];
  for (const r of candidate.results ?? []) {
    const before = bResults.get(r.index);
    if (!before) continue;
    if (!before.correct && r.correct) wins.push(`#${r.index}: ${JSON.stringify(String(r.output).slice(0, 220))}`);
    if (before.correct && !r.correct) regressions.push(`#${r.index}: expected ${JSON.stringify(r.expected)}, got ${JSON.stringify(String(r.output).slice(0, 220))}`);
  }
  return [
    `# Eval comparison`,
    `- Baseline: \`${baselinePath}\` — ${(bAcc * 100).toFixed(1)}% (${baseline.num_correct}/${baseline.num_examples})`,
    `- Candidate: \`${candidatePath}\` — ${(cAcc * 100).toFixed(1)}% (${candidate.num_correct}/${candidate.num_examples})`,
    `- Delta: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} points`,
    effortMismatch ? `- ⚠️ Effort mismatch: baseline=${baseline.effort}, candidate=${candidate.effort}. Re-run at identical effort before attributing the delta to fine-tuning.` : (baseline.effort !== null && baseline.effort !== undefined ? `- Inkling effort: ${baseline.effort} (matched)` : ""),
    wins.length ? `\n## Wins\n${wins.slice(0, 10).map((x) => `- ${x}`).join("\n")}` : "\n## Wins\nNone recorded.",
    regressions.length ? `\n## Regressions\n${regressions.slice(0, 10).map((x) => `- ${x}`).join("\n")}` : "\n## Regressions\nNone recorded.",
  ].join("\n");
}

type WizardState = {
  version: 1;
  createdAt: number;
  updatedAt: number;
  sourceDataFile?: string;
  sourceDataFileInput?: string;
  sourceDataHash?: string;
  dataFile?: string;
  dataFileInput?: string;
  trainingDataHash?: string;
  evalDataHash?: string;
  evalGenerated?: boolean;
  evalReviewedHash?: string;
  model?: string;
  metric?: string;
  logPath?: string;
  validationAt?: number;
  baselineResult?: string;
  baselineFingerprint?: string;
  smokeAt?: number;
  smokeLogDir?: string;
  checkpointPath?: string;
  candidateCheckpointPath?: string;
  candidateResult?: string;
  candidateFingerprint?: string;
  candidateDecision?: "approved" | "rejected";
  approvedCheckpointPath?: string;
  approvedResult?: string;
  approvedFingerprint?: string;
  approvedAt?: number;
  registeredModel?: string;
  effort?: number;
  effortSweepAt?: number;
  effortSweepFingerprint?: string;
};

function wizardRunFingerprint(cwd: string, state: WizardState): string {
  return provenanceFingerprint({
    data: hashFile(state.dataFile),
    effort: state.effort,
    eval: hashFile(path.join(cwd, "data", "eval.jsonl")),
    evalScript: hashFile(path.join(cwd, "eval.py")),
    model: state.model,
  });
}

function evalBeatBaseline(baselinePath: string, candidatePath: string): {
  ok: boolean;
  delta: number;
  baseline: number;
  candidate: number;
} {
  const baseline = readEvalSummary(baselinePath).accuracy ?? 0;
  const candidate = readEvalSummary(candidatePath).accuracy ?? 0;
  return { ok: candidate > baseline, delta: candidate - baseline, baseline, candidate };
}

function nextImproveCommand(cwd: string, state: WizardState): string {
  const data = state.sourceDataFileInput || state.dataFileInput || (state.sourceDataFile ? rel(cwd, state.sourceDataFile) : state.dataFile ? rel(cwd, state.dataFile) : "data/train.jsonl");
  const model = state.model ?? DEFAULT_MODEL;
  const goal = state.metric ?? "what should improve";
  const effort = isInklingModel(model) && state.effort !== undefined ? ` --effort ${state.effort}` : "";
  const review = state.evalGenerated && state.evalReviewedHash !== state.evalDataHash ? " --eval-reviewed" : "";
  const flags = ` --goal ${JSON.stringify(goal)} --model ${model}${effort}${review}`;
  if (!state.dataFile || !existsSync(state.dataFile)) {
    return `/tinker improve ${data} --budget demo${flags}`;
  }
  if (!state.smokeAt) {
    return `/tinker improve ${data} --budget smoke --yes${flags}`;
  }
  if (state.candidateDecision === "rejected" && state.baselineResult && state.candidateResult) {
    try {
      const cmp = evalBeatBaseline(state.baselineResult, state.candidateResult);
      return `/tinker improve ${data} --budget smoke --yes${flags}\n# checkpoint was rejected (${(cmp.candidate * 100).toFixed(1)}% vs ${(cmp.baseline * 100).toFixed(1)}%). Add eval-like examples, then re-run smoke.`;
    } catch {
      return `/tinker improve ${data} --budget smoke --yes${flags}`;
    }
  }
  if (state.candidateDecision === "approved" && state.approvedCheckpointPath && state.approvedCheckpointPath === state.candidateCheckpointPath) {
    return `/tinker deploy ${state.registeredModel ?? "latest"}`;
  }
  return `/tinker improve ${data} --budget small --yes${flags}`;
}

function wizardDir(cwd: string) {
  return path.join(cwd, ".tinker-pi");
}

function wizardStatePath(cwd: string) {
  return path.join(wizardDir(cwd), "state.json");
}

async function readWizardState(cwd: string): Promise<WizardState | undefined> {
  try {
    return JSON.parse(await readFile(wizardStatePath(cwd), "utf8")) as WizardState;
  } catch {
    return undefined;
  }
}

async function writeWizardState(cwd: string, state: WizardState): Promise<void> {
  await mkdir(wizardDir(cwd), { recursive: true });
  await writeFile(wizardStatePath(cwd), JSON.stringify({ ...state, updatedAt: Date.now() }, null, 2) + "\n");
}

async function patchWizardState(cwd: string, patch: Partial<WizardState>): Promise<void> {
  const existing = await readWizardState(cwd);
  if (!existing) return;
  await writeWizardState(cwd, { ...existing, ...patch, version: 1 });
}

function rel(cwd: string, filePath?: string) {
  if (!filePath) return "";
  const r = path.relative(cwd, filePath);
  return r && !r.startsWith("..") ? r : filePath;
}

type WizardStep = {
  key: string;
  label: string;
  done: boolean;
  detail: string;
  nextCommand?: string;
  apiUsage?: boolean;
};

function findLatestCandidateEval(cwd: string): string | undefined {
  const dir = path.join(cwd, "eval_results");
  if (!existsSync(dir)) return undefined;
  try {
    const candidates = readdirSync(dir)
      .filter((name) => name.endsWith(".json") && name !== "baseline.json")
      .map((name) => path.join(dir, name))
      .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
    return candidates[0];
  } catch {
    return undefined;
  }
}

function firstSamplerCheckpoint(logDir?: string): string | undefined {
  if (!logDir || !existsSync(path.join(logDir, "checkpoints.jsonl"))) return undefined;
  try {
    return readCheckpoints(logDir).filter((c) => c.sampler_path).slice(-1)[0]?.sampler_path;
  } catch {
    return undefined;
  }
}

function wizardSteps(cwd: string, state: WizardState): WizardStep[] {
  const dataFile = state.dataFile;
  const logDir = state.logPath ? path.resolve(cwd, state.logPath) : undefined;
  const baseline = state.baselineResult ?? (existsSync(path.join(cwd, "eval_results", "baseline.json")) ? path.join(cwd, "eval_results", "baseline.json") : undefined);
  const checkpoint = state.checkpointPath ?? firstSamplerCheckpoint(logDir);
  const candidate = state.candidateResult ?? findLatestCandidateEval(cwd);
  const registered = state.registeredModel || (checkpoint ? undefined : state.registeredModel);

  return [
    {
      key: "env",
      label: "Environment ready",
      done: Boolean(process.env.TINKER_API_KEY) && existsSync(process.execPath),
      detail: process.env.TINKER_API_KEY ? "TINKER_API_KEY is set" : "Set TINKER_API_KEY first",
      nextCommand: "/tinker setup",
    },
    {
      key: "data",
      label: "Training data selected",
      done: Boolean(dataFile && existsSync(dataFile)),
      detail: dataFile ? rel(cwd, dataFile) : "No JSONL selected",
      nextCommand: "/tinker start path/to/train.jsonl",
    },
    {
      key: "files",
      label: "Project files created",
      done: existsSync(path.join(cwd, "train_sft.py")) && existsSync(path.join(cwd, "eval.py")) && existsSync(path.join(cwd, "data", "eval.jsonl")),
      detail: "train_sft.py, eval.py, data/eval.jsonl",
      nextCommand: "/tinker next --create-files",
    },
    {
      key: "validate",
      label: "Data validated",
      done: Boolean(state.validationAt),
      detail: state.validationAt ? new Date(state.validationAt).toLocaleString() : "Need readiness report",
      nextCommand: dataFile ? `/tinker validate ${rel(cwd, dataFile)} --model ${state.model ?? DEFAULT_MODEL}` : undefined,
    },
    {
      key: "baseline",
      label: "Baseline eval run",
      done: Boolean(baseline && existsSync(baseline)),
      detail: baseline ? rel(cwd, baseline) : "Measure base model before training",
      nextCommand: `/tinker eval baseline --model ${state.model ?? DEFAULT_MODEL}${isInklingModel(state.model ?? DEFAULT_MODEL) ? " --effort 0.9" : ""} --yes`,
      apiUsage: true,
    },
    {
      key: "smoke",
      label: "2-step smoke test run",
      done: Boolean(state.smokeAt) || Boolean(logDir && existsSync(path.join(logDir, "metrics.jsonl"))),
      detail: state.smokeAt ? new Date(state.smokeAt).toLocaleString() : "Run tiny training test",
      nextCommand: "/tinker smoke train_sft.py --yes",
      apiUsage: true,
    },
    {
      key: "train",
      label: "Training/checkpoint available",
      done: Boolean(checkpoint),
      detail: checkpoint ? checkpoint : "Run more steps, then save/check checkpoints",
      nextCommand: state.logPath ? `/tinker monitor ${state.logPath}` : "python train_sft.py max_steps=100",
      apiUsage: true,
    },
    {
      key: "checkpoint-eval",
      label: "Checkpoint eval run",
      done: Boolean(candidate && existsSync(candidate)),
      detail: candidate ? rel(cwd, candidate) : "Evaluate trained checkpoint on same eval set",
      nextCommand: checkpoint ? `/tinker eval checkpoint ${checkpoint} --model ${state.model ?? DEFAULT_MODEL}${isInklingModel(state.model ?? DEFAULT_MODEL) ? " --effort 0.9" : ""} --yes` : undefined,
      apiUsage: true,
    },
    {
      key: "compare",
      label: "Before/after compared",
      done: Boolean(baseline && candidate && existsSync(baseline) && existsSync(candidate)),
      detail: baseline && candidate ? `${rel(cwd, baseline)} vs ${rel(cwd, candidate)}` : "Compare baseline and checkpoint evals",
      nextCommand: baseline && candidate ? `/tinker eval compare ${rel(cwd, baseline)} ${rel(cwd, candidate)}` : undefined,
    },
    {
      key: "chat",
      label: "Checkpoint registered for chat",
      done: Boolean(registered),
      detail: registered ? registered : "Select checkpoint as a Pi model",
      nextCommand: checkpoint ? `/tinker use ${checkpoint} my-finetune` : undefined,
    },
  ];
}

function renderWizard(cwd: string, state: WizardState): string {
  const steps = wizardSteps(cwd, state);
  const next = steps.find((step) => !step.done);
  const progress = `${steps.filter((s) => s.done).length}/${steps.length}`;
  return [
    `# Fine-tune wizard`,
    `- Progress: ${progress}`,
    `- Model: \`${state.model ?? DEFAULT_MODEL}\``,
    state.dataFile ? `- Training data: \`${rel(cwd, state.dataFile)}\`` : "- Training data: not selected",
    state.metric ? `- Success metric: ${state.metric}` : "- Success metric: not defined",
    "",
    ...steps.map((step) => `${step.done ? "✅" : "⬜"} **${step.label}** — ${step.detail}`),
    "",
    next
      ? `## Next step\n${next.apiUsage ? "⚠️ This step may use the Tinker API.\n\n" : ""}Run this one command:\n\n\`\`\`text\n${nextImproveCommand(cwd, state)}\n\`\`\``
      : `## Complete\nYou have a validated dataset, before/after evals, and a checkpoint ready to chat with in Pi.\n\n\`\`\`text\n${nextImproveCommand(cwd, state)}\n\`\`\``,
    "",
    "Use `/tinker next` any time to see the next recommended action. The front door is `/tinker improve`.",
  ].join("\n");
}

async function createWizardFiles(cwd: string, state: WizardState, force = false): Promise<string[]> {
  if (!state.dataFile) throw new Error("Wizard has no dataFile. Run `/tinker improve data.csv --budget demo` first.");
  const model = state.model ?? DEFAULT_MODEL;
  const logPath = state.logPath ?? `logs/sft-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`;
  const files = [
    { rel: "README.md", content: makeProjectReadme({ model, dataFile: state.dataFile, logPath, successMetric: state.metric ?? "Define before scaling." }) },
    { rel: "train_sft.py", content: makeSftScript({ dataFile: state.dataFile, model, logPath, maxSteps: "20", batchSize: String(Math.min(8, Math.max(1, readJsonl(state.dataFile).length))), learningRate: "2e-4", testSize: "0", maxLength: "32768" }) },
    { rel: "eval_checkpoint.py", content: makeEvalScript() },
    { rel: "eval.py", content: makeExactEvalScript() },
    { rel: "tinker.yaml", content: `task: sft\ndata: ${state.dataFile}\nmodel: ${model}\nlog_path: ${logPath}\nmax_steps: 20\nbatch_size: 8\nlearning_rate: 2e-4\nsuccess_metric: ${state.metric ?? ""}\n${isInklingModel(model) ? `effort: ${state.effort ?? 0.9}\n` : ""}` },
    { rel: path.join("notes", "plan.md"), content: `# Fine-tuning plan\n\n## Goal\n\n${state.metric ?? "Define what should improve."}\n\n## Next\n\n\`/tinker next\` prints one command. The front door is \`/tinker improve\`.\n` },
  ];
  const written: string[] = [];
  for (const file of files) {
    const target = path.join(cwd, file.rel);
    if (existsSync(target) && !force) continue;
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, file.content);
    written.push(file.rel);
  }
  await writeWizardState(cwd, { ...state, logPath, version: 1 });
  return written;
}

export default async function (pi: ExtensionAPI) {
  let state = await loadState();
  let monitorInterval: ReturnType<typeof setInterval> | undefined;

  function sendReport(title: string, body: string, level: ReportLevel = "info") {
    pi.sendMessage({
      customType: MESSAGE_TYPE,
      content: `# ${title}\n\n${body}`,
      display: true,
      details: { level, timestamp: Date.now() },
    });
    if (process.env.PI_TINKER_AGENT_CLI === "1") {
      process.stdout.write(`# ${title}\n\n${body}\n`);
    }
  }

  function registerTinkerProvider() {
    const inklingModels = [
      {
        id: INKLING_SMALL_MODEL,
        name: "Inkling-Small (Tinker, 64K)",
        reasoning: true,
        thinkingLevelMap: inklingThinkingLevels(),
        input: ["text", "image"] as ("text" | "image")[],
        cost: { input: 0.62, output: 1.56, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 65_536,
        maxTokens: 16_384,
        compat: inklingCompat(),
      },
      {
        id: INKLING_MODEL,
        name: "Inkling (Tinker, 64K)",
        reasoning: true,
        thinkingLevelMap: inklingThinkingLevels(),
        input: ["text", "image"] as ("text" | "image")[],
        cost: { input: 1.87, output: 4.68, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 65_536,
        maxTokens: 16_384,
        compat: inklingCompat(),
      },
      {
        id: INKLING_256K_MODEL,
        name: "Inkling (Tinker, 256K)",
        reasoning: true,
        thinkingLevelMap: inklingThinkingLevels(),
        input: ["text", "image"] as ("text" | "image")[],
        cost: { input: 3.74, output: 9.36, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 262_144,
        maxTokens: 16_384,
        compat: inklingCompat(),
      },
    ];
    const checkpointModels = state.checkpoints.map((checkpoint) => {
      const inkling = checkpoint.reasoning ?? isInklingModel(checkpoint.baseModel);
      return {
        id: checkpoint.id,
        name: checkpoint.name,
        reasoning: inkling,
        ...(inkling ? { thinkingLevelMap: inklingThinkingLevels() } : {}),
        input: (checkpoint.vision ?? inkling ? ["text", "image"] : ["text"]) as ("text" | "image")[],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: checkpoint.contextWindow,
        maxTokens: checkpoint.maxTokens,
        ...(inkling ? { compat: inklingCompat() } : {}),
      };
    });
    pi.registerProvider("tinker", {
      name: "Tinker",
      baseUrl: TINKER_ANTHROPIC_BASE_URL,
      apiKey: "$TINKER_API_KEY",
      api: "anthropic-messages",
      models: [...inklingModels, ...checkpointModels],
    });
  }

  registerTinkerProvider();

  pi.registerMessageRenderer(MESSAGE_TYPE, (message, _options, theme) => {
    const details = message.details as { level?: ReportLevel } | undefined;
    const level = details?.level ?? "info";
    const color = level === "error" ? "error" : level === "warning" ? "warning" : level === "success" ? "success" : "accent";
    const box = new Box(1, 1, (s: string) => theme.bg("customMessageBg", s));
    box.addChild(new Text(theme.fg(color, String(message.content)), 0, 0));
    return box;
  });

  pi.on("session_shutdown", () => {
    if (monitorInterval) clearInterval(monitorInterval);
  });

  pi.registerCommand("tinker", {
    description: "Tinker + Inkling helper: chat, sweep effort, fine-tune, evaluate, monitor, and use checkpoints",
    handler: async (args, ctx) => {
      let [subcommandRaw, ...rest] = shellSplit(args);
      let subcommand = subcommandRaw ?? "help";
      if (subcommand === "demo") {
        rest = ["--example", "customer-support", "--budget", "demo", ...rest];
        subcommand = "improve";
      } else if (subcommand === "new" || subcommand === "finetune" || subcommand === "start" || subcommand === "init") {
        const parsed = parseOptions(rest);
        if (parsed.options.budget === undefined) rest = [...rest, "--budget", "demo"];
        if (!parsed.options.goal && parsed.options.metric) rest = [...rest, "--goal", String(parsed.options.metric)];
        subcommand = "improve";
      }

      if (subcommand === "help" || subcommand === "--help" || subcommand === "-h") {
        sendReport("Tinker helper", [
          "## Front door",
          "- `/tinker demo` — free local example; no API calls.",
          "- `/tinker improve data.csv --goal \"what should improve\" --budget demo` — prepare your data without API calls.",
          "- `/tinker next` — the one next command, filled in.",
          "- `/tinker doctor` — check what is installed or missing.",
          "- `/tinker inkling` — Inkling-Small vs Inkling, effort, `/model`.",
          "",
          "## Then, with `--yes`",
          "- `/tinker improve --budget smoke --eval-reviewed --yes` — review holdout, select effort, baseline + 2-step train + checkpoint eval.",
          "- `/tinker improve --budget small --yes` — short training only if smoke beat baseline.",
          "- `/tinker monitor <log_dir>` — watch metrics.",
          "- `/tinker deploy latest` — Tinker API snippets + export/serving plan.",
          "",
          "`new`, `start`, `init`, and `finetune` now run `improve --budget demo`.",
          "Advanced (`validate`, `eval`, `sft`, `smoke`, `use`, …): `docs/commands.md`.",
          "API-using commands require confirmation; `--yes` is explicit approval.",
        ].join("\n"));
        return;
      }

      if (subcommand === "inkling") {
        const [actionRaw, ...inklingRest] = rest;
        const action = actionRaw ?? "info";
        const { positional, options } = parseOptions(inklingRest);

        if (action === "info" || action === "help" || action === "--help") {
          sendReport("Inkling on Tinker", [
            "Inkling is ready to use in Pi. Set `TINKER_API_KEY`, run `/model`, and choose an Inkling model under the Tinker provider.",
            "",
            "## What to pick",
            `- **Default:** \`${INKLING_SMALL_MODEL}\` — 276B total / 12B active. Same renderer, tokenizer, and effort interface as full Inkling. Prefer this for coding, grading, and synthetic data.`,
            `- **Full:** \`${INKLING_MODEL}\` — 975B total / 41B active.`,
            "- 256K long-context variant: `thinkingmachines/Inkling:peft:262144`.",
            "",
            "## What works",
            "- Chat, coding tools, images, audio (via Cookbook sampling scripts), and streamed reasoning.",
            "- Shift+Tab changes reasoning effort: low, medium, high, or xhigh.",
            "- Effort is mandatory. The renderer defaults to 0.9 if you omit it — make that a deliberate choice.",
            "- Sample evals at temperature=1.0. Raise max_tokens with effort (high effort can need 16k+).",
            "- Never tokenizer.encode() a chat prompt for Inkling; use TMLv0 / tml-renderers.",
            "",
            "## To fine-tune",
            "Python 3.11+, Tinker SDK 0.23+, torch>=2.10. tml-renderers ships in the default Cookbook install:",
            "```bash",
            COOKBOOK_INSTALL,
            "```",
            "Then start without API usage:",
            "```text",
            `/tinker improve data.csv --goal "what should improve" --budget demo --model ${INKLING_SMALL_MODEL}`,
            "```",
            "",
            "Keep the same effort when comparing the base model and a trained checkpoint. Sweep first with `/tinker inkling sweep`.",
            "Cookbook does not publish a default Inkling learning rate; calibrate it. Stay on Tinker for serving — there is no self-host export path.",
          ].join("\n"));
          return;
        }

        if (action === "sweep" || action === "sample") {
          const prompt = String(options.prompt ?? positional.join(" ")).trim();
          if (!prompt) {
            sendReport("Inkling effort sweep", "Usage: `/tinker inkling sweep --prompt \"your representative task\" --efforts low,medium,high,xhigh --yes`", "warning");
            return;
          }
          let efforts: number[];
          try {
            efforts = parseInklingEfforts(options.efforts);
          } catch (error) {
            sendReport("Inkling effort sweep", error instanceof Error ? error.message : String(error), "warning");
            return;
          }
          if (!process.env.TINKER_API_KEY) {
            sendReport("Inkling effort sweep", "`TINKER_API_KEY` is not set. Run `/tinker setup` after exporting the key.", "warning");
            return;
          }
          const confirmed = options.yes === true || (ctx.hasUI ? await ctx.ui.confirm("Run Inkling effort sweep?", `Sampling ${efforts.length} responses uses the Tinker API.`) : false);
          if (!confirmed) {
            sendReport("Inkling effort sweep", "Stopped before API usage. Re-run with `--yes` when ready.", "warning");
            return;
          }
          const pythonArgs = [
            "-m",
            "tinker_cookbook.scripts.inkling.sample_reasoning",
            `prompt=${prompt}`,
            `efforts=[${efforts.join(",")}]`,
          ];
          if (options["max-tokens"]) pythonArgs.push(`max_tokens=${Number(options["max-tokens"])}`);
          if (options.temperature) pythonArgs.push(`temperature=${Number(options.temperature)}`);
          try {
            ctx.ui.setStatus("tinker", "Inkling effort sweep");
            const { stdout, stderr } = await execFile("python3", pythonArgs, {
              cwd: ctx.cwd,
              timeout: Number(options.timeout ?? 1_800_000),
              maxBuffer: 24 * 1024 * 1024,
            });
            sendReport("Inkling effort sweep completed", `\`efforts=${efforts.join(", ")}\`\n\n\`\`\`text\n${`${stdout}\n${stderr}`.trim()}\n\`\`\``, "success");
          } catch (error: any) {
            sendReport("Inkling effort sweep failed", [
              error?.message ?? String(error),
              error?.stdout ? `\n## stdout\n\`\`\`text\n${String(error.stdout).split(/\n/).slice(-100).join("\n")}\n\`\`\`` : "",
              error?.stderr ? `\n## stderr\n\`\`\`text\n${String(error.stderr).split(/\n/).slice(-100).join("\n")}\n\`\`\`` : "",
              "\nInstall/upgrade with `uv pip install -U tinker-cookbook`.",
            ].filter(Boolean).join("\n"), "error");
          } finally {
            ctx.ui.setStatus("tinker", undefined);
          }
          return;
        }

        sendReport("Inkling", `Unknown action: ${action}. Use \`/tinker inkling\` or \`/tinker inkling sweep ...\`.`, "warning");
        return;
      }

      if (subcommand === "new" || subcommand === "finetune" || subcommand === "demo") {
        const { positional, options } = parseOptions(rest);
        const force = options.force === true;
        let dataFileInput = positional[0] ? String(positional[0]) : "";
        let metric = String(options.goal ?? options.metric ?? "");
        let model = String(options.model ?? "");
        const exampleSlug = subcommand === "demo" ? "customer-support" : options.example === true ? "customer-support" : options.example ? String(options.example) : "";

        if (exampleSlug) {
          try {
            const result = await writeExampleTemplate(ctx.cwd, exampleSlug, force);
            dataFileInput = path.join("examples", result.template.slug, "train.jsonl");
            metric = metric || result.template.goal;
          } catch (error) {
            sendReport("Tinker new", error instanceof Error ? error.message : String(error), "error");
            return;
          }
        }

        if (ctx.hasUI && !dataFileInput) {
          const choice = await ctx.ui.select("How do you want to start?", [
            "use built-in customer-support example",
            "use built-in structured-extraction example",
            "use built-in concise-writing example",
            "convert my CSV/JSON/docs",
            "I already have chat JSONL",
          ]);
          if (choice?.startsWith("use built-in")) {
            const slug = choice.includes("structured") ? "structured-extraction" : choice.includes("concise") ? "concise-writing" : "customer-support";
            const result = await writeExampleTemplate(ctx.cwd, slug, force);
            dataFileInput = path.join("examples", result.template.slug, "train.jsonl");
            metric = metric || result.template.goal;
          } else {
            dataFileInput = (await ctx.ui.input("Path to CSV, JSON, JSONL, docs dir, or train.jsonl", "data/train.csv"))?.trim() ?? "";
          }
        }

        if (!dataFileInput) {
          sendReport("Tinker new", [
            "Usage:",
            "```text",
            "/tinker improve --example customer-support --budget demo",
            "/tinker new data/train.csv --goal \"better support answers\"",
            "/tinker new data/train.jsonl --goal \"what should improve\"",
            "```",
          ].join("\n"), "warning");
          return;
        }

        let dataFile = path.resolve(ctx.cwd, dataFileInput);
        if (!existsSync(dataFile)) {
          sendReport("Tinker new", `Input not found: ${dataFile}`, "error");
          return;
        }
        if (!/\.jsonl$/i.test(dataFile) || options.prepare === true) {
          const out = path.resolve(ctx.cwd, String(options.out ?? "data/train.jsonl"));
          try {
            const prepared = await prepareDataset(dataFile, out);
            dataFile = prepared.outFile;
            dataFileInput = rel(ctx.cwd, prepared.outFile);
            sendReport("Dataset prepared", [
              `Converted \`${rel(ctx.cwd, prepared.inputPath)}\` (${prepared.format}) to \`${rel(ctx.cwd, prepared.outFile)}\` with ${prepared.rows} rows.`,
              prepared.warnings.length ? `\n## Warnings\n${prepared.warnings.map((w) => `- ${w}`).join("\n")}` : "",
              `\n## Samples\n${prepared.samples.map((x, i) => `### ${i + 1}\n\`\`\`json\n${x}\n\`\`\``).join("\n\n")}`,
            ].filter(Boolean).join("\n"), "success");
          } catch (error) {
            sendReport("Dataset prepare failed", error instanceof Error ? error.message : String(error), "error");
            return;
          }
        }

        if (ctx.hasUI && !model) {
          const choice = await ctx.ui.select("Pick a starter model", starterModelChoices());
          model = choice === "custom" ? ((await ctx.ui.input("Tinker model id", DEFAULT_MODEL))?.trim() || DEFAULT_MODEL) : (choice?.split(" — ")[0] || DEFAULT_MODEL);
        }
        model = model || DEFAULT_MODEL;

        if (ctx.hasUI && !metric) {
          metric = (await ctx.ui.input("What should improve?", "e.g. support answer quality, extraction accuracy, concise writing"))?.trim() ?? "";
        }
        metric = metric || "Improve task quality on held-out examples.";

        const state: WizardState = {
          version: 1,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          dataFile,
          dataFileInput,
          model,
          metric,
          logPath: String(options.log ?? `logs/sft-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`),
        };
        await writeWizardState(ctx.cwd, state);
        const written = await createWizardFiles(ctx.cwd, state, force);
        const saved = await readWizardState(ctx.cwd) ?? state;
        sendReport("Tinker golden path ready", [
          written.length ? `Created ${written.map((x) => `\`${x}\``).join(", ")}.` : "Project files already existed; left them untouched.",
          "",
          recommendPlan(metric, readJsonl(dataFile).length),
          "",
          renderWizard(ctx.cwd, saved),
        ].join("\n"), "success");
        return;
      }

      if (subcommand === "prepare") {
        const { positional, options } = parseOptions(rest);
        const input = positional[0];
        if (!input) {
          sendReport("Tinker prepare", "Usage: `/tinker prepare data.csv --out data/train.jsonl`", "warning");
          return;
        }
        const inputPath = path.resolve(ctx.cwd, input);
        const outFile = path.resolve(ctx.cwd, String(options.out ?? "data/train.jsonl"));
        if (!existsSync(inputPath)) {
          sendReport("Tinker prepare", `Input not found: ${inputPath}`, "error");
          return;
        }
        try {
          const prepared = await prepareDataset(inputPath, outFile);
          sendReport("Dataset prepared", [
            `Converted \`${rel(ctx.cwd, inputPath)}\` (${prepared.format}) to \`${rel(ctx.cwd, outFile)}\`.`,
            `- Rows: ${prepared.rows}`,
            prepared.warnings.length ? `\n## Warnings\n${prepared.warnings.map((w) => `- ${w}`).join("\n")}` : "",
            "\n## Next",
            "```text",
            `/tinker new ${rel(ctx.cwd, outFile)} --goal "what should improve"`,
            `/tinker validate ${rel(ctx.cwd, outFile)} --quick`,
            "```",
          ].filter(Boolean).join("\n"), "success");
        } catch (error) {
          sendReport("Dataset prepare failed", error instanceof Error ? error.message : String(error), "error");
        }
        return;
      }

      if (subcommand === "recommend") {
        const { positional, options } = parseOptions(rest);
        const goal = String(options.goal ?? positional.join(" ") ?? "");
        const data = options.data ? path.resolve(ctx.cwd, String(options.data)) : undefined;
        let rows: number | undefined;
        if (data && existsSync(data) && /\.jsonl$/i.test(data)) {
          try { rows = readJsonl(data).length; } catch {}
        }
        sendReport("Tinker recommendation", recommendPlan(goal, rows), "info");
        return;
      }

      if (subcommand === "examples") {
        const { positional, options } = parseOptions(rest);
        const action = positional[0] ?? "list";
        if (action === "list") {
          sendReport("Tinker examples", [
            "Available starter examples:",
            ...EXAMPLE_TEMPLATES.map((t) => `- \`${t.slug}\` — ${t.title}`),
            "",
            "Copy one into this project:",
            "```text",
            "/tinker examples copy customer-support",
            "/tinker improve --example customer-support --budget demo",
            "```",
          ].join("\n"));
          return;
        }
        if (action === "copy") {
          const slug = positional[1] ?? "customer-support";
          try {
            const result = await writeExampleTemplate(ctx.cwd, slug, options.force === true);
            sendReport("Tinker example copied", [
              `Copied \`${result.template.title}\`.`,
              result.files.length ? `Wrote ${result.files.map((f) => `\`${f}\``).join(", ")}.` : "Files already existed; left them untouched.",
              "",
              "Start from it:",
              "```text",
              `/tinker improve examples/${result.template.slug}/train.jsonl --goal "${result.template.goal}" --budget demo`,
              "```",
            ].join("\n"), "success");
          } catch (error) {
            sendReport("Tinker examples", error instanceof Error ? error.message : String(error), "error");
          }
          return;
        }
        sendReport("Tinker examples", "Usage: `/tinker examples list` or `/tinker examples copy customer-support`", "warning");
        return;
      }

      if (subcommand === "doctor") {
        const { positional } = parseOptions(rest);
        sendReport("Tinker doctor", await buildDoctorReport(ctx.cwd, positional[0]), "info");
        return;
      }


      if (subcommand === "improve") {
        const { positional, options } = parseOptions(rest);
        let budget: ImproveBudget;
        try {
          budget = parseImproveBudget(options.budget ?? "demo");
        } catch (error) {
          sendReport("Tinker improve", error instanceof Error ? error.message : String(error), "warning");
          return;
        }

        const force = options.force === true;
        const yes = options.yes === true;
        const existingWizard = await readWizardState(ctx.cwd);
        const model = String(options.model ?? existingWizard?.model ?? DEFAULT_MODEL);
        let metric = String(options.goal ?? options.metric ?? existingWizard?.metric ?? "Improve task quality on held-out examples.");
        const steps = budgetMaxSteps(budget, options.steps ?? options.max_steps);
        const exampleSlug = options.example === true ? "customer-support" : options.example ? String(options.example) : "";
        let dataFileInput = positional[0] ? String(positional[0]) : existingWizard?.sourceDataFileInput ?? existingWizard?.dataFileInput ?? existingWizard?.sourceDataFile ?? existingWizard?.dataFile ?? "";
        let dataFile = dataFileInput ? path.resolve(ctx.cwd, dataFileInput) : "";
        let evalOverride = options.eval ? path.resolve(ctx.cwd, String(options.eval)) : undefined;
        const sections: string[] = [managedRunPlan(budget)];

        try {
          if (exampleSlug) {
            const result = await writeExampleTemplate(ctx.cwd, exampleSlug, force);
            dataFileInput = path.join("examples", result.template.slug, "train.jsonl");
            dataFile = path.resolve(ctx.cwd, dataFileInput);
            evalOverride = path.resolve(ctx.cwd, "examples", result.template.slug, "eval.jsonl");
            if (options.goal === undefined && options.metric === undefined) metric = result.template.goal;
            sections.push(`## Example\nCopied \`${result.template.title}\` to \`${dataFileInput}\`.`);
          }
          if (isRetiredModel(model) && !force) {
            sendReport("Tinker improve", [
              `\`${model}\` is not on Tinker's current lineup.`,
              `Check ${MODELS_DOCS_URL} and ${DEPRECATIONS_URL}.`,
              "Pass `--force` to proceed anyway, or pick Inkling-Small / a live Qwen id.",
            ].join("\n"), "error");
            return;
          }
          if (!dataFileInput) {
            sendReport("Tinker improve", [
              "Usage:",
              "```text",
              "/tinker improve data.csv --goal \"better support answers\" --budget demo",
              "/tinker improve data/train.jsonl --goal \"better extraction\" --budget small --yes",
              "```",
              "",
              "Use `--budget demo` to do everything up to real API usage.",
            ].join("\n"), "warning");
            return;
          }
          if (!existsSync(dataFile)) {
            sendReport("Tinker improve", `Input not found: ${dataFile}`, "error");
            return;
          }

          let normalizedSource = dataFile;
          if (!/\.jsonl$/i.test(dataFile) || options.prepare === true) {
            const prepared = await prepareDataset(dataFile, path.resolve(ctx.cwd, ".tinker-pi", "normalized-input.jsonl"));
            normalizedSource = prepared.outFile;
            sections.push(`## Data prepared\nConverted \`${rel(ctx.cwd, prepared.inputPath)}\` (${prepared.format}) into ${prepared.rows} normalized conversation rows.`);
            if (prepared.warnings.length) sections.push(`## Data preparation warnings\n${prepared.warnings.map((w) => `- ${w}`).join("\n")}`);
          } else {
            sections.push(`## Data selected\nUsing \`${rel(ctx.cwd, dataFile)}\` as the source.`);
          }
          if (evalOverride && !existsSync(evalOverride)) throw new Error(`Eval file not found: ${evalOverride}`);

          const normalizedSourceHash = normalizedJsonlHash(normalizedSource);
          const sourceUnchanged = existingWizard?.sourceDataHash === normalizedSourceHash;
          const preserveGeneratedEval = !force && sourceUnchanged && existingWizard?.evalGenerated === true && existsSync(path.join(ctx.cwd, "data", "eval.jsonl"));
          const managed = await writeManagedDataFiles({ cwd: ctx.cwd, normalizedSource, evalOverride, preserveGeneratedEval });
          dataFile = managed.trainFile;
          const sourceInputResolved = path.resolve(ctx.cwd, dataFileInput);
          const sourceDataFileInput = sourceInputResolved === managed.trainFile ? rel(ctx.cwd, managed.sourceFile) : dataFileInput;
          dataFileInput = rel(ctx.cwd, managed.trainFile);
          sections.push(`## Held-out data\n- Source: ${managed.sourceRows} rows\n- Training: ${managed.trainRows} rows in \`${dataFileInput}\`\n- Eval: ${managed.evalRows} rows in \`${rel(ctx.cwd, managed.evalFile)}\`\n${managed.warnings.map((warning) => `- ${warning}`).join("\n")}`);

          const stateForRun: WizardState = existingWizard ? { ...existingWizard } : {
            version: 1,
            createdAt: Date.now(),
            updatedAt: Date.now(),
          };
          const modelChanged = Boolean(existingWizard?.model && existingWizard.model !== model);
          const effortChanged = options.effort !== undefined && Number(options.effort) !== existingWizard?.effort;
          const artifactsChanged = Boolean(existingWizard && (
            existingWizard.sourceDataHash !== managed.sourceHash ||
            existingWizard.trainingDataHash !== managed.trainingHash ||
            existingWizard.evalDataHash !== managed.evalHash ||
            modelChanged ||
            effortChanged
          ));
          if (artifactsChanged) {
            Object.assign(stateForRun, {
              validationAt: undefined,
              baselineResult: undefined,
              baselineFingerprint: undefined,
              smokeAt: undefined,
              smokeLogDir: undefined,
              checkpointPath: undefined,
              candidateCheckpointPath: undefined,
              candidateResult: undefined,
              candidateFingerprint: undefined,
              candidateDecision: undefined,
              approvedCheckpointPath: undefined,
              approvedResult: undefined,
              approvedFingerprint: undefined,
              approvedAt: undefined,
              registeredModel: undefined,
              effortSweepAt: undefined,
              effortSweepFingerprint: undefined,
            });
            sections.push("## Prior results invalidated\nThe source data, held-out eval, or model changed. Old baseline, checkpoint, and approval state will not be reused.");
          }
          stateForRun.sourceDataFile = managed.sourceFile;
          stateForRun.sourceDataFileInput = sourceDataFileInput;
          stateForRun.sourceDataHash = managed.sourceHash;
          stateForRun.dataFile = managed.trainFile;
          stateForRun.dataFileInput = dataFileInput;
          stateForRun.trainingDataHash = managed.trainingHash;
          stateForRun.evalDataHash = managed.evalHash;
          stateForRun.evalGenerated = managed.evalGenerated;
          stateForRun.evalReviewedHash = evalOverride ? managed.evalHash : options["eval-reviewed"] === true ? managed.evalHash : (stateForRun.evalReviewedHash === managed.evalHash ? managed.evalHash : undefined);
          stateForRun.model = model;
          stateForRun.metric = metric;
          stateForRun.logPath = String(options.log ?? (artifactsChanged ? undefined : stateForRun.logPath) ?? `logs/sft-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`);
          await writeWizardState(ctx.cwd, stateForRun);
          const written = await createWizardFiles(ctx.cwd, stateForRun, force);
          sections.push(written.length ? `## Project files\nCreated ${written.map((x) => `\`${x}\``).join(", ")}.` : "## Project files\nProject files already exist; left them untouched.");

          sections.push(await buildDoctorReport(ctx.cwd, dataFileInput));
          sections.push(`## Lightweight validation\n${validateDataset(dataFile)}`);
          await patchWizardState(ctx.cwd, { validationAt: Date.now() });

          if (budget === "demo") {
            const saved = (await readWizardState(ctx.cwd)) ?? stateForRun;
            sections.push("## Stopped before API usage\nDemo budget completed setup, project files, doctor, and lightweight validation without touching the Tinker API.");
            sections.push(`## Next command\n\`\`\`text\n${nextImproveCommand(ctx.cwd, saved)}\n\`\`\``);
            sendReport("Tinker improve plan ready", sections.join("\n\n---\n\n"), "success");
            return;
          }

          if (!process.env.TINKER_API_KEY) {
            sendReport("Tinker improve", "`TINKER_API_KEY` is not set. Run `/tinker improve ... --budget demo` for no-API setup, or set the key before `smoke`, `small`, or `real` budgets.", "warning");
            return;
          }

          const confirmed = yes || (ctx.hasUI ? await ctx.ui.confirm("Run managed Tinker improvement?", `Budget ${budget} may incur Tinker API usage. Continue?`) : false);
          if (!confirmed) {
            sendReport("Tinker improve", "Stopped before API usage. Re-run with `--yes` when you are ready.", "warning");
            return;
          }

          const script = path.resolve(ctx.cwd, "eval.py");
          const evalData = path.resolve(ctx.cwd, "data", "eval.jsonl");
          if (!existsSync(script) || !existsSync(evalData)) {
            sendReport("Tinker improve", "Missing `eval.py` or `data/eval.jsonl`. Run `/tinker eval init`, edit eval rows, then re-run improve.", "warning");
            return;
          }
          const evalRowCount = readFileSync(evalData, "utf8").split(/\n/).filter((line) => line.trim()).length;
          if (evalRowCount < 2 && !force) {
            sendReport("Tinker improve", `Held-out eval has ${evalRowCount} row(s). Add at least 2 reviewed rows to \`data/eval.jsonl\` before API usage.`, "warning");
            return;
          }
          const currentEvalHash = hashFile(evalData)!;
          if (stateForRun.evalGenerated && stateForRun.evalReviewedHash !== currentEvalHash && options["eval-reviewed"] !== true) {
            sendReport("Tinker improve", "Review `data/eval.jsonl`, then re-run the exact command with `--eval-reviewed`. Auto-held-out answers use exact matching and should not be trusted silently.", "warning");
            return;
          }
          if (options["eval-reviewed"] === true) {
            stateForRun.evalReviewedHash = currentEvalHash;
            await patchWizardState(ctx.cwd, { evalReviewedHash: currentEvalHash });
          }

          const baselineOut = path.resolve(ctx.cwd, String(options["baseline-out"] ?? "eval_results/baseline.json"));
          const sweepFingerprint = provenanceFingerprint({
            eval: currentEvalHash,
            evalScript: hashFile(script),
            model: stateForRun.model,
          });
          let evalEffort = Number(options.effort ?? stateForRun.effort ?? 0.9);
          if (!Number.isFinite(evalEffort) || evalEffort < 0 || evalEffort >= 1) evalEffort = 0.9;
          let baselineCreatedBySweep = false;
          if (isInklingModel(stateForRun.model) && options.effort === undefined && options["no-sweep"] !== true && stateForRun.effortSweepFingerprint !== sweepFingerprint) {
            const efforts = parseInklingEfforts(options.efforts);
            const results: Array<{ effort: number; accuracy: number; out: string }> = [];
            ctx.ui.setStatus("tinker", "managed improve: score effort sweep on held-out eval");
            for (const effort of efforts) {
              const out = path.resolve(ctx.cwd, `eval_results/effort-${String(effort).replace(".", "-")}.json`);
              await execFile("python3", [script, "--base-model", stateForRun.model ?? model, "--effort", String(effort), "--data", evalData, "--out", out], { cwd: ctx.cwd, timeout: Number(options.timeout ?? 1_800_000), maxBuffer: 12 * 1024 * 1024 });
              results.push({ effort, accuracy: readEvalSummary(out).accuracy ?? 0, out });
            }
            results.sort((a, b) => b.accuracy - a.accuracy || a.effort - b.effort);
            const selected = results[0]!;
            evalEffort = selected.effort;
            await mkdir(path.dirname(baselineOut), { recursive: true });
            await writeFile(baselineOut, readFileSync(selected.out));
            baselineCreatedBySweep = true;
            sections.push([
              "## Inkling effort selected from eval behavior",
              "| Effort | Accuracy |",
              "|---:|---:|",
              ...results.slice().sort((a, b) => a.effort - b.effort).map((result) => `| ${result.effort} | ${(result.accuracy * 100).toFixed(1)}% |`),
              "",
              `Pinned **${evalEffort}**: the lowest effort tied for best held-out accuracy. Training, baseline, and checkpoint eval use this same value.`,
            ].join("\n"));
            stateForRun.effortSweepAt = Date.now();
            stateForRun.effortSweepFingerprint = sweepFingerprint;
          } else if (isInklingModel(stateForRun.model)) {
            sections.push(`## Inkling effort\nPinned **${evalEffort}** for training, baseline, and checkpoint eval.`);
          }
          stateForRun.effort = evalEffort;

          const trainScript = path.resolve(ctx.cwd, "train_sft.py");
          if (isInklingModel(stateForRun.model) && !readFileSync(trainScript, "utf8").includes("EffortConversationFileBuilder")) {
            sendReport("Tinker improve", "This project has an older `train_sft.py` that cannot pin Inkling training effort. Re-run the demo/setup command with `--force`, review the regenerated script, then continue.", "warning");
            return;
          }

          const runFingerprint = provenanceFingerprint({
            data: hashFile(dataFile),
            effort: evalEffort,
            eval: currentEvalHash,
            evalScript: hashFile(script),
            model: stateForRun.model,
          });
          await patchWizardState(ctx.cwd, {
            effort: evalEffort,
            effortSweepAt: stateForRun.effortSweepAt,
            effortSweepFingerprint: stateForRun.effortSweepFingerprint,
          });
          const yamlPath = path.join(ctx.cwd, "tinker.yaml");
          if (isInklingModel(stateForRun.model) && existsSync(yamlPath)) {
            const yaml = readFileSync(yamlPath, "utf8");
            await writeFile(yamlPath, /^effort:/m.test(yaml) ? yaml.replace(/^effort:.*$/m, `effort: ${evalEffort}`) : `${yaml.trimEnd()}\neffort: ${evalEffort}\n`);
          }

          ctx.ui.setStatus("tinker", "managed improve: baseline eval");
          let baselineNeedsRun = !baselineCreatedBySweep && (!existsSync(baselineOut) || force || stateForRun.baselineFingerprint !== runFingerprint);
          if (baselineNeedsRun && existsSync(baselineOut)) sections.push("## Baseline invalidated\nTraining data, eval data, eval code, model, or effort changed.");
          if (baselineNeedsRun) {
            const { stdout, stderr } = await execFile("python3", [script, "--base-model", stateForRun.model ?? model, "--effort", String(evalEffort), "--data", evalData, "--out", baselineOut], { cwd: ctx.cwd, timeout: Number(options.timeout ?? 1_800_000), maxBuffer: 12 * 1024 * 1024 });
            sections.push(`## Baseline eval\n${formatEvalSummary(baselineOut)}\n\n<details><summary>Output tail</summary>\n\n\`\`\`text\n${`${stdout}\n${stderr}`.trim().split(/\n/).slice(-40).join("\n")}\n\`\`\`\n</details>`);
          } else {
            sections.push(`## Baseline eval\nUsing provenance-matched \`${rel(ctx.cwd, baselineOut)}\`.\n\n${formatEvalSummary(baselineOut)}`);
          }
          stateForRun.baselineResult = baselineOut;
          stateForRun.baselineFingerprint = runFingerprint;
          await patchWizardState(ctx.cwd, { baselineResult: baselineOut, baselineFingerprint: runFingerprint, model: stateForRun.model });

          ctx.ui.setStatus("tinker", "managed improve: smoke train");
          const baseLog = String(stateForRun.logPath ?? "logs/sft-managed");
          const smokeLog = `${baseLog}-smoke`;
          const smoke = await execFile("python3", [trainScript, "max_steps=2", "save_every=1", `log_path=${smokeLog}`, `effort=${evalEffort}`], { cwd: ctx.cwd, timeout: Number(options.timeout ?? 1_800_000), maxBuffer: 12 * 1024 * 1024 });
          sections.push(`## Smoke train\n2-step smoke training completed.\n\nLatest metrics:\n${latestMetrics(path.join(ctx.cwd, smokeLog, "metrics.jsonl")) ?? "No metrics found yet."}\n\n<details><summary>Output tail</summary>\n\n\`\`\`text\n${`${smoke.stdout}\n${smoke.stderr}`.trim().split(/\n/).slice(-40).join("\n")}\n\`\`\`\n</details>`);
          await patchWizardState(ctx.cwd, { smokeAt: Date.now(), smokeLogDir: smokeLog });

          const smokeCheckpoint = firstSamplerCheckpoint(path.resolve(ctx.cwd, smokeLog));
          let smokeBeatBaseline = force;
          if (smokeCheckpoint) {
            ctx.ui.setStatus("tinker", "managed improve: smoke checkpoint eval");
            const smokeEvalOut = path.resolve(ctx.cwd, "eval_results/smoke.json");
            const smokeEval = await execFile("python3", [script, "--model-path", smokeCheckpoint, "--renderer-model", stateForRun.model ?? model, "--effort", String(evalEffort), "--data", evalData, "--out", smokeEvalOut], { cwd: ctx.cwd, timeout: Number(options.timeout ?? 1_800_000), maxBuffer: 12 * 1024 * 1024 });
            sections.push(`## Smoke checkpoint eval\n${formatEvalSummary(smokeEvalOut)}\n\n<details><summary>Output tail</summary>\n\n\`\`\`text\n${`${smokeEval.stdout}\n${smokeEval.stderr}`.trim().split(/\n/).slice(-40).join("\n")}\n\`\`\`\n</details>`);
            sections.push(compareEvalSummaries(baselineOut, smokeEvalOut));
            const cmp = evalBeatBaseline(baselineOut, smokeEvalOut);
            smokeBeatBaseline = cmp.ok || force;
            stateForRun.checkpointPath = smokeCheckpoint;
            stateForRun.candidateCheckpointPath = smokeCheckpoint;
            stateForRun.candidateResult = smokeEvalOut;
            stateForRun.candidateFingerprint = runFingerprint;
            stateForRun.candidateDecision = cmp.ok || force ? "approved" : "rejected";
            await patchWizardState(ctx.cwd, {
              checkpointPath: smokeCheckpoint,
              candidateCheckpointPath: smokeCheckpoint,
              candidateResult: smokeEvalOut,
              candidateFingerprint: runFingerprint,
              candidateDecision: stateForRun.candidateDecision,
            });
            if (!cmp.ok) {
              sections.push(suggestDataImprovementsFromEval(smokeEvalOut));
              sections.push(`## Stopped: checkpoint did not beat baseline\n${(cmp.candidate * 100).toFixed(1)}% vs ${(cmp.baseline * 100).toFixed(1)}%. Add 2–5 training examples like the failures, then re-run smoke. Pass --force to scale anyway.`);
              const saved = (await readWizardState(ctx.cwd)) ?? stateForRun;
              sections.push(`## Next command\n\`\`\`text\n${nextImproveCommand(ctx.cwd, saved)}\n\`\`\``);
              if (budget !== "smoke" && !force) {
                sendReport("Tinker improve stopped: no improvement", sections.join("\n\n---\n\n"), "warning");
                return;
              }
            }
          } else {
            sections.push("## Smoke checkpoint\nNo sampler checkpoint after 2 steps (even with `save_every=1`). Inspect logs before scaling.");
          }

          if (budget === "smoke") {
            const saved = (await readWizardState(ctx.cwd)) ?? stateForRun;
            sections.push(smokeBeatBaseline
              ? "## Stopped after smoke budget\nSmoke beat baseline (or `--force`). Next: `--budget small --yes`."
              : "## Stopped after smoke budget\nDo not scale yet.");
            sections.push(`## Next command\n\`\`\`text\n${nextImproveCommand(ctx.cwd, saved)}\n\`\`\``);
            sendReport("Tinker improve smoke completed", sections.join("\n\n---\n\n"), smokeBeatBaseline ? "success" : "warning");
            return;
          }

          if (!smokeBeatBaseline && !force) {
            sendReport("Tinker improve refused to scale", sections.join("\n\n---\n\n"), "warning");
            return;
          }

          ctx.ui.setStatus("tinker", `managed improve: train ${steps} steps`);
          const trainLog = `${baseLog}-${budget}-${steps}`;
          const train = await execFile("python3", [trainScript, `max_steps=${steps}`, `log_path=${trainLog}`, `effort=${evalEffort}`], { cwd: ctx.cwd, timeout: Number(options.trainTimeout ?? options.timeout ?? 3_600_000), maxBuffer: 12 * 1024 * 1024 });
          const checkpoint = firstSamplerCheckpoint(path.resolve(ctx.cwd, trainLog));
          sections.push(`## Training\nRan ${steps} steps in \`${trainLog}\`.\n\nLatest metrics:\n${latestMetrics(path.join(ctx.cwd, trainLog, "metrics.jsonl")) ?? "No metrics found yet."}\n\n${checkpoint ? `Sampler checkpoint: \`${checkpoint}\`` : "No sampler checkpoint found yet."}\n\n<details><summary>Output tail</summary>\n\n\`\`\`text\n${`${train.stdout}\n${train.stderr}`.trim().split(/\n/).slice(-60).join("\n")}\n\`\`\`\n</details>`);
          await patchWizardState(ctx.cwd, { checkpointPath: checkpoint, candidateCheckpointPath: checkpoint, candidateFingerprint: runFingerprint, logPath: trainLog });

          if (checkpoint) {
            ctx.ui.setStatus("tinker", "managed improve: checkpoint eval");
            const candidateOut = path.resolve(ctx.cwd, String(options["candidate-out"] ?? `eval_results/${sanitizeName(path.basename(trainLog))}.json`));
            const evalRun = await execFile("python3", [script, "--model-path", checkpoint, "--renderer-model", stateForRun.model ?? model, "--effort", String(evalEffort), "--data", evalData, "--out", candidateOut], { cwd: ctx.cwd, timeout: Number(options.timeout ?? 1_800_000), maxBuffer: 12 * 1024 * 1024 });
            sections.push(`## Checkpoint eval\n${formatEvalSummary(candidateOut)}\n\n<details><summary>Output tail</summary>\n\n\`\`\`text\n${`${evalRun.stdout}\n${evalRun.stderr}`.trim().split(/\n/).slice(-40).join("\n")}\n\`\`\`\n</details>`);
            sections.push(compareEvalSummaries(baselineOut, candidateOut));
            const cmp = evalBeatBaseline(baselineOut, candidateOut);
            await patchWizardState(ctx.cwd, {
              checkpointPath: checkpoint,
              candidateCheckpointPath: checkpoint,
              candidateResult: candidateOut,
              candidateFingerprint: runFingerprint,
              candidateDecision: cmp.ok || force ? "approved" : "rejected",
            });
            if (!cmp.ok && !force) {
              sections.push(suggestDataImprovementsFromEval(candidateOut));
              sections.push(`## Not registered\nCheckpoint did not beat baseline (${(cmp.candidate * 100).toFixed(1)}% vs ${(cmp.baseline * 100).toFixed(1)}%). Not registering for chat. Add examples and re-run smoke, or pass --force.`);
              const saved = (await readWizardState(ctx.cwd)) ?? stateForRun;
              sections.push(`## Next command\n\`\`\`text\n${nextImproveCommand(ctx.cwd, saved)}\n\`\`\``);
              sendReport("Tinker improve completed without a win", sections.join("\n\n---\n\n"), "warning");
              return;
            }
            sections.push(suggestDataImprovementsFromEval(candidateOut));
            await patchWizardState(ctx.cwd, {
              approvedCheckpointPath: checkpoint,
              approvedResult: candidateOut,
              approvedFingerprint: runFingerprint,
              approvedAt: Date.now(),
              candidateDecision: "approved",
            });
            const alias = sanitizeName(String(options.alias ?? `tinker-${path.basename(trainLog)}`));
            if (options.register !== false) {
              state.checkpoints = state.checkpoints.filter((m) => m.id !== checkpoint && m.name !== alias);
              state.checkpoints.push(checkpointRegistration({ id: checkpoint, name: alias, baseModel: stateForRun.model }));
              await saveState(state);
              registerTinkerProvider();
              await patchWizardState(ctx.cwd, {
                checkpointPath: checkpoint,
                candidateCheckpointPath: checkpoint,
                candidateResult: candidateOut,
                candidateFingerprint: runFingerprint,
                candidateDecision: "approved",
                registeredModel: alias,
              });
              sections.push(`## Registered for chat\nRegistered \`${alias}\`. Use \`/model\` to select it, then:\n\n\`\`\`text\n/tinker deploy ${alias}\n\`\`\``);
            } else {
              await patchWizardState(ctx.cwd, { registeredModel: undefined });
            }
          } else {
            sections.push("## No checkpoint yet\nTraining completed but no sampler checkpoint was found. Check `checkpoints.jsonl` or increase steps/save frequency.");
          }

          const saved = (await readWizardState(ctx.cwd)) ?? stateForRun;
          sections.push(`## Next command\n\`\`\`text\n${nextImproveCommand(ctx.cwd, saved)}\n\`\`\``);
          sendReport("Tinker improve completed", sections.join("\n\n---\n\n"), "success");
        } catch (error: any) {
          sendReport("Tinker improve failed", [
            error?.message ?? String(error),
            error?.stdout ? `\n## stdout\n\`\`\`text\n${String(error.stdout).split(/\n/).slice(-80).join("\n")}\n\`\`\`` : "",
            error?.stderr ? `\n## stderr\n\`\`\`text\n${String(error.stderr).split(/\n/).slice(-80).join("\n")}\n\`\`\`` : "",
            "\nRun `/tinker doctor` and `/skill:tinker-debug` with this output.",
          ].filter(Boolean).join("\n"), "error");
        } finally {
          ctx.ui.setStatus("tinker", undefined);
        }
        return;
      }

      if (subcommand === "deploy") {
        const { positional, options } = parseOptions(rest);
        const wizard = await readWizardState(ctx.cwd);
        const resolved = resolveCheckpointRef(positional[0], state, wizard);
        if (!resolved.checkpoint) {
          sendReport("Tinker deploy", [
            positional[0] === "latest" || !positional[0] ? "No approved checkpoint found to deploy. A candidate must beat its provenance-matched baseline first." : "No checkpoint found to deploy.",
            "",
            "Usage:",
            "```text",
            "/tinker deploy tinker://.../sampler_weights/... my-model",
            "/tinker deploy latest",
            "/tinker deploy <registered-alias>",
            "```",
          ].join("\n"), "warning");
          return;
        }
        if (wizard?.candidateCheckpointPath === resolved.checkpoint && wizard.approvedCheckpointPath !== resolved.checkpoint && options.force !== true) {
          sendReport("Tinker deploy blocked", "That checkpoint is a rejected or unevaluated candidate. Deploy an approved checkpoint, or pass `--force` with the explicit checkpoint URI.", "warning");
          return;
        }
        if (wizard?.approvedCheckpointPath === resolved.checkpoint && wizard.approvedFingerprint !== wizardRunFingerprint(ctx.cwd, wizard) && options.force !== true) {
          sendReport("Tinker deploy blocked", "The approved checkpoint's data/eval/code/model/effort provenance is stale. Re-run `/tinker improve` to evaluate again, or pass `--force` with the explicit checkpoint URI.", "warning");
          return;
        }
        const alias = sanitizeName(String(options.alias ?? positional[1] ?? resolved.alias ?? "my-finetune"));
        const outDir = String(options.out ?? path.join("deploy", alias));
        const baseModel = String(options.model ?? options["base-model"] ?? resolved.baseModel ?? "");
        const written = await writeDeployFiles(ctx.cwd, {
          checkpoint: resolved.checkpoint,
          alias,
          outDir,
          force: options.force === true,
          baseModel: baseModel || undefined,
        });
        sendReport("Tinker deploy files ready", [
          written.length ? `Wrote ${written.map((x) => `\`${x}\``).join(", ")}.` : "Deploy files already existed; left them untouched. Re-run with `--force` to overwrite.",
          baseModel ? `\nBase model: \`${baseModel}\`` : "\nPass `--model <base-model>` if export/serving guidance looks generic.",
          "",
          "## Tinker API (inspection)",
          "```bash",
          `cd ${outDir}`,
          "cp .env.example .env  # then fill TINKER_API_KEY",
          "python3 -m pip install openai",
          "python3 python_client.py",
          "```",
          "",
          "## Export / serving decision",
          "Read `EXPORT.md` and `SERVING.md` before standing up GPUs. Inkling stays on Tinker.",
          "```bash",
          "python3 export.py --merge   # no-op / exits for Inkling",
          "```",
        ].join("\n"), "success");
        return;
      }

      if (subcommand === "start") {
        const { positional, options } = parseOptions(rest);
        let dataFileInput = positional[0] ? String(positional[0]) : "";
        let model = String(options.model ?? "");
        let metric = String(options.metric ?? "");
        const force = options.force === true;

        if (ctx.hasUI && !dataFileInput) {
          dataFileInput = (await ctx.ui.input("Where is your training JSONL?", "data/train.jsonl"))?.trim() ?? "";
        }
        if (!dataFileInput) {
          sendReport("Fine-tune wizard", "Usage: `/tinker start data/train.jsonl --model thinkingmachines/Inkling-Small --metric 'what should improve'`", "warning");
          return;
        }
        const dataFile = path.resolve(ctx.cwd, dataFileInput);
        if (!existsSync(dataFile)) {
          sendReport("Fine-tune wizard", `Data file not found: ${dataFile}`, "error");
          return;
        }

        if (ctx.hasUI && !model) {
          const choice = await ctx.ui.select("Pick a starter model", starterModelChoices());
          model = choice === "custom" ? ((await ctx.ui.input("Tinker model id", DEFAULT_MODEL))?.trim() || DEFAULT_MODEL) : (choice?.split(" — ")[0] || DEFAULT_MODEL);
        }
        model = model || DEFAULT_MODEL;

        if (ctx.hasUI && !metric) {
          metric = (await ctx.ui.input("What should the model get better at?", "e.g. support answers, concise rewrites, JSON extraction accuracy"))?.trim() ?? "";
        }
        metric = metric || "Define success before scaling beyond a smoke test.";

        const state: WizardState = {
          version: 1,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          dataFile,
          dataFileInput,
          model,
          metric,
          logPath: String(options.log ?? `logs/sft-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`),
        };
        await writeWizardState(ctx.cwd, state);
        const written = await createWizardFiles(ctx.cwd, state, force);
        const saved = await readWizardState(ctx.cwd) ?? state;
        sendReport("Fine-tune wizard started", [
          written.length ? `Created ${written.map((x) => `\`${x}\``).join(", ")}.` : "Project files already existed; left them untouched.",
          "",
          renderWizard(ctx.cwd, saved),
        ].join("\n"), "success");
        return;
      }

      if (subcommand === "next") {
        const { options } = parseOptions(rest);
        const state = await readWizardState(ctx.cwd);
        if (!state) {
          sendReport("Fine-tune wizard", "No wizard state found. Start with `/tinker improve data.csv --budget demo` or `/tinker demo`.", "warning");
          return;
        }
        const steps = wizardSteps(ctx.cwd, state);
        const next = steps.find((step) => !step.done);
        if ((options["create-files"] === true || next?.key === "files") && state.dataFile) {
          const written = await createWizardFiles(ctx.cwd, state, options.force === true);
          const saved = await readWizardState(ctx.cwd) ?? state;
          sendReport("Fine-tune wizard files", [
            written.length ? `Created ${written.map((x) => `\`${x}\``).join(", ")}.` : "All project files already exist.",
            "",
            renderWizard(ctx.cwd, saved),
          ].join("\n"), "success");
          return;
        }
        sendReport("Fine-tune wizard", renderWizard(ctx.cwd, state), next?.apiUsage ? "warning" : "info");
        return;
      }

      if (subcommand === "reset") {
        await rm(wizardDir(ctx.cwd), { recursive: true, force: true });
        ctx.ui.setWidget("tinker-monitor", undefined);
        sendReport("Fine-tune wizard", "Reset `.tinker-pi/`. Run `/tinker improve data.csv --budget demo` or `/tinker demo` to begin again.", "success");
        return;
      }

      if (subcommand === "setup") {
        const checks: string[] = [];
        checks.push(process.env.TINKER_API_KEY ? "✅ `TINKER_API_KEY` is set." : "❌ `TINKER_API_KEY` is not set. Get one from https://tinker-console.thinkingmachines.ai and export it.");
        checks.push((await commandExists("python3")) ? "✅ `python3` found." : "❌ `python3` not found.");
        try {
          const { stdout } = await execFile("python3", ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], { timeout: 20_000 });
          const [major, minor] = stdout.trim().split(".").map(Number);
          checks.push(major > 3 || (major === 3 && minor >= 11) ? `✅ Python ${stdout.trim()} supports Inkling.` : `❌ Inkling requires Python 3.11+ (found ${stdout.trim()}).`);
        } catch {}
        checks.push((await commandExists("uv")) ? "✅ `uv` found." : "⚠️ `uv` not found. You can still use pip, but uv is recommended.");
        checks.push((await commandExists("tinker")) ? "✅ `tinker` CLI found." : "⚠️ `tinker` CLI not found on PATH.");
        try {
          const { stdout, stderr } = await execFile("python3", ["-c", "import importlib.metadata as m; print(m.version('tinker'))"], { timeout: 20_000 });
          const version = stdout.trim() || stderr.trim();
          const [major, minor] = version.split(".").map(Number);
          checks.push(major > 0 || minor >= 23 ? `✅ Tinker SDK ${version} supports Inkling.` : `❌ Inkling requires Tinker SDK 0.23+ (found ${version}).`);
        } catch {
          checks.push("⚠️ Python cannot import `tinker`. Install with: `uv pip install -U tinker-cookbook`.");
        }
        try {
          const { stdout } = await execFile("python3", ["-c", "import tinker_cookbook; print('ok')"], { timeout: 20_000 });
          checks.push(`✅ Python can import \`tinker_cookbook\` (${stdout.trim()}).`);
        } catch {
          checks.push("⚠️ Python cannot import `tinker_cookbook`. Install with: `uv pip install -U tinker-cookbook`.");
        }
        try {
          await execFile("python3", ["-c", "import tml_renderers; print('ok')"], { timeout: 20_000 });
          checks.push("✅ Python can import `tml_renderers` for Inkling.");
        } catch {
          checks.push("⚠️ Python cannot import `tml_renderers`. Install with: `uv pip install -U tinker-cookbook` (Python 3.11+ required).");
        }
        sendReport("Tinker setup check", checks.join("\n"));
        return;
      }

      if (subcommand === "validate") {
        const { positional, options } = parseOptions(rest);
        const file = positional[0];
        if (!file) {
          sendReport("Tinker validate", "Usage: `/tinker validate data/train.jsonl --model thinkingmachines/Inkling-Small`", "warning");
          return;
        }
        const dataFile = path.resolve(ctx.cwd, file);
        const model = String(options.model ?? DEFAULT_MODEL);
        const maxExamples = Number(options.examples ?? 200);
        const maxLength = Number(options["max-length"] ?? 32768);
        const lightweight = validateDataset(dataFile);
        if (options.quick === true) {
          sendReport("Tinker dataset validation", lightweight, "info");
          return;
        }
        try {
          const python = await validateDatasetWithPython(dataFile, model, maxExamples, maxLength);
          await patchWizardState(ctx.cwd, { validationAt: Date.now(), dataFile, dataFileInput: file, model });
          sendReport("Tinker dataset validation", `${python}\n\n---\n\n${lightweight}`, "info");
        } catch (error) {
          sendReport("Tinker dataset validation", `${lightweight}\n\n## Python-backed validation failed\n${error instanceof Error ? error.message : String(error)}`, "warning");
        }
        return;
      }

      if (subcommand === "eval") {
        const [actionRaw, ...evalRest] = rest;
        const action = actionRaw ?? "help";
        const { positional, options } = parseOptions(evalRest);

        if (action === "help" || action === "--help" || action === "-h") {
          sendReport("Tinker eval", [
            "Commands:",
            "- `/tinker eval init` — create `eval.py` and `data/eval.jsonl`.",
            "- `/tinker eval baseline --model thinkingmachines/Inkling-Small --effort 0.9` — evaluate base Inkling.",
            "- `/tinker eval checkpoint tinker://... --model thinkingmachines/Inkling-Small --effort 0.9` — evaluate a sampler checkpoint at the same effort.",
            "- `/tinker eval compare eval_results/baseline.json eval_results/checkpoint.json` — compare results.",
          ].join("\n"));
          return;
        }

        if (action === "init") {
          const force = options.force === true;
          const files = [
            { rel: "eval.py", content: makeExactEvalScript() },
            { rel: path.join("data", "eval.jsonl"), content: makeExampleEvalJsonl() },
          ];
          const written: string[] = [];
          for (const file of files) {
            const target = path.join(ctx.cwd, file.rel);
            if (existsSync(target) && !force) {
              sendReport("Tinker eval init", `${target} already exists. Re-run with --force to overwrite.`, "warning");
              return;
            }
            await mkdir(path.dirname(target), { recursive: true });
            await writeFile(target, file.content);
            written.push(file.rel);
          }
          sendReport("Tinker eval initialized", [
            `Wrote ${written.map((x) => `\`${x}\``).join(", ")}.`,
            "",
            "Edit `data/eval.jsonl`, then run:",
            "```text",
            "/tinker eval baseline --model thinkingmachines/Inkling-Small --effort 0.9 --yes",
            "/tinker eval checkpoint tinker://.../sampler_weights/... --model thinkingmachines/Inkling-Small --effort 0.9 --yes",
            "/tinker eval compare eval_results/baseline.json eval_results/checkpoint.json",
            "```",
          ].join("\n"), "success");
          return;
        }

        if (action === "baseline" || action === "checkpoint") {
          const script = path.resolve(ctx.cwd, String(options.script ?? "eval.py"));
          if (!existsSync(script)) {
            sendReport("Tinker eval", `Missing \`${script}\`. Run \`/tinker eval init\` first.`, "warning");
            return;
          }
          const model = String(options.model ?? options["base-model"] ?? DEFAULT_MODEL);
          const data = path.resolve(ctx.cwd, String(options.data ?? "data/eval.jsonl"));
          if (!existsSync(data)) {
            sendReport("Tinker eval", `Eval data not found: ${data}`, "error");
            return;
          }
          const checkpoint = action === "checkpoint" ? positional[0] : undefined;
          if (action === "checkpoint" && !checkpoint) {
            sendReport("Tinker eval checkpoint", "Usage: `/tinker eval checkpoint tinker://.../sampler_weights/... --model thinkingmachines/Inkling-Small --effort 0.9`", "warning");
            return;
          }
          const out = path.resolve(ctx.cwd, String(options.out ?? (action === "baseline" ? "eval_results/baseline.json" : `eval_results/${String(checkpoint).split("/").slice(-1)[0] || "checkpoint"}.json`)));
          const ok = options.yes === true || !ctx.hasUI ? true : await ctx.ui.confirm("Run Tinker eval?", "This will sample from Tinker and may incur API usage. Continue?");
          if (!ok) {
            sendReport("Tinker eval", "Cancelled.", "warning");
            return;
          }
          const argsForPython = action === "baseline"
            ? [script, "--base-model", model, "--data", data, "--out", out]
            : [script, "--model-path", String(checkpoint), "--renderer-model", model, "--data", data, "--out", out];
          if (options.limit) argsForPython.push("--limit", String(options.limit));
          if (options["max-tokens"]) argsForPython.push("--max-tokens", String(options["max-tokens"]));
          if (options.temperature) argsForPython.push("--temperature", String(options.temperature));
          if (options.effort) argsForPython.push("--effort", String(options.effort));
          try {
            ctx.ui.setStatus("tinker", `Tinker eval: ${action}`);
            const { stdout, stderr } = await execFile("python3", argsForPython, {
              cwd: ctx.cwd,
              timeout: Number(options.timeout ?? 1_800_000),
              maxBuffer: 12 * 1024 * 1024,
            });
            await patchWizardState(ctx.cwd, action === "baseline" ? { baselineResult: out, model } : { candidateResult: out, checkpointPath: String(checkpoint), model });
            sendReport("Tinker eval completed", [
              formatEvalSummary(out),
              `\n## Output tail\n\`\`\`text\n${`${stdout}\n${stderr}`.trim().split(/\n/).slice(-60).join("\n")}\n\`\`\``,
            ].join("\n"), "success");
          } catch (error: any) {
            sendReport("Tinker eval failed", [
              error?.message ?? String(error),
              error?.stdout ? `\n## stdout\n\`\`\`text\n${String(error.stdout).split(/\n/).slice(-80).join("\n")}\n\`\`\`` : "",
              error?.stderr ? `\n## stderr\n\`\`\`text\n${String(error.stderr).split(/\n/).slice(-80).join("\n")}\n\`\`\`` : "",
            ].filter(Boolean).join("\n"), "error");
          } finally {
            ctx.ui.setStatus("tinker", undefined);
          }
          return;
        }

        if (action === "compare") {
          const baseline = positional[0] ? path.resolve(ctx.cwd, positional[0]) : path.resolve(ctx.cwd, "eval_results/baseline.json");
          const candidate = positional[1] ? path.resolve(ctx.cwd, positional[1]) : "";
          if (!candidate) {
            sendReport("Tinker eval compare", "Usage: `/tinker eval compare eval_results/baseline.json eval_results/checkpoint.json`", "warning");
            return;
          }
          try {
            sendReport("Tinker eval comparison", compareEvalSummaries(baseline, candidate), "info");
          } catch (error) {
            sendReport("Tinker eval comparison", `Could not compare eval results: ${error instanceof Error ? error.message : String(error)}`, "error");
          }
          return;
        }

        sendReport("Tinker eval", `Unknown eval action: ${action}. Run \`/tinker eval help\`.`, "warning");
        return;
      }

      if (subcommand === "init") {
        const { positional, options } = parseOptions(rest);
        let dataFileArg = positional[0];
        let model = String(options.model ?? "");
        let successMetric = String(options.metric ?? "");
        if (ctx.hasUI && !dataFileArg) {
          dataFileArg = (await ctx.ui.input("Training JSONL path", "data/train.jsonl"))?.trim();
        }
        if (ctx.hasUI && !model) {
          const modelChoice = await ctx.ui.select("Choose a starting model", starterModelChoices());
          if (modelChoice === "custom") model = (await ctx.ui.input("Tinker model id", DEFAULT_MODEL))?.trim() || DEFAULT_MODEL;
          else model = modelChoice?.split(" — ")[0] || DEFAULT_MODEL;
        }
        if (ctx.hasUI && !successMetric) {
          successMetric = (await ctx.ui.input("What should improve?", "e.g. held-out exact match, support response quality, benchmark score"))?.trim() || "Define before scaling beyond a smoke test.";
        }
        if (!dataFileArg) {
          sendReport("Tinker init", "Usage: `/tinker init data/train.jsonl --model thinkingmachines/Inkling-Small --metric 'held-out accuracy'`", "warning");
          return;
        }
        const dataFile = path.resolve(ctx.cwd, dataFileArg);
        if (!existsSync(dataFile)) {
          sendReport("Tinker init", `Data file not found: ${dataFile}`, "error");
          return;
        }
        model = model || DEFAULT_MODEL;
        const force = options.force === true;
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const logPath = String(options.log ?? `logs/sft-${stamp}`);
        const maxSteps = String(options.steps ?? "20");
        const batchSize = String(options["batch-size"] ?? "8");
        const learningRate = String(options.lr ?? options["learning-rate"] ?? "2e-4");
        const testSize = String(options["test-size"] ?? "0");
        const maxLength = String(options["max-length"] ?? "32768");
        const files = [
          { rel: "README.md", content: makeProjectReadme({ model, dataFile, logPath, successMetric }) },
          { rel: "train_sft.py", content: makeSftScript({ dataFile, model, logPath, maxSteps, batchSize, learningRate, testSize, maxLength }) },
          { rel: "eval_checkpoint.py", content: makeEvalScript() },
          { rel: "tinker.yaml", content: `task: sft\ndata: ${dataFile}\nmodel: ${model}\nlog_path: ${logPath}\nmax_steps: ${maxSteps}\nbatch_size: ${batchSize}\nlearning_rate: ${learningRate}\nsuccess_metric: ${successMetric}\n` },
          { rel: path.join("notes", "plan.md"), content: `# Tinker SFT plan\n\n## Goal\n\nFine-tune ${model} on ${dataFile}.\n\n## Success metric\n\n${successMetric || "TODO"}\n\n## First checks\n\n- [ ] /tinker validate ${dataFileArg} --model ${model}\n- [ ] python train_sft.py max_steps=2\n- [ ] inspect metrics and decoded examples\n- [ ] define baseline eval before scale-up\n` },
        ];
        const written: string[] = [];
        for (const file of files) {
          const target = path.join(ctx.cwd, file.rel);
          if (existsSync(target) && !force) {
            sendReport("Tinker init", `${target} already exists. Re-run with --force to overwrite.`, "warning");
            return;
          }
          await mkdir(path.dirname(target), { recursive: true });
          await writeFile(target, file.content);
          written.push(file.rel);
        }
        sendReport("Tinker project initialized", [
          `Wrote ${written.map((x) => `\`${x}\``).join(", ")}.`,
          "",
          "Recommended next steps:",
          "```text",
          `/tinker validate ${dataFileArg} --model ${model}`,
          "/tinker smoke train_sft.py --yes",
          `/tinker monitor ${logPath}`,
          "```",
        ].join("\n"), "success");
        return;
      }

      if (subcommand === "sft") {
        const { positional, options } = parseOptions(rest);
        const dataFileArg = positional[0];
        if (!dataFileArg) {
          sendReport("Tinker SFT scaffold", "Usage: `/tinker sft data/train.jsonl --model thinkingmachines/Inkling-Small --steps 20`", "warning");
          return;
        }
        const dataFile = path.resolve(ctx.cwd, dataFileArg);
        if (!existsSync(dataFile)) {
          sendReport("Tinker SFT scaffold", `Data file not found: ${dataFile}`, "error");
          return;
        }
        const force = options.force === true;
        const model = String(options.model ?? DEFAULT_MODEL);
        const maxSteps = String(options.steps ?? options.max_steps ?? "20");
        const batchSize = String(options["batch-size"] ?? "8");
        const learningRate = String(options.lr ?? options["learning-rate"] ?? "2e-4");
        const testSize = String(options["test-size"] ?? "0");
        const maxLength = String(options["max-length"] ?? "32768");
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const logPath = String(options.log ?? `logs/sft-${stamp}`);

        const files = [
          { rel: "train_sft.py", content: makeSftScript({ dataFile, model, logPath, maxSteps, batchSize, learningRate, testSize, maxLength }) },
          { rel: "eval_checkpoint.py", content: makeEvalScript() },
          { rel: "tinker.yaml", content: `task: sft\ndata: ${dataFile}\nmodel: ${model}\nlog_path: ${logPath}\nmax_steps: ${maxSteps}\nbatch_size: ${batchSize}\nlearning_rate: ${learningRate}\n` },
          { rel: path.join("notes", "plan.md"), content: `# Tinker SFT plan\n\n## Goal\n\nFine-tune ${model} on ${dataFile}.\n\n## Baseline eval\n\nTODO: define success metric before scaling the run.\n\n## Smoke test\n\nRun a tiny job first:\n\n\`\`\`bash\npython train_sft.py max_steps=2\n\`\`\`\n\nThen inspect metrics and decoded samples before increasing steps.\n` },
        ];

        const written: string[] = [];
        for (const file of files) {
          const target = path.join(ctx.cwd, file.rel);
          if (existsSync(target) && !force) {
            sendReport("Tinker SFT scaffold", `${target} already exists. Re-run with --force to overwrite.`, "warning");
            return;
          }
          await mkdir(path.dirname(target), { recursive: true });
          await writeFile(target, file.content);
          written.push(file.rel);
        }

        sendReport("Tinker SFT scaffold created", [
          `Wrote ${written.map((x) => `\`${x}\``).join(", ")}.`,
          "",
          "Next commands:",
          "```bash",
          COOKBOOK_INSTALL,
          "python train_sft.py max_steps=2",
          "# inspect logs, then scale up:",
          `python train_sft.py max_steps=${maxSteps}`,
          "```",
          "",
          "Tip: use `/skill:tinker-research` for experiment planning/monitoring before spending real compute.",
        ].join("\n"), "success");
        return;
      }

      if (subcommand === "smoke") {
        const { positional, options } = parseOptions(rest);
        const script = path.resolve(ctx.cwd, positional[0] ?? "train_sft.py");
        if (!existsSync(script)) {
          sendReport("Tinker smoke test", `Script not found: ${script}`, "error");
          return;
        }
        const ok = options.yes === true || !ctx.hasUI ? true : await ctx.ui.confirm("Run Tinker smoke test?", "This may create a real Tinker training client and incur small API usage. Run `python train_sft.py max_steps=2`?");
        if (!ok) {
          sendReport("Tinker smoke test", "Cancelled.", "warning");
          return;
        }
        try {
          ctx.ui.setStatus("tinker", "Tinker smoke running");
          const { stdout, stderr } = await execFile("python3", [script, "max_steps=2"], {
            cwd: ctx.cwd,
            timeout: Number(options.timeout ?? 1_800_000),
            maxBuffer: 10 * 1024 * 1024,
          });
          const combined = `${stdout}\n${stderr}`.trim();
          const logMatch = combined.match(/log_path='([^']+)'|log_path=([^,\s)]+)/);
          const logDir = logMatch ? path.resolve(ctx.cwd, (logMatch[1] ?? logMatch[2] ?? "").replace(/^['\"]|['\"]$/g, "")) : undefined;
          const metrics = logDir ? latestMetrics(path.join(logDir, "metrics.jsonl")) : undefined;
          await patchWizardState(ctx.cwd, { smokeAt: Date.now(), smokeLogDir: logDir, logPath: logDir ? rel(ctx.cwd, logDir) : undefined });
          sendReport("Tinker smoke test passed", [
            "The 2-step command completed.",
            metrics ? `\n## Latest metrics\n${metrics}` : "",
            `\n## Output tail\n\`\`\`text\n${combined.split(/\n/).slice(-80).join("\n")}\n\`\`\``,
          ].filter(Boolean).join("\n"), "success");
        } catch (error: any) {
          sendReport("Tinker smoke test failed", [
            error?.message ?? String(error),
            error?.stdout ? `\n## stdout\n\`\`\`text\n${String(error.stdout).split(/\n/).slice(-80).join("\n")}\n\`\`\`` : "",
            error?.stderr ? `\n## stderr\n\`\`\`text\n${String(error.stderr).split(/\n/).slice(-80).join("\n")}\n\`\`\`` : "",
            "\nUse `/skill:tinker-debug` with the error output if this is not obvious.",
          ].filter(Boolean).join("\n"), "error");
        } finally {
          ctx.ui.setStatus("tinker", undefined);
        }
        return;
      }

      if (subcommand === "monitor") {
        const logDirArg = rest[0];
        if (!logDirArg) {
          sendReport("Tinker monitor", "Usage: `/tinker monitor logs/my-run` or `/tinker monitor --stop`", "warning");
          return;
        }
        if (logDirArg === "--stop" || logDirArg === "stop") {
          if (monitorInterval) clearInterval(monitorInterval);
          monitorInterval = undefined;
          ctx.ui.setWidget("tinker-monitor", undefined);
          ctx.ui.setStatus("tinker", undefined);
          sendReport("Tinker monitor", "Stopped.", "success");
          return;
        }
        const logDir = path.resolve(ctx.cwd, logDirArg);
        if (monitorInterval) clearInterval(monitorInterval);
        const update = () => {
          const lines = monitorSummary(logDir);
          ctx.ui.setWidget("tinker-monitor", lines);
          ctx.ui.setStatus("tinker", `Tinker: ${lines[1] ?? "monitoring"}`);
        };
        update();
        monitorInterval = setInterval(update, 5000);
        sendReport("Tinker monitor", `Monitoring \`${logDir}\`. Use \`/tinker monitor --stop\` to clear the widget.`, "success");
        return;
      }

      if (subcommand === "checkpoints") {
        const logDirArg = rest[0];
        if (!logDirArg) {
          sendReport("Tinker checkpoints", "Usage: `/tinker checkpoints logs/my-run`", "warning");
          return;
        }
        const logDir = path.resolve(ctx.cwd, logDirArg);
        let checkpoints: CheckpointRecord[] = [];
        try {
          checkpoints = readCheckpoints(logDir).filter((c) => c.sampler_path || c.state_path);
        } catch (error) {
          sendReport("Tinker checkpoints", `Could not read checkpoints.jsonl: ${error instanceof Error ? error.message : String(error)}`, "error");
          return;
        }
        if (checkpoints.length === 0) {
          sendReport("Tinker checkpoints", `No checkpoints found in \`${path.join(logDir, "checkpoints.jsonl")}\`.`, "warning");
          return;
        }
        const lines = checkpoints.map((c, i) => `${i + 1}. ${c.name ?? "checkpoint"}${c.final ? " (final)" : ""}${c.batch !== undefined ? ` batch=${c.batch}` : ""}\n   sampler: ${c.sampler_path ?? "none"}\n   state: ${c.state_path ?? "none"}`);
        const samplerCheckpoints = checkpoints.filter((c) => c.sampler_path);
        const latestSampler = samplerCheckpoints.slice(-1)[0]?.sampler_path;
        if (latestSampler) await patchWizardState(ctx.cwd, { checkpointPath: latestSampler, logPath: rel(ctx.cwd, logDir) });
        if (ctx.hasUI && samplerCheckpoints.length > 0) {
          const choice = await ctx.ui.select("Register sampler checkpoint as Pi model?", ["just list", ...samplerCheckpoints.map((c) => `${c.name ?? "checkpoint"} — ${c.sampler_path}`)]);
          if (choice && choice !== "just list") {
            const record = samplerCheckpoints.find((c) => choice.endsWith(String(c.sampler_path)));
            if (record?.sampler_path) {
              const alias = `${path.basename(logDir)}-${record.name ?? "checkpoint"}`.replace(/[^a-zA-Z0-9._-]+/g, "-");
              state.checkpoints = state.checkpoints.filter((m) => m.id !== record.sampler_path && m.name !== alias);
              const wizard = await readWizardState(ctx.cwd);
              state.checkpoints.push(checkpointRegistration({ id: record.sampler_path, name: alias, baseModel: wizard?.model }));
              await saveState(state);
              await patchWizardState(ctx.cwd, { checkpointPath: record.sampler_path, registeredModel: alias });
              registerTinkerProvider();
              sendReport("Tinker checkpoint registered", `Registered \`${alias}\` for \`${record.sampler_path}\`. Use \`/model\` to select it.`, "success");
              return;
            }
          }
        }
        sendReport("Tinker checkpoints", lines.join("\n\n"));
        return;
      }

      if (subcommand === "status") {
        const logDir = rest[0] ? path.resolve(ctx.cwd, rest[0]) : undefined;
        const sections: string[] = [];
        if (await commandExists("tinker")) {
          try {
            const { stdout, stderr } = await execFile("tinker", ["run", "list", "--limit", "10"], { timeout: 30_000 });
            sections.push(`## Recent Tinker runs\n\n\`\`\`text\n${(stdout || stderr).trim()}\n\`\`\``);
          } catch (error: any) {
            sections.push(`## Recent Tinker runs\n\nCould not run \`tinker run list\`: ${error?.message ?? String(error)}`);
          }
        } else {
          sections.push("## Recent Tinker runs\n\n`tinker` CLI not found on PATH.");
        }
        if (logDir) {
          const metrics = latestMetrics(path.join(logDir, "metrics.jsonl"));
          sections.push(`## Latest metrics from ${logDir}\n\n${metrics ?? "No metrics.jsonl found."}`);
        }
        sendReport("Tinker status", sections.join("\n\n"));
        return;
      }

      if (subcommand === "use") {
        const { positional, options } = parseOptions(rest);
        if (options.list === true) {
          const lines = state.checkpoints.length
            ? state.checkpoints.map((m) => `- ${m.name}: \`${m.id}\` (${m.contextWindow} ctx, max ${m.maxTokens})`)
            : ["No Tinker checkpoints registered yet."];
          sendReport("Registered Tinker checkpoint models", lines.join("\n"));
          return;
        }
        if (options.remove) {
          const key = String(options.remove);
          const before = state.checkpoints.length;
          state.checkpoints = state.checkpoints.filter((m) => m.id !== key && m.name !== key);
          await saveState(state);
          registerTinkerProvider();
          sendReport("Tinker checkpoint removed", before === state.checkpoints.length ? `No checkpoint matched ${key}.` : `Removed ${key}.`, "success");
          return;
        }
        const checkpoint = positional[0];
        if (!checkpoint || !checkpoint.startsWith("tinker://")) {
          sendReport("Register Tinker checkpoint", "Usage: `/tinker use tinker://.../sampler_weights/... [alias]`", "warning");
          return;
        }
        const alias = String(options.alias ?? positional[1] ?? `tinker-${state.checkpoints.length + 1}`);
        const baseModel = String(options["base-model"] ?? options.model ?? "") || undefined;
        const inkling = isInklingModel(baseModel);
        const contextWindow = Number(options.context ?? (inkling ? 65_536 : 32_768));
        const maxTokens = Number(options["max-tokens"] ?? (inkling ? 16_384 : 4_096));
        state.checkpoints = state.checkpoints.filter((m) => m.id !== checkpoint && m.name !== alias);
        state.checkpoints.push(checkpointRegistration({ id: checkpoint, name: alias, baseModel, contextWindow, maxTokens }));
        await saveState(state);
        await patchWizardState(ctx.cwd, { checkpointPath: checkpoint, registeredModel: alias });
        registerTinkerProvider();
        sendReport("Tinker checkpoint registered", [
          `Registered \`${alias}\` as provider/model \`tinker/${checkpoint}\`.`,
          "",
          "Use `/model` to select it. Pi uses Tinker's beta Anthropic-compatible endpoint for tool use, image input, streaming thinking, and Inkling effort controls; it is still intended for testing rather than production serving.",
          "",
          `Saved registrations in \`${STATE_PATH}\`.`,
        ].join("\n"), "success");
        return;
      }

      sendReport("Unknown Tinker command", `Unknown subcommand: ${subcommand}\n\nRun \`/tinker help\` for usage.`, "warning");
    },
  });
}
