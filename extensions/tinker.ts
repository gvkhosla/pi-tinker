import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Box, Text } from "@earendil-works/pi-tui";
import { execFile as execFileCb } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFile = promisify(execFileCb);

const TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1";
const STATE_PATH = path.join(os.homedir(), ".pi", "agent", "tinker-checkpoints.json");
const MESSAGE_TYPE = "tinker-report";

type ReportLevel = "info" | "success" | "warning" | "error";

type CheckpointModel = {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  addedAt: number;
};

type TinkerState = {
  checkpoints: CheckpointModel[];
};

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
    return [
      `# Python-backed validation could not run`,
      `- Stage: ${result.stage ?? "unknown"}`,
      `- Error: ${result.error ?? "unknown"}`,
      "",
      "Install/upgrade dependencies with:",
      "```bash",
      "uv pip install -U tinker-cookbook",
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
  return `import asyncio\nimport sys\n\nimport chz\n\nfrom tinker_cookbook import cli_utils, model_info\nfrom tinker_cookbook.renderers import TrainOnWhat\nfrom tinker_cookbook.supervised import train\nfrom tinker_cookbook.supervised.data import FromConversationFileBuilder\nfrom tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig\n\n\ndef build_config_blueprint() -> chz.Blueprint[train.Config]:\n    model_name = ${JSON.stringify(options.model)}\n    renderer_name = model_info.get_recommended_renderer_name(model_name)\n\n    common_config = ChatDatasetBuilderCommonConfig(\n        model_name_for_tokenizer=model_name,\n        renderer_name=renderer_name,\n        max_length=${options.maxLength},\n        batch_size=${options.batchSize},\n        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,\n    )\n\n    dataset = FromConversationFileBuilder(\n        common_config=common_config,\n        file_path=${JSON.stringify(options.dataFile)},\n        test_size=${options.testSize},\n        shuffle_seed=0,\n    )\n\n    return chz.Blueprint(train.Config).apply({\n        "log_path": ${JSON.stringify(options.logPath)},\n        "model_name": model_name,\n        "recipe_name": "pi_tinker_sft",\n        "renderer_name": renderer_name,\n        "dataset_builder": dataset,\n        "learning_rate": ${options.learningRate},\n        "lr_schedule": "linear",\n        "num_epochs": 1,\n        "lora_rank": 32,\n        "save_every": 20,\n        "eval_every": 10,\n        "max_steps": ${options.maxSteps},\n    })\n\n\ndef main(config: train.Config):\n    print("Resolved Tinker SFT config:")\n    print(config)\n    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")\n    asyncio.run(train.main(config))\n\n\nif __name__ == "__main__":\n    blueprint = build_config_blueprint()\n    blueprint.make_from_argv(sys.argv[1:])\n    main(blueprint.make())\n`;
}

function makeEvalScript() {
  return `"""\nOptional quick checkpoint smoke test.\n\nUsage:\n  python eval_checkpoint.py 'tinker://.../sampler_weights/...'\n"""\nimport asyncio\nimport sys\n\nimport tinker\n\n\nasync def main(model_path: str):\n    svc = tinker.ServiceClient()\n    sc = await svc.create_sampling_client_async(model_path=model_path)\n    tok = sc.get_tokenizer()\n    prompt = tinker.ModelInput.from_ints(tok.encode("The best way to test a fine-tuned model is"))\n    result = await sc.sample_async(\n        prompt=prompt,\n        num_samples=1,\n        sampling_params=tinker.SamplingParams(max_tokens=80, temperature=0.7),\n    )\n    seq = result.sequences[0] if hasattr(result, "sequences") else result.samples[0]\n    print(tok.decode(seq.tokens))\n\n\nif __name__ == "__main__":\n    if len(sys.argv) != 2:\n        raise SystemExit("Usage: python eval_checkpoint.py 'tinker://.../sampler_weights/...'" )\n    asyncio.run(main(sys.argv[1]))\n`;
}

function makeProjectReadme(options: { model: string; dataFile: string; logPath: string; successMetric: string }) {
  return `# Tinker fine-tuning project\n\nThis project was scaffolded by \`pi-tinker\`. The important files are normal editable Python, not hidden framework state.\n\n## Goal\n\nFine-tune \`${options.model}\` on:\n\n\`${options.dataFile}\`\n\n## Success metric\n\n${options.successMetric || "Define this before scaling beyond a smoke test."}\n\n## Smoke test\n\n\`\`\`bash\nuv pip install tinker-cookbook\npython train_sft.py max_steps=2\n\`\`\`\n\n## Monitor\n\nInside Pi:\n\n\`\`\`text\n/tinker monitor ${options.logPath}\n/tinker status ${options.logPath}\n/tinker checkpoints ${options.logPath}\n\`\`\`\n\n## Scale up\n\nOnly scale after checking:\n\n- JSONL validation passed\n- renderer/token validation passed\n- smoke test produced metrics\n- decoded examples look correct\n- success metric/eval is defined\n\n\`\`\`bash\npython train_sft.py max_steps=100\n\`\`\`\n\n## Chat with a checkpoint in Pi\n\nAfter a sampler checkpoint appears in \`checkpoints.jsonl\`:\n\n\`\`\`text\n/tinker checkpoints ${options.logPath}\n/model\n\`\`\`\n`;
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

function makeExactEvalScript() {
  return `"""Simple editable eval for Tinker checkpoints.

Input JSONL format:
  {"messages": [{"role": "user", "content": "..."}], "expected": "...", "match": "contains"}

Match modes:
  - exact: normalized output equals normalized expected
  - contains: normalized expected appears in normalized output
  - prefix: normalized output starts with normalized expected

Examples:
  python eval.py --base-model Qwen/Qwen3.5-9B-Base --data data/eval.jsonl --out eval_results/baseline.json
  python eval.py --model-path 'tinker://.../sampler_weights/...' --base-model Qwen/Qwen3.5-9B-Base --data data/eval.jsonl --out eval_results/step-20.json
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
from tinker_cookbook.renderers import get_renderer


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
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    args = parser.parse_args()

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
        prompt = renderer.build_generation_prompt(messages, role="assistant")
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
        output = tokenizer.decode(sequence.tokens)
        is_correct = score_output(output, expected, match)
        correct += int(is_correct)
        results.append({
            "index": index,
            "correct": is_correct,
            "match": match,
            "expected": expected,
            "output": output,
            "messages": messages,
        })
        print(f"[{index}/{len(rows)}] {'✓' if is_correct else '✗'} expected={expected!r} output={output[:120]!r}")

    summary = {
        "target": target_name,
        "renderer_model": renderer_model,
        "renderer": renderer_name,
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
    wins.length ? `\n## Wins\n${wins.slice(0, 10).map((x) => `- ${x}`).join("\n")}` : "\n## Wins\nNone recorded.",
    regressions.length ? `\n## Regressions\n${regressions.slice(0, 10).map((x) => `- ${x}`).join("\n")}` : "\n## Regressions\nNone recorded.",
  ].join("\n");
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
  }

  function registerTinkerProvider() {
    if (state.checkpoints.length === 0) {
      pi.unregisterProvider("tinker");
      return;
    }
    pi.registerProvider("tinker", {
      name: "Tinker",
      baseUrl: TINKER_OAI_BASE_URL,
      apiKey: "$TINKER_API_KEY",
      authHeader: true,
      api: "openai-completions",
      models: state.checkpoints.map((checkpoint) => ({
        id: checkpoint.id,
        name: checkpoint.name,
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: checkpoint.contextWindow,
        maxTokens: checkpoint.maxTokens,
        compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens" },
      })),
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
    description: "Tinker fine-tuning helper: setup, init, validate, smoke, monitor, checkpoints, use",
    handler: async (args, ctx) => {
      const [subcommandRaw, ...rest] = shellSplit(args);
      const subcommand = subcommandRaw ?? "help";

      if (subcommand === "help" || subcommand === "--help" || subcommand === "-h") {
        sendReport("Tinker helper", [
          "Commands:",
          "- `/tinker setup` — check API key, Python, uv/pip, Tinker SDK.",
          "- `/tinker init` — guided SFT project wizard (or `/tinker init data/train.jsonl`).",
          "- `/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base` — JSONL + renderer/token-mask validation.",
          "- `/tinker eval init|baseline|checkpoint|compare` — create and run simple evals before/after training.",
          "- `/tinker sft data/train.jsonl --model Qwen/Qwen3.5-9B-Base --steps 20` — scaffold editable SFT files.",
          "- `/tinker smoke [train_sft.py]` — run a 2-step smoke test and summarize logs.",
          "- `/tinker monitor <log_dir>` — pin a live metrics widget above the editor.",
          "- `/tinker checkpoints <log_dir>` — list/pick sampler checkpoints and register them as Pi models.",
          "- `/tinker status [log_dir]` — show `tinker run list` and latest metrics if present.",
          "- `/tinker use tinker://.../sampler_weights/... [alias]` — register checkpoint as a Pi model via Tinker OpenAI-compatible inference.",
          "- `/tinker use --list` — list registered checkpoint models.",
          "- `/skill:tinker-research ...` — load the Tinker research workflow.",
          "- `/skill:tinker-debug ...` — load Tinker debugging triage.",
        ].join("\n"));
        return;
      }

      if (subcommand === "setup") {
        const checks: string[] = [];
        checks.push(process.env.TINKER_API_KEY ? "✅ `TINKER_API_KEY` is set." : "❌ `TINKER_API_KEY` is not set. Get one from https://tinker-console.thinkingmachines.ai and export it.");
        checks.push((await commandExists("python3")) ? "✅ `python3` found." : "❌ `python3` not found.");
        checks.push((await commandExists("uv")) ? "✅ `uv` found." : "⚠️ `uv` not found. You can still use pip, but uv is recommended.");
        checks.push((await commandExists("tinker")) ? "✅ `tinker` CLI found." : "⚠️ `tinker` CLI not found on PATH.");
        try {
          const { stdout, stderr } = await execFile("python3", ["-c", "import tinker; print(getattr(tinker, '__version__', 'unknown'))"], { timeout: 20_000 });
          checks.push(`✅ Python can import \`tinker\` (${stdout.trim() || stderr.trim()}).`);
        } catch {
          checks.push("⚠️ Python cannot import `tinker`. Install with: `uv pip install tinker-cookbook` (or `python3 -m pip install tinker-cookbook`).");
        }
        try {
          const { stdout } = await execFile("python3", ["-c", "import tinker_cookbook; print('ok')"], { timeout: 20_000 });
          checks.push(`✅ Python can import \`tinker_cookbook\` (${stdout.trim()}).`);
        } catch {
          checks.push("⚠️ Python cannot import `tinker_cookbook`. Install with: `uv pip install tinker-cookbook`.");
        }
        sendReport("Tinker setup check", checks.join("\n"));
        return;
      }

      if (subcommand === "validate") {
        const { positional, options } = parseOptions(rest);
        const file = positional[0];
        if (!file) {
          sendReport("Tinker validate", "Usage: `/tinker validate data/train.jsonl --model Qwen/Qwen3.5-9B-Base`", "warning");
          return;
        }
        const dataFile = path.resolve(ctx.cwd, file);
        const model = String(options.model ?? "Qwen/Qwen3.5-9B-Base");
        const maxExamples = Number(options.examples ?? 200);
        const maxLength = Number(options["max-length"] ?? 32768);
        const lightweight = validateDataset(dataFile);
        if (options.quick === true) {
          sendReport("Tinker dataset validation", lightweight, "info");
          return;
        }
        try {
          const python = await validateDatasetWithPython(dataFile, model, maxExamples, maxLength);
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
            "- `/tinker eval baseline --model Qwen/Qwen3.5-9B-Base` — evaluate the base model.",
            "- `/tinker eval checkpoint tinker://... --model Qwen/Qwen3.5-9B-Base` — evaluate a sampler checkpoint.",
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
            "/tinker eval baseline --model Qwen/Qwen3.5-9B-Base --yes",
            "/tinker eval checkpoint tinker://.../sampler_weights/... --model Qwen/Qwen3.5-9B-Base --yes",
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
          const model = String(options.model ?? options["base-model"] ?? "Qwen/Qwen3.5-9B-Base");
          const data = path.resolve(ctx.cwd, String(options.data ?? "data/eval.jsonl"));
          if (!existsSync(data)) {
            sendReport("Tinker eval", `Eval data not found: ${data}`, "error");
            return;
          }
          const checkpoint = action === "checkpoint" ? positional[0] : undefined;
          if (action === "checkpoint" && !checkpoint) {
            sendReport("Tinker eval checkpoint", "Usage: `/tinker eval checkpoint tinker://.../sampler_weights/... --model Qwen/Qwen3.5-9B-Base`", "warning");
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
          try {
            ctx.ui.setStatus("tinker", `Tinker eval: ${action}`);
            const { stdout, stderr } = await execFile("python3", argsForPython, {
              cwd: ctx.cwd,
              timeout: Number(options.timeout ?? 1_800_000),
              maxBuffer: 12 * 1024 * 1024,
            });
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
          const modelChoice = await ctx.ui.select("Choose a starting model", [
            "Qwen/Qwen3.5-9B-Base — small/base/good default",
            "Qwen/Qwen3.5-35B-A3B-Base — stronger MoE base",
            "meta-llama/Llama-3.2-1B — cheapest smoke tests",
            "custom",
          ]);
          if (modelChoice === "custom") model = (await ctx.ui.input("Tinker model id", "Qwen/Qwen3.5-9B-Base"))?.trim() || "Qwen/Qwen3.5-9B-Base";
          else model = modelChoice?.split(" — ")[0] || "Qwen/Qwen3.5-9B-Base";
        }
        if (ctx.hasUI && !successMetric) {
          successMetric = (await ctx.ui.input("What should improve?", "e.g. held-out exact match, support response quality, benchmark score"))?.trim() || "Define before scaling beyond a smoke test.";
        }
        if (!dataFileArg) {
          sendReport("Tinker init", "Usage: `/tinker init data/train.jsonl --model Qwen/Qwen3.5-9B-Base --metric 'held-out accuracy'`", "warning");
          return;
        }
        const dataFile = path.resolve(ctx.cwd, dataFileArg);
        if (!existsSync(dataFile)) {
          sendReport("Tinker init", `Data file not found: ${dataFile}`, "error");
          return;
        }
        model = model || "Qwen/Qwen3.5-9B-Base";
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
          sendReport("Tinker SFT scaffold", "Usage: `/tinker sft data/train.jsonl --model Qwen/Qwen3.5-9B-Base --steps 20`", "warning");
          return;
        }
        const dataFile = path.resolve(ctx.cwd, dataFileArg);
        if (!existsSync(dataFile)) {
          sendReport("Tinker SFT scaffold", `Data file not found: ${dataFile}`, "error");
          return;
        }
        const force = options.force === true;
        const model = String(options.model ?? "Qwen/Qwen3.5-9B-Base");
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
          "uv pip install tinker-cookbook  # if needed",
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
        if (ctx.hasUI && samplerCheckpoints.length > 0) {
          const choice = await ctx.ui.select("Register sampler checkpoint as Pi model?", ["just list", ...samplerCheckpoints.map((c) => `${c.name ?? "checkpoint"} — ${c.sampler_path}`)]);
          if (choice && choice !== "just list") {
            const record = samplerCheckpoints.find((c) => choice.endsWith(String(c.sampler_path)));
            if (record?.sampler_path) {
              const alias = `${path.basename(logDir)}-${record.name ?? "checkpoint"}`.replace(/[^a-zA-Z0-9._-]+/g, "-");
              state.checkpoints = state.checkpoints.filter((m) => m.id !== record.sampler_path && m.name !== alias);
              state.checkpoints.push({ id: record.sampler_path, name: alias, contextWindow: 32768, maxTokens: 4096, addedAt: Date.now() });
              await saveState(state);
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
        const contextWindow = Number(options.context ?? 32768);
        const maxTokens = Number(options["max-tokens"] ?? 4096);
        state.checkpoints = state.checkpoints.filter((m) => m.id !== checkpoint && m.name !== alias);
        state.checkpoints.push({ id: checkpoint, name: alias, contextWindow, maxTokens, addedAt: Date.now() });
        await saveState(state);
        registerTinkerProvider();
        sendReport("Tinker checkpoint registered", [
          `Registered \`${alias}\` as provider/model \`tinker/${checkpoint}\`.`,
          "",
          "Use `/model` to select it. This uses Tinker's beta OpenAI-compatible inference endpoint, best for quick inspection rather than production serving.",
          "",
          `Saved registrations in \`${STATE_PATH}\`.`,
        ].join("\n"), "success");
        return;
      }

      sendReport("Unknown Tinker command", `Unknown subcommand: ${subcommand}\n\nRun \`/tinker help\` for usage.`, "warning");
    },
  });
}
