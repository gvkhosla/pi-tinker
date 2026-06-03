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
  const checkpointPath = path.join(logDir, "checkpoints.jsonl");
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

async function validateDatasetWithPython(filePath: string, model: string, maxExamples: number): Promise<string> {
  const code = String.raw`
import json, sys, statistics, traceback
file_path, model_name, max_examples_s = sys.argv[1:4]
max_examples = int(max_examples_s)
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
rows = 0
bad = []
lengths = []
trainable = []
previews = []
roles = {}
try:
    with open(file_path) as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                obj = json.loads(line)
                messages = obj.get("messages")
                if not isinstance(messages, list):
                    bad.append(f"line {idx}: missing messages[]")
                    continue
                for msg in messages:
                    roles[msg.get("role", "<missing>")] = roles.get(msg.get("role", "<missing>"), 0) + 1
                if len(lengths) < max_examples:
                    model_input, weights = renderer.build_supervised_example(
                        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
                    )
                    toks = model_input.to_ints() if hasattr(model_input, "to_ints") else []
                    w = [float(x) for x in list(weights)]
                    lengths.append(len(toks))
                    trainable.append(sum(1 for x in w if x > 0))
                    if len(previews) < 3:
                        previews.append(tokenizer.decode(toks[: min(len(toks), 400)]))
            except Exception as e:
                bad.append(f"line {idx}: {type(e).__name__}: {e}")
except Exception as e:
    print(json.dumps({"ok": False, "stage": "file", "error": f"{type(e).__name__}: {e}"}))
    raise SystemExit(0)
def stats(xs):
    if not xs:
        return {"count": 0}
    return {"count": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs), "mean": sum(xs) / len(xs)}
print(json.dumps({
    "ok": len(bad) == 0,
    "stage": "done",
    "model": model_name,
    "renderer": renderer_name,
    "rows": rows,
    "roles": roles,
    "checked_examples": len(lengths),
    "token_lengths": stats(lengths),
    "trainable_tokens": stats(trainable),
    "bad": bad[:50],
    "bad_count": len(bad),
    "previews": previews,
}, ensure_ascii=False))
`;
  const { stdout } = await execFile("python3", ["-c", code, filePath, model, String(maxExamples)], {
    timeout: 180_000,
    maxBuffer: 8 * 1024 * 1024,
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
      "Falling back to the lightweight JSONL checks is still useful, but renderer/token-mask validation requires Python dependencies.",
    ].join("\n");
  }
  const lenStats = result.token_lengths ?? {};
  const trainStats = result.trainable_tokens ?? {};
  const warnings: string[] = [];
  if ((trainStats.max ?? 0) === 0) warnings.push("No trainable assistant tokens found in checked examples — training would be a no-op or data masking is wrong.");
  if ((lenStats.max ?? 0) > 32768) warnings.push(`Some checked examples exceed 32k tokens (max ${lenStats.max}); set --max-length carefully or inspect truncation.`);
  if (result.rows < 20) warnings.push("Very small dataset; fine for smoke tests, weak for real fine-tuning.");
  return [
    `# Renderer/token validation: ${filePath}`,
    `- Model: \`${result.model}\``,
    `- Recommended renderer: \`${result.renderer}\``,
    `- JSONL rows: ${result.rows}`,
    `- Checked examples: ${result.checked_examples}`,
    `- Roles: ${Object.entries(result.roles ?? {}).map(([k, v]) => `${k}=${v}`).join(", ") || "none"}`,
    `- Token lengths: min=${lenStats.min ?? "n/a"}, median=${lenStats.median ?? "n/a"}, max=${lenStats.max ?? "n/a"}, mean=${typeof lenStats.mean === "number" ? lenStats.mean.toFixed(1) : "n/a"}`,
    `- Trainable tokens: min=${trainStats.min ?? "n/a"}, median=${trainStats.median ?? "n/a"}, max=${trainStats.max ?? "n/a"}, mean=${typeof trainStats.mean === "number" ? trainStats.mean.toFixed(1) : "n/a"}`,
    result.bad_count ? `\n## Issues\n${(result.bad ?? []).map((x: string) => `- ${x}`).join("\n")}${result.bad_count > 50 ? `\n- … ${result.bad_count - 50} more` : ""}` : "\n## Issues\nNo renderer/tokenization issues found in checked examples.",
    warnings.length ? `\n## Warnings\n${warnings.map((x) => `- ${x}`).join("\n")}` : "",
    `\n## Decoded previews\n${(result.previews ?? []).map((x: string, i: number) => `### Preview ${i + 1}\n\n\`\`\`text\n${x.slice(0, 1800)}\n\`\`\``).join("\n\n")}`,
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
        const maxExamples = Number(options.examples ?? 50);
        const lightweight = validateDataset(dataFile);
        if (options.quick === true) {
          sendReport("Tinker dataset validation", lightweight, "info");
          return;
        }
        try {
          const python = await validateDatasetWithPython(dataFile, model, maxExamples);
          sendReport("Tinker dataset validation", `${python}\n\n---\n\n${lightweight}`, "info");
        } catch (error) {
          sendReport("Tinker dataset validation", `${lightweight}\n\n## Python-backed validation failed\n${error instanceof Error ? error.message : String(error)}`, "warning");
        }
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
