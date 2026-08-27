#!/usr/bin/env node
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);

if (argv.length === 0 || argv[0] === "--help" || argv[0] === "-h") {
  console.log(`pi-tinker-agent — non-interactive adapter for coding agents

Usage:
  pi-tinker-agent inkling
  pi-tinker-agent doctor
  pi-tinker-agent validate data/train.jsonl --model thinkingmachines/Inkling
  pi-tinker-agent --cwd /path/to/project new data.csv --goal "better answers"
  pi-tinker-agent --raw 'improve data.csv --goal "better answers" --budget demo'

This adapter runs pi-tinker's /tinker command in Pi print mode. Pi remains the
runtime, while Amp, OpenCode, Claude Code, Codex, Cursor, Copilot, Gemini CLI,
and other agents can invoke the workflow as a normal shell command.
`);
  process.exit(0);
}

let cwd = process.cwd();
const cwdIndex = argv.indexOf("--cwd");
if (cwdIndex !== -1) {
  const requested = argv[cwdIndex + 1];
  if (!requested) {
    console.error("--cwd requires a directory");
    process.exit(2);
  }
  cwd = path.resolve(requested);
  argv.splice(cwdIndex, 2);
}

const rawMode = argv[0] === "--raw";
if (rawMode) argv.shift();
if (argv[0] === "/tinker" || argv[0] === "tinker") argv.shift();
if (argv.length === 0) argv.push("help");

const command = rawMode
  ? `/tinker ${argv.join(" ")}`
  : `/tinker ${argv.map((value) => JSON.stringify(value)).join(" ")}`;
const piBinary = process.env.PI_TINKER_PI_BIN || "pi";
const child = spawn(
  piBinary,
  ["--no-extensions", "-e", packageRoot, "--no-session", "-p", command],
  { cwd, env: { ...process.env, PI_TINKER_AGENT_CLI: "1" }, stdio: ["inherit", "pipe", "pipe"] },
);

// Pi reserves stderr for print-mode extension output. Normalize both streams to
// stdout so shell-based coding agents receive one inspectable report stream.
child.stdout.pipe(process.stdout);
child.stderr.pipe(process.stdout);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("error", (error) => {
  if (error.code === "ENOENT") {
    console.error("Could not find the `pi` executable. Install Pi first, then retry.");
  } else {
    console.error(error.message);
  }
  process.exitCode = 1;
});

child.on("exit", (code, signal) => {
  if (signal) process.exitCode = 1;
  else process.exitCode = code ?? 1;
});
