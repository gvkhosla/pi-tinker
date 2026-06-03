#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { copyFile, mkdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const repo = process.cwd();
const statePath = path.join(os.homedir(), ".pi", "agent", "tinker-checkpoints.json");
const backupPath = `${statePath}.pi-tinker-test-backup`;
const tmp = mkdtempSync(path.join(os.tmpdir(), "pi-tinker-integration-"));

function run(cmd, args, options = {}) {
  return execFileSync(cmd, args, {
    cwd: options.cwd ?? repo,
    encoding: "utf8",
    stdio: options.stdio ?? "pipe",
    env: { TINKER_API_KEY: "pi-tinker-local-test", ...process.env, ...(options.env ?? {}) },
  });
}

function pi(args, options = {}) {
  return run("pi", ["--no-extensions", "-e", repo, "--no-session", ...args], options);
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function main() {
  // Fail fast if pi is not available locally.
  run("pi", ["--version"]);

  // Preserve user checkpoint registrations while testing /tinker use.
  if (existsSync(backupPath)) rmSync(backupPath);
  if (existsSync(statePath)) renameSync(statePath, backupPath);

  try {
    await copyFile(path.join(repo, "examples", "conversations.jsonl"), path.join(tmp, "train.jsonl"));

    // Extension loads without crashing.
    pi(["--list-models"], { cwd: tmp });

    // /tinker init should generate the golden-path project files.
    pi(["-p", "/tinker init train.jsonl --metric quality --force"], { cwd: tmp });
    for (const file of ["README.md", "train_sft.py", "eval_checkpoint.py", "tinker.yaml", "notes/plan.md"]) {
      assert(existsSync(path.join(tmp, file)), `/tinker init did not create ${file}`);
    }

    // Generated Python should be syntactically valid without importing dependencies.
    run("python3", ["-m", "py_compile", "train_sft.py", "eval_checkpoint.py"], { cwd: tmp });

    const readme = readFileSync(path.join(tmp, "README.md"), "utf8");
    assert(readme.includes("/tinker monitor"), "generated README missing monitor instructions");
    assert(readFileSync(path.join(tmp, "tinker.yaml"), "utf8").includes("success_metric: quality"), "tinker.yaml missing success metric");

    // /tinker sft should refuse to clobber unless --force, then regenerate.
    pi(["-p", "/tinker sft train.jsonl --force --steps 3"], { cwd: tmp });
    assert(readFileSync(path.join(tmp, "train_sft.py"), "utf8").includes('"max_steps": 3'), "/tinker sft did not apply steps override");

    // Checkpoint listing should handle cookbook checkpoint records.
    await mkdir(path.join(tmp, "logs", "run"), { recursive: true });
    writeFileSync(
      path.join(tmp, "logs", "run", "checkpoints.jsonl"),
      JSON.stringify({ name: "step-2", batch: 2, sampler_path: "tinker://pi-tinker-test/sampler_weights/000002", state_path: "tinker://pi-tinker-test/weights/000002" }) + "\n",
    );
    writeFileSync(path.join(tmp, "logs", "run", "metrics.jsonl"), JSON.stringify({ "progress/batch": 2, "train/loss": 1.23, "time/total": 4.56 }) + "\n");
    pi(["-p", "/tinker checkpoints logs/run"], { cwd: tmp });
    pi(["-p", "/tinker status logs/run"], { cwd: tmp });
    pi(["-p", "/tinker monitor logs/run"], { cwd: tmp });
    pi(["-p", "/tinker monitor --stop"], { cwd: tmp });

    // /tinker use should register a checkpoint model in Pi's model registry.
    pi(["-p", "/tinker use tinker://pi-tinker-test/sampler_weights/000001 pi-tinker-test"], { cwd: tmp });
    assert(existsSync(statePath), "/tinker use did not create checkpoint state file");
    const state = JSON.parse(readFileSync(statePath, "utf8"));
    assert(state.checkpoints?.some((m) => m.name === "pi-tinker-test"), "/tinker use did not persist checkpoint model");

    const models = run("bash", ["-lc", `pi --no-extensions -e '${repo}' --list-models pi-tinker-test 2>&1`], { cwd: tmp });
    assert(models.includes("tinker://pi-tinker-test/sampler_weights/000001") || models.includes("pi-tinker-test"), "registered Tinker model did not appear in --list-models output");

    pi(["-p", "/tinker use --remove pi-tinker-test"], { cwd: tmp });

    console.log(`local integration test passed (${tmp})`);
  } finally {
    // Restore user state.
    if (existsSync(statePath)) rmSync(statePath);
    if (existsSync(backupPath)) renameSync(backupPath, statePath);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
