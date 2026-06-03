#!/usr/bin/env node
import { existsSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const required = [
  "package.json",
  "README.md",
  "LICENSE",
  "NOTICE",
  "extensions/tinker.ts",
  "skills/tinker-research/SKILL.md",
  "skills/tinker-debug/SKILL.md",
  "docs/commands.md",
  "docs/design.md",
  "examples/conversations.jsonl",
];

let failed = false;

for (const file of required) {
  if (!existsSync(file)) {
    console.error(`missing required file: ${file}`);
    failed = true;
  }
}

function checkSkill(file, expectedName) {
  const text = readFileSync(file, "utf8");
  const frontmatter = text.match(/^---\n([\s\S]*?)\n---/);
  if (!frontmatter) {
    console.error(`${file}: missing frontmatter`);
    failed = true;
    return;
  }
  if (!frontmatter[1].includes(`name: ${expectedName}`)) {
    console.error(`${file}: expected name: ${expectedName}`);
    failed = true;
  }
  const desc = frontmatter[1].match(/^description:\s*(.*)$/m)?.[1] ?? "";
  if (!desc) {
    console.error(`${file}: missing description`);
    failed = true;
  }
  if (desc.length > 1024) {
    console.error(`${file}: description too long (${desc.length})`);
    failed = true;
  }
}

checkSkill("skills/tinker-research/SKILL.md", "tinker-research");
checkSkill("skills/tinker-debug/SKILL.md", "tinker-debug");

try {
  execFileSync(process.execPath, ["--check", "extensions/tinker.ts"], { stdio: "pipe" });
} catch (error) {
  console.error("extensions/tinker.ts failed node --check");
  console.error(error.stderr?.toString() || error.message);
  failed = true;
}

const pkg = JSON.parse(readFileSync("package.json", "utf8"));
if (!pkg.pi?.extensions || !pkg.pi?.skills) {
  console.error("package.json: missing pi.extensions or pi.skills");
  failed = true;
}
if (!pkg.keywords?.includes("pi-package")) {
  console.error("package.json: missing pi-package keyword");
  failed = true;
}

if (failed) process.exit(1);
console.log("smoke test passed");
