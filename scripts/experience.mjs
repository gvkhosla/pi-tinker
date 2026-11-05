#!/usr/bin/env node
// Local-only candidate curation. No model/API calls; no Pi runtime required.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const hash = value => createHash('sha256').update(value).digest('hex');
const json = file => JSON.parse(readFileSync(file, 'utf8'));
const lines = rows => rows.map(row => JSON.stringify(row)).join('\n') + '\n';
const save = (file, value) => writeFileSync(file, JSON.stringify(value, null, 2) + '\n', { flag: 'wx', mode: 0o600 });

export function scan(text) {
  const warnings = [];
  if (/(?:sk-[\w-]{16,}|AKIA[A-Z0-9]{16}|-----BEGIN .*PRIVATE KEY-----|(?:api[_-]?key|password|secret|token)\s*[:=]\s*["']?[^\s"']{8,})/i.test(text)) warnings.push('possible-secret');
  if (/[\w.+-]+@[\w.-]+\.[a-z]{2,}/i.test(text)) warnings.push('possible-email');
  return warnings;
}

function textOnly(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content) || content.some(b => !['text', 'thinking'].includes(b.type))) return undefined;
  return content.filter(b => b.type === 'text').map(b => b.text).join('\n');
}

export function importSession(file) {
  const raw = readFileSync(file, 'utf8');
  const entries = raw.split(/\r?\n/).filter(s => s.trim()).map(s => JSON.parse(s));
  const header = entries[0];
  if (header?.type !== 'session' || header.version !== 3 || !header.id) throw new Error('Expected a Pi v3 session: ' + file);
  const byId = new Map();
  for (const entry of entries.slice(1)) {
    if (!entry.id || byId.has(entry.id)) throw new Error('Missing/duplicate entry ID: ' + file);
    if (entry.parentId !== null && !byId.has(entry.parentId)) throw new Error('Missing/forward parent: ' + file);
    byId.set(entry.id, entry);
  }
  const candidates = [];
  for (const entry of entries) {
    if (entry.type !== 'message' || entry.message?.role !== 'assistant') continue;
    if (entry.message.stopReason && entry.message.stopReason !== 'stop') continue;
    const path = [];
    let parent = byId.get(entry.parentId);
    while (parent) {
      path.unshift(parent);
      if (parent.type === 'message' && parent.message?.role === 'user') break;
      parent = byId.get(parent.parentId);
    }
    const user = path[0];
    if (user?.message?.role !== 'user') continue;
    const prompt = textOnly(user.message.content);
    const answer = textOnly(entry.message.content);
    const blocked = entries.some(e => e.type === 'context_edit') || prompt === undefined || answer === undefined || path.slice(1).some(e => e.type === 'message' || ['compaction', 'context_edit', 'branch_summary', 'custom_message'].includes(e.type));
    const messages = blocked ? [] : [{ role: 'user', content: prompt }, { role: 'assistant', content: answer }];
    candidates.push({
      id: hash(header.id + ':' + entry.id).slice(0, 24),
      source: { file: resolve(file), sessionId: header.id, sourceHash: hash(raw), userEntryId: user.id, assistantEntryId: entry.id },
      // Reviewer must group related tasks/forks together, not merely individual turns.
      group: header.parentSession || header.id,
      status: 'pending', split: null, evidence: '', privacyReviewed: false, standaloneReviewed: false,
      blocked, warnings: blocked ? ['unsupported-tool-or-context-trajectory'] : ['standalone-context-review-required', ...scan(JSON.stringify(messages))],
      messages,
    });
  }
  return candidates;
}

export function buildDataset(manifest) {
  if (manifest.version !== 1 || !Array.isArray(manifest.candidates)) throw new Error('Invalid candidate manifest');
  const train = [], evaluation = [], provenance = [], seen = new Set(), groups = new Map(), prompts = new Map();
  for (const c of manifest.candidates) {
    if (!['pending', 'approved', 'rejected'].includes(c.status)) throw new Error('Invalid review status');
    if (c.status !== 'approved') continue;
    if (seen.has(c.id)) throw new Error('Duplicate candidate ID');
    seen.add(c.id);
    if (c.blocked || !c.privacyReviewed || !c.standaloneReviewed || !c.evidence?.trim()) throw new Error(`${c.id}: approval requires evidence, privacy and standalone review; blocked trajectories cannot export`);
    if (!['train', 'eval'].includes(c.split) || typeof c.group !== 'string' || !c.group.trim()) throw new Error(`${c.id}: assign train/eval and a task group`);
    const m = c.messages;
    if (!Array.isArray(m) || m.length !== 2 || m[0]?.role !== 'user' || m[1]?.role !== 'assistant' || m.some(x => typeof x.content !== 'string' || !x.content.trim())) throw new Error(`${c.id}: expected a nonempty user/assistant text pair`);
    if (scan(JSON.stringify({ messages: m, evidence: c.evidence })).length) throw new Error(`${c.id}: redact possible secrets/PII before export`);
    if (!c.source?.sessionId || !c.source?.sourceHash || !c.source?.assistantEntryId) throw new Error(`${c.id}: missing source provenance`);
    const group = groups.get('task:' + c.group);
    const session = groups.get('session:' + c.source?.sessionId);
    if ((group && group !== c.split) || (session && session !== c.split)) throw new Error('Task/session crosses train/eval boundary');
    groups.set('task:' + c.group, c.split);
    groups.set('session:' + c.source?.sessionId, c.split);
    const promptHash = hash(m[0].content.toLowerCase().replace(/\s+/g, ' ').trim());
    if (prompts.has(promptHash)) throw new Error('Duplicate prompt: deduplicate or reject related candidates');
    prompts.set(promptHash, c.split);
    if (c.split === 'train') train.push({ messages: m });
    else {
      if (!['exact', 'contains'].includes(c.evalMatch)) throw new Error(`${c.id}: evalMatch must be explicitly reviewed (exact or contains)`);
      evaluation.push({ messages: [m[0]], expected: m[1].content, match: c.evalMatch, notes: c.evidence });
    }
    provenance.push({ id: c.id, source: c.source, group: c.group, split: c.split, evidence: c.evidence, contentHash: hash(JSON.stringify(m)) });
  }
  if (!train.length || !evaluation.length) throw new Error('Need reviewed train and eval examples from separate sessions/task groups');
  return { train, evaluation, provenance, report: { approved: provenance.length, train: train.length, eval: evaluation.length, pending: manifest.candidates.filter(c => c.status === 'pending').length, rejected: manifest.candidates.filter(c => c.status === 'rejected').length, limitations: ['Secret scanning is heuristic; human review is required.', 'Near-duplicate and related-task grouping is manual.', 'Text pairs only; no tool trajectories or executable coding evals.', 'Export does not upload data or authorize training.'] } };
}

export function main(args) {
  const [command, ...rest] = args;
  if (command === 'discover') {
    if (rest.length !== 1 || !rest[0].trim() || rest[0].startsWith('-')) throw new Error('discover requires one quoted goal (not a flag)');
    const result = spawnSync('pickbrain', ['--no-ingest', '-q', '--type', 'pi', '-n', '20', rest[0]], { encoding: 'utf8', timeout: 30000, maxBuffer: 4 * 1024 * 1024 });
    if (result.error) throw new Error('Pickbrain unavailable or timed out. Install/index Pickbrain separately, or import explicit Pi session files.');
    if (result.status !== 0) throw new Error('Pickbrain search failed. Check its local index; no automatic ingestion was attempted.');
    process.stdout.write(result.stdout);
    console.log('\nDiscovery only: import original session JSONL files, not these excerpts.');
  } else if (command === 'import') {
    const [out, ...files] = rest;
    if (!out || !files.length) throw new Error('import <new-review.json> <session.jsonl> [...]');
    const candidates = files.flatMap(importSession);
    if (new Set(candidates.map(c => c.id)).size !== candidates.length) throw new Error('Duplicate sessions/candidates');
    save(out, { version: 1, candidates });
    console.log(`Wrote ${candidates.length} pending candidates to ${out}. Local sensitive file: review before export.`);
  } else if (command === 'export') {
    if (rest.length !== 2) throw new Error('export <review.json> <new-output-directory>');
    const dataset = buildDataset(json(rest[0]));
    // mkdir without recursive refuses existing destinations, avoiding clobbering.
    mkdirSync(rest[1], { mode: 0o700 });
    for (const [name, rows] of [['train.jsonl', dataset.train], ['eval.jsonl', dataset.evaluation]]) writeFileSync(join(rest[1], name), lines(rows), { flag: 'wx', mode: 0o600 });
    save(join(rest[1], 'provenance.json'), dataset.provenance);
    save(join(rest[1], 'report.json'), dataset.report);
    console.log(`Exported ${dataset.train.length} train / ${dataset.evaluation.length} eval examples. No API calls made.`);
  } else if (!command || ['--help', '-h'].includes(command)) {
    console.log('pi-tinker-experience (local-only)\n  discover "goal"\n  import review.json session.jsonl [session2.jsonl ...]\n  export review.json new-dataset-directory\n\nEdit review.json to curate; see docs/experience.md. Never commit private traces.');
  } else throw new Error('Unknown command: ' + command);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try { main(process.argv.slice(2)); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
