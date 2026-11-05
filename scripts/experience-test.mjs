import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { importSession, buildDataset, main } from './experience.mjs';

const fixture = (id, prompt = 'Question') => [
  { type: 'session', version: 3, id },
  { type: 'message', id: 'u', parentId: null, message: { role: 'user', content: prompt } },
  { type: 'message', id: 'a', parentId: 'u', message: { role: 'assistant', stopReason: 'stop', content: [{ type: 'thinking', thinking: 'PRIVATE THOUGHT' }, { type: 'text', text: 'Answer' }] } },
];
function workspace(fn) {
  const dir = mkdtempSync(join(tmpdir(), 'experience-test-'));
  try { return fn(dir); } finally { rmSync(dir, { recursive: true, force: true }); }
}
function put(dir, name, rows) {
  const file = join(dir, name);
  writeFileSync(file, rows.map(x => JSON.stringify(x)).join('\n'));
  return file;
}
function reviewed(dir) {
  const candidates = ['one', 'two'].flatMap((id, i) => importSession(put(dir, id, fixture(id, 'Question ' + i))));
  candidates.forEach((c, i) => Object.assign(c, { status: 'approved', split: i ? 'eval' : 'train', evidence: 'Reviewed against fixture', privacyReviewed: true, standaloneReviewed: true, evalMatch: 'exact' }));
  return { version: 1, candidates };
}

test('import is pending, source-linked, strips thinking, respects branches', () => workspace(dir => {
  const rows = fixture('one');
  rows.push({ ...rows[2], id: 'branch', parentId: 'u' });
  const candidates = importSession(put(dir, 's', rows));
  assert.equal(candidates.length, 2);
  assert.equal(candidates[1].source.userEntryId, 'u');
  assert.equal(candidates[0].status, 'pending');
  assert.ok(!JSON.stringify(candidates).includes('PRIVATE THOUGHT'));
}));

test('tool trajectories are blocked, never flattened', () => workspace(dir => {
  const rows = fixture('tools');
  rows.splice(2, 0, { type: 'message', id: 'tool', parentId: 'u', message: { role: 'toolResult', content: 'result' } });
  rows[3].parentId = 'tool';
  const [candidate] = importSession(put(dir, 's', rows));
  assert.equal(candidate.blocked, true);
  assert.deepEqual(candidate.messages, []);
}));

test('export produces normal JSONL and refuses overwrite', () => workspace(dir => {
  const manifest = reviewed(dir);
  const file = join(dir, 'review.json');
  writeFileSync(file, JSON.stringify(manifest));
  main(['export', file, join(dir, 'out')]);
  assert.equal(JSON.parse(readFileSync(join(dir, 'out/train.jsonl'), 'utf8')).messages.length, 2);
  assert.equal(JSON.parse(readFileSync(join(dir, 'out/eval.jsonl'), 'utf8')).match, 'exact');
  assert.throws(() => main(['export', file, join(dir, 'out')]));
}));

test('review gates fail closed', () => workspace(dir => {
  for (const patch of [{ privacyReviewed: false }, { standaloneReviewed: false }, { evidence: '' }, { blocked: true }, { split: null }, { status: 'pending' }]) {
    const manifest = reviewed(dir);
    Object.assign(manifest.candidates[0], patch);
    assert.throws(() => buildDataset(manifest));
  }
}));

test('session/task leakage, duplicate prompts and secrets block export', () => workspace(dir => {
  for (const mutate of [
    m => { m.candidates[1].group = m.candidates[0].group; },
    m => { m.candidates[1].source.sessionId = m.candidates[0].source.sessionId; },
    m => { m.candidates[1].messages[0].content = '  QUESTION   0 '; },
    m => { m.candidates[0].messages[1].content = 'password=reallysecret123'; },
    m => { m.candidates[0].messages[1].content = 'contact me@example.com'; },
    m => { delete m.candidates[1].evalMatch; },
    m => { m.candidates[1].evidence = 'password=reallysecret123'; },
  ]) {
    const manifest = reviewed(dir); mutate(manifest);
    assert.throws(() => buildDataset(manifest));
  }
}));

test('context edits and image input block candidates', () => workspace(dir => {
  const rows = fixture('edited');
  rows.push({ type: 'context_edit', id: 'edit', parentId: 'a', targetId: 'u', replacement: null });
  assert.equal(importSession(put(dir, 'edited', rows))[0].blocked, true);
  const images = fixture('images');
  images[1].message.content = [{ type: 'image', data: 'private-image' }];
  const candidate = importSession(put(dir, 'images', images))[0];
  assert.equal(candidate.blocked, true);
  assert.ok(!JSON.stringify(candidate).includes('private-image'));
}));

test('malformed trees and unsupported versions fail', () => workspace(dir => {
  const rows = fixture('bad'); rows[2].parentId = 'missing';
  assert.throws(() => importSession(put(dir, 'bad', rows)));
  rows[0].version = 1;
  assert.throws(() => importSession(put(dir, 'old', rows)));
}));
