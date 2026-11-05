# Experience → reviewed datasets (experimental)

A local-only bridge from Witchcraft/Pickbrain discovery to editable Tinker data.
This is candidate mining, **not automatic learning from history**. It never calls
Tinker, uploads traces, or judges whether a past answer was correct.

Run `pi-tinker-experience` after installing the package, or from this checkout:

```bash
node scripts/experience.mjs --help
```

## 1. Discover (optional)

```bash
pi-tinker-experience discover "explaining Python import errors"
```

Requires an independently installed, already indexed Pickbrain. Searches only Pi
using `--no-ingest`; it does not run ingestion hooks, install services, or download
models. Missing Pickbrain does not prevent import/export. Search output is only a
locator: import original session JSONL, never formatted search excerpts.

## 2. Import explicit source files

Use a private working directory outside your git repository:

```bash
pi-tinker-experience import review.json /path/to/session-one.jsonl /path/to/session-two.jsonl
```

Pi v3 sessions normally live under `~/.pi/agent/sessions/`. This command only reads
files you explicitly name. It refuses to overwrite an existing manifest. The
manifest is written with owner-only permissions, but still contains sensitive
conversation text. Do not commit it, paste it publicly, or sync it unintentionally.

Each candidate has source session/message IDs and a SHA-256 of the original file.
Extraction follows parent links, not adjacent lines, so different branches are not
stitched together. Assistant thinking is omitted. Tool/multimodal trajectories
are marked blocked and their contents are not flattened into training messages.
Sessions containing context edits are conservatively blocked. Sources are not
modified; raw tool content remains in the original session for future adapters.

Only user → assistant text pairs are currently supported. Earlier conversation,
system prompts, and repository state are **not** reconstructed. That makes manual
standalone review essential. This version is best for independently understandable
text tasks, not learning a coding agent's tool policy.

## 3. Curate `review.json`

All candidates start `pending`. Edit the manifest in a local editor:

- Correct/redact `messages`, or set `status` to `rejected`.
- Approve only examples whose answers you verified; set `status` to `approved`.
- Add nonempty `evidence`: e.g. a checked reference, explicit user acceptance, or
  test result. A historical assistant saying “done” is not evidence on its own.
- Set `privacyReviewed: true` after checking **all** content for secrets, personal
  information, proprietary material, and permission to use it for training.
- Set `standaloneReviewed: true` only if the pair makes sense without omitted
  conversation, tools, system instructions, or files.
- Assign `split: "train"` or `split: "eval"`.
- Assign related tasks, forks, and near-duplicates the **same `group`** across
  source sessions. The default groups by session (or fork parent); this is not
  automatic task clustering. Reject duplicate prompts.
- For eval candidates explicitly set `evalMatch: "exact"` or `"contains"` and
  review the expected assistant answer. These simple text matchers are unsuitable
  for most open-ended coding tasks; use separately authored executable evals there.

Do not edit source identifiers, hashes, IDs, or unblock unsupported trajectories.
The manifest is a trusted local review artifact, not a tamper-proof audit system.

## 4. Export

```bash
pi-tinker-experience export review.json curated-dataset
```

The output directory must not exist. Export requires at least one approved example
in each split, with separate source sessions/task groups. It rejects duplicate
normalized prompts and obvious secret/email patterns. This scanner is heuristic:
**a clean scan is not a privacy guarantee**. Near-duplicate detection is manual.

Files:

```text
curated-dataset/train.jsonl       user/assistant chat rows
curated-dataset/eval.jsonl        held-out prompts, expected answers, matchers
curated-dataset/provenance.json   reviewed content hashes and source references
curated-dataset/report.json       counts and limitations
```

Pending/rejected candidates are not exported. Dataset files have owner-only
permissions; source paths and review evidence may still be sensitive.

## 5. Inspect before training

In Pi, validate the exported training file locally:

```text
/tinker validate curated-dataset/train.jsonl --quick
```

Keep the separately reviewed eval file. Do not replace it with an automatic split
of the training set. Use the manual eval/training workflow in `docs/commands.md`,
review the target model's renderer/token masks, and obtain explicit approval
before any API-using stage. Export is not permission to upload or train.

Start with a narrow task and 20–50 carefully reviewed examples. Measure the base
model and retrieval-only baseline before deciding whether fine-tuning is useful.
No automatic success labels, preference pairs, tool-trajectory SFT, remote upload,
or executable coding-eval reconstruction are implemented yet.
