# Real Tinker training runbook

Use this to produce a verified before/after result for the README or a launch post.

> This requires `TINKER_API_KEY`. The repository's automated tests intentionally do not run real Tinker API jobs.

## 1. Install and set up

```bash
pi install git:github.com/gvkhosla/pi-tinker
export TINKER_API_KEY="..."
uv pip install tinker-cookbook
```

## 2. Start with the bundled customer-support CSV

```text
/tinker improve examples/customer-support.csv --goal "better concise customer support answers" --budget demo --force
```

Confirm the doctor report is clean enough to proceed.

## 3. Run the smoke budget

```text
/tinker improve examples/customer-support.csv --goal "better concise customer support answers" --budget smoke --yes
```

Expected result:

- baseline eval result in `eval_results/baseline.json`,
- 2-step training smoke run,
- metrics in the generated log directory.

## 4. Run a short training budget

```text
/tinker improve examples/customer-support.csv --goal "better concise customer support answers" --budget small --yes
```

Expected result:

- training metrics,
- sampler checkpoint path,
- checkpoint eval result,
- baseline vs checkpoint comparison,
- checkpoint registered in Pi.

## 5. Generate app snippets

```text
/tinker deploy latest
```

## 6. Document the result

Add a short result block to the README or release notes:

```text
Dataset: examples/customer-support.csv
Base model: <model>
Budget: small
Baseline eval: <score>
Checkpoint eval: <score>
Delta: <+/- points>
Checkpoint: <tinker://.../sampler_weights/...>
What improved: <1-3 bullets>
Remaining failures: <1-3 bullets>
```

## Notes

- Use tiny data only to verify the workflow; do not claim broad model quality from toy examples.
- Prefer eval examples that are held out from training.
- If the checkpoint does not improve, that is still useful: document the failure and what data/eval changes are needed.
