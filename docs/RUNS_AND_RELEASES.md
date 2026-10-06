# Run identity and release decisions

The July 2026 experiments are new follow-up work after the 2024 UT Austin
assistantship. Preserve that distinction in citations, biographies and summaries.
The historical result files and their existing checksums remain unchanged.

## Start a new run instead of reusing a directory

`beach-rl train` and `beach-rl study` reject nonempty output directories before
training or replacing configuration. There is no implicit resume mode. A previous
100-step checkpoint must never compete in a new 50-step study.

Every run writes an exclusive `run.json`: UUID, full config and hash, actual
source-file hashes, Git revision/dirty state, installed environment, and disjoint
procedural seed namespaces. Each completed training run writes `completed.json`
with all emitted checkpoint hashes. Study selection reads that receipt, verifies
integrity, and includes the final step even when no periodic checkpoint was due.
The study completion receipt records selected checkpoint hashes and validation
scores. An interrupted run remains identifiable and must not be called complete.

XBeach runs additionally retain the existing case-bank manifest; seed separation
does not itself prove that externally supplied files represent independent sites.
External dataset provenance and licenses remain the researcher's responsibility.

## Regenerate the historical numbers without retraining

```bash
python -m pip install -r requirements-ci.txt
python -m pip install --no-deps -e .
python -m beach_rl.artifacts report \
  --episodes results/research/episodes.csv --output runs/regenerated-dense
```

This validates finite, unique, paired rows and recomputes the statistics with the
documented bootstrap seed. CI compares means, ratios and confidence intervals to
the archived table. Its receipt identifies input/output hashes and current code.
It does **not** recover the trained agents or establish field performance.

The original `.pt`/`.pth` checkpoints were not found among tracked files or
targeted Git history. They remain unavailable; no replacement weights are passed
off as original. New training produces a separate dated experiment.

## Export a completed run for review

```bash
python -m beach_rl.artifacts export-release --run runs/my-study --output releases/my-study
# Only if you hold redistribution rights to the trained artifacts:
python -m beach_rl.artifacts export-release --run runs/my-study \
  --output releases/my-study-with-weights --include-checkpoints
```

Export verifies the run receipt and artifact hashes before copying. For a study,
this includes all training checkpoints, even candidates validation did not select.
A release
manifest lists included files and explicitly omitted checkpoints. It never
uploads or publishes files. Inspect the bundle, then publish it as an immutable
release tied to the recorded commit. Existing exports cannot be overwritten.

The current CI lock is separate from the historical experiment lock. CPU smoke
training verifies software execution, not the dense result. Full-grid retraining,
new seeds, field tests and XBeach calibration require separate budgets and data.
