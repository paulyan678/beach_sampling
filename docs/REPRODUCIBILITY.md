# Reproducibility and claim protocol

## Two levels of run

- `configs/smoke.yaml`: fast integration test; never use its metrics in a paper.
- `configs/research.yaml`: three fixed training seeds, 30,000 environment steps per
  seed, 128 validation cases, and 1,024 lockbox test cases.
- Recommended publication extension: ten seeds, 50,000 steps, 2,048 cases (command
  in the README), plus a second XBeach-backed test set.

Training uses million-scale seed namespaces, demonstration cases use a separate
offset, validation is 20,000–20,127, and the final test lockbox is
50,000–51,023. The earlier development pilots used 10,000-series cases, not the
lockbox. Every output directory records the resolved YAML, per-seed runtime metadata,
training series, checkpoints, and raw evaluation rows. Evaluation is deterministic
for Rainbow and case-matched across policies. Random-policy RNG seeds are derived
from but distinct from profile seeds.

For each training seed, checkpoints at 10k, 20k, and 30k steps are compared only on
the validation split; `model_selection.csv` records the scores and the best is
loaded before the lockbox is opened. No checkpoint is selected from test results.

## Recreating the environment

`requirements-lock.txt` pins the direct Python packages used for the checked run;
the package metadata constrains their compatible ranges. CI independently resolves
those ranges on Python 3.10 and 3.12, runs all unit tests, and completes an
end-to-end CPU study.

For strict archival work, additionally record:

```bash
python -VV
python -m pip freeze --all
git rev-parse HEAD
uname -a
```

PyTorch may use MPS/CUDA kernels that are numerically but not bitwise identical to
CPU. Deterministic algorithms are requested with warnings; research conclusions
must be checked across seeds, not inferred from identical bytes.

## What would constitute replication of the screenshot

The historical 100× claim can only be replicated after obtaining its original:

1. XBeach inputs/outputs and executable version;
2. microplastic transport or concentration-generation model;
3. state/action/reward definitions and information-gain units;
4. policy implementations, checkpoints, seeds, and data split;
5. raw episode-level results and analysis code.

Until then, this repository is an independent, testable reconstruction. A measured
ratio is reported exactly as observed. It is never rounded to “100×” unless its
definition, baseline denominator, and confidence interval support that statement.
Absolute differences and random-denominator values must accompany every ratio.

## Test coverage

- entropy reduction equals the closed-form mutual information;
- posterior covariance remains positive semidefinite and predictive uncertainty
  does not increase after conditioning;
- seed determinism, action masks, and sample reward identity;
- n-step terminal flush and prioritised replay tensor shapes;
- C51 target projection conserves probability;
- deterministic procedural generation and XBeach export validation;
- complete CLI smoke study in continuous integration.
