# Research protocol and reproducibility

## Experimental configurations

I maintain separate configurations for integration checks and research studies so
that software validation cannot be mistaken for scientific evidence:

- `configs/smoke.yaml`: fast integration test; I never use its metrics in a paper.
- `configs/research.yaml`: three fixed training seeds, 30,000 environment steps per
  seed, 128 validation cases, and 1,024 lockbox test cases.
- Recommended publication extension: ten seeds, 50,000 steps, 2,048 cases (command
  in the README), plus a second XBeach-backed test set.
- `configs/rare_hotspot_100x.yaml`: my separate three-seed nonstationary wrackline
  stress test with 128 validation and 2,048 lockbox profiles. Its 100× result is
  scoped only to primitive random actions and uses the explicit paired margin in
  `docs/RARE_HOTSPOT_100X.md`.

I use million-scale seed namespaces for training and a separate offset for
demonstration cases. Validation uses cases 20,000–20,127, and the final test lockbox
uses 50,000–51,023. My earlier development pilots used 10,000-series cases, not
the lockbox. Every output directory records the resolved YAML, per-seed runtime
metadata, training series, checkpoints, and raw evaluation rows. Rainbow evaluation
is deterministic and case-matched across policies. Random-policy RNG seeds are
derived from, but distinct from, profile seeds.

For each training seed, I compare checkpoints at 10k, 20k, and 30k steps only on
the validation split. `model_selection.csv` records those scores, and I load the
best checkpoint before opening the lockbox. I never select a checkpoint from test
results.

## Recreating my software environment

`requirements-lock.txt` preserves the historical experiment's direct-package
environment. `requirements-ci.txt` pins the dependency closure tested for the
October 2026 audit fixes. Current CI uses those pins on Python 3.12, runs all unit
tests, regenerates the archived dense statistics, and completes an end-to-end CPU
smoke study. This does not rerun the full historical training experiment.

For strict archival work, I additionally record:

```bash
python -VV
python -m pip freeze --all
git rev-parse HEAD
uname -a
```

PyTorch may use MPS/CUDA kernels that are numerically but not bitwise identical to
CPU. I request deterministic algorithms with warnings, but I base research
conclusions on consistency across seeds rather than identical output bytes.

## Evidence and claim boundaries

I treat the dense-field lockbox as the primary research result because it evaluates
the policy under the general procedural regime against both weak and strong
baselines. I report every ratio with its definition, absolute information
difference, denominator, and uncertainty interval. This protocol yields an
approximately 2× improvement over primitive random actions and a learned policy
that remains slightly below the model-based greedy planner.

The rare-wrackline study answers a narrower sensitivity question. Its larger ratio
uses a deliberately weak background observation loading and a nearly zero primitive
random denominator. I therefore report the 5.856-nat absolute result, the comparison
with uniform-waypoint sampling, and the background-loading sweep alongside that
ratio. I do not generalise it to the dense experiment or to real beaches.

Any field-facing claim would require calibrated XBeach inputs and executable
version, an independently justified microplastic observation model, held-out
measurements, whole-site data splits, and the same raw episode-level analysis used
here. Those data are outside the scope of the current study.

## Test coverage supporting the experiments

I test the following numerical and implementation properties:

- entropy reduction equals the closed-form mutual information;
- posterior covariance remains positive semidefinite and predictive uncertainty
  does not increase after conditioning;
- seed determinism, action masks, and sample reward identity;
- n-step terminal flush and prioritised replay tensor shapes;
- C51 target projection conserves probability;
- deterministic procedural generation and XBeach export validation;
- complete CLI smoke study in continuous integration.
