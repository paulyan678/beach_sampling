# Archived research run

This directory contains the lightweight, version-controlled outputs of the frozen
`configs/research.yaml` study run on 2026-07-13. Large PyTorch checkpoints are
excluded from Git, but all inputs needed to retrain them, validation-only checkpoint
scores, raw test rows, runtime metadata, and training curves are retained.

## Result

Across 1,024 lockbox procedural profiles, the three validation-selected
Rainbow-DQfD agents averaged **5.315 nats** of information versus **2.661 nats** for
random: **1.997×**, 95% hierarchical bootstrap interval **[1.974, 2.022]**. The
paired absolute improvement was 2.654 nats [2.615, 2.696], with paired sign
randomisation p < 0.00005. The 100× screenshot claim was not reproduced.

## Files

- `summary.csv`, `episodes.csv`: complete lockbox statistics and raw rows.
- `validation/summary.csv`: validation report used only for model selection.
- `model_selection.csv`: 10k/20k/30k validation means for each training seed.
- `training/seed_*/training.csv`: learning traces including target-support clipping.
- `training/seed_*/metadata.json`: resolved config, hardware, versions, and runtime.
- `benchmark.png`, `coverage.png`, `trajectory_rainbow.png`, `learning_curves.png`:
  generated figures.
- `resolved_config.yaml`: exact study configuration.
- `SHA256SUMS`: hashes of archived result artifacts.

All policies were paired on seeds 50,000–51,023. Confidence intervals resample both
training seeds and profiles where applicable. These outputs are synthetic-only;
they are not historical XBeach runs.
