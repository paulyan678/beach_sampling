# Dense-field research experiment

I use this directory to archive the lightweight outputs of my frozen
`configs/research.yaml` experiment run on 2026-07-13. Large PyTorch checkpoints are
excluded from Git, while the configuration needed to retrain them, validation-only
checkpoint scores, raw test rows, runtime metadata, and training curves are retained.

## Result

Across 1,024 lockbox procedural profiles, the three validation-selected
Rainbow-DQfD agents averaged **5.315 nats** of information versus **2.661 nats** for
random: **1.997×**, 95% hierarchical bootstrap interval **[1.974, 2.022]**. The
paired absolute improvement was 2.654 nats [2.615, 2.696], with paired sign
randomisation p < 0.00005. This approximately 2× result is the primary quantitative
finding of the project. Rainbow-DQfD reached 97.3% of the model-based greedy
planner's information gain rather than outperforming it.

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

I paired every policy on seeds 50,000–51,023. Confidence intervals resample both
training seeds and profiles where applicable. These outputs come from my procedural
simulator; I did not use XBeach or field observations for this experiment.
