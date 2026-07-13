# Autonomous beach microplastic sampling with Bayesian Rainbow-DQfD

This repository is a reproducible, research-oriented reconstruction of the project
described in the supplied résumé screenshot: an autonomous robot chooses where to
move and sample across many simulated beach profiles. It provides the missing MDP,
Bayesian information metric, agent, baselines, held-out evaluation, statistical
analysis, and spatial visualisations.

> **Scope of the replication.** The screenshot contains no original code, XBeach
> inputs, microplastic observations, hyperparameters, seeds, or result files. It is
> therefore impossible to verify the historical “approximately 100×” number from
> that image alone. This repository independently reconstructs and tests the claim;
> it never inserts a target ratio into the reward or result generator.

## What is implemented

- A formally specified finite-horizon belief MDP with safe action masking.
- A Gaussian RBF spatial posterior whose one-sample reward is the exact conditional
  mutual information
  \(I(\theta;y_x\mid D_t)=\tfrac12\log(1+\phi_x^T\Sigma_t\phi_x/\sigma_n^2)\).
- A masked **Rainbow-DQfD** agent: Double-Q action selection, dueling C51
  distributions, proportional prioritised replay, 5-step targets, factorised
  NoisyNet layers, and model-based demonstration pretraining with a margin loss.
- Uniform-random, systematic lawnmower, and path-aware greedy-information baselines.
- Paired evaluation on 1,024 untouched profiles, hierarchical bootstrap confidence
  intervals, paired randomisation tests, posterior RMSE, paths, and coverage maps.
- A deterministic procedural backend and a checked adapter for exported XBeach
  fields. XBeach covariates inform a separate microplastic model; XBeach sediment
  concentration is never relabelled as plastic concentration.

The complete posterior mean and covariance are in the observation. That belief is
a sufficient Markov state, so a feed-forward agent is appropriate; recurrence is
not used to disguise missing state. There is no universally “best” RL algorithm.
Rainbow is selected because this problem has a small discrete action space and
expensive, reusable off-policy transitions. DQfD-style demonstrations address the
long delay between movement and an informative sample. The strong planning baseline
remains essential: with fixed Gaussian noise, informative path planning can be
solved very well without deep RL.

## Checked result

The frozen three-seed study selected checkpoints on 128 validation profiles, then
evaluated them once on 1,024 new procedural profiles (seeds 50,000–51,023):

| Policy | Information gain (nats) | 95% hierarchical CI | Ratio to random |
| --- | ---: | ---: | ---: |
| Greedy information | 5.463 | [5.450, 5.476] | 2.053× |
| **Rainbow-DQfD** | **5.315** | **[5.291, 5.345]** | **1.997× [1.974, 2.022]** |
| Lawnmower | 3.451 | [3.427, 3.474] | 1.297× |
| Random | 2.661 | [2.632, 2.690] | 1.000× |

Thus the screenshot's approximate 100× claim is **not reproduced**. The independent
reconstruction supports an approximately 2× gain over random and places the learned
policy within 2.7% of the model-based greedy planner. Raw episode rows, checkpoint
selection, runtime metadata, and figures are archived in
[`results/research`](results/research/README.md).

![Held-out benchmark](results/research/benchmark.png)

## Quick start

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
pytest

# End-to-end CPU smoke run
beach-rl study --config configs/smoke.yaml --output runs/smoke

# Prespecified 3-seed study and 1,024-profile held-out benchmark
beach-rl study --config configs/research.yaml --output runs/research
```

The study produces `summary.csv`, per-episode data, checkpoints, resolved
configuration, runtime metadata, a learning curve, benchmark confidence intervals,
coverage heatmaps, and an example trajectory. Training and held-out profile seeds
occupy disjoint numeric ranges.

Useful overrides:

```bash
beach-rl study --config configs/research.yaml --output runs/ten-seed \
  --seeds 11 29 47 71 89 101 131 173 211 251 --steps 50000 --profiles 2048

beach-rl evaluate --config configs/research.yaml \
  --checkpoint runs/research/training/seed_11/checkpoint_final.pt \
  --output runs/re-evaluation --profiles 1024
```

## Repository guide

- [`docs/METHODS.md`](docs/METHODS.md): generative model, belief MDP, reward, and metrics.
- [`docs/ALGORITHM.md`](docs/ALGORITHM.md): Rainbow losses and implementation details.
- [`docs/XBEACH.md`](docs/XBEACH.md): scientifically valid XBeach boundary and adapter schema.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): experiment protocol and claim policy.
- [`src/beach_rl`](src/beach_rl): tested package and command-line study runner.
- [`tests`](tests): mathematical invariants, replay, C51, environment, and adapter tests.

## Scientific interpretation

“Information gain” is reported in nats, not an arbitrary score. Movement and revisit
costs affect the RL return but never the reported information metric. Ratios are
reported alongside absolute differences because a ratio is unstable when a random
baseline is near zero. Deep Q-learning with nonlinear function approximation has no
global optimality guarantee; mathematical rigor here means a correct probabilistic
objective, explicit Bellman estimator, tested invariants, controlled data splits,
and uncertainty-aware reporting—not a false proof of optimality.

The implementation follows the components studied in [Rainbow (Hessel et al.,
2018)](https://doi.org/10.1609/aaai.v32i1.11796), including [Double DQN (van Hasselt
et al., 2016)](https://doi.org/10.1609/aaai.v30i1.10295). The spatial objective is
related to Gaussian-process sensor placement [Krause et al.,
2008](https://jmlr.org/papers/v9/krause08a.html). XBeach process claims follow the
[official manual](https://xbeach.readthedocs.io/en/latest/xbeach_manual.html) and
[Roelvink et al. (2009)](https://doi.org/10.1016/j.coastaleng.2009.08.006).

## License

MIT. Cite the repository metadata in [`CITATION.cff`](CITATION.cff) when reusing it.
