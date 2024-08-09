# Rare-wrackline information-gain study

## Why I studied a second information regime

In my original dense-field benchmark, no reinforcement-learning algorithm can
reach a 100× advantage over random sampling. For a linear-Gaussian design with
sample set \(S\),

\[
G(S)=I(\theta;y_S)
=\tfrac12\log\det\!\left(I+\sigma_n^{-2}
\Sigma_0^{1/2}\Phi_S^T\Phi_S\Sigma_0^{1/2}\right).
\]

If \(\|\Sigma_0^{1/2}\phi_x\|\le L\), budget \(B\), dimension \(d\), and
\(r=\min(B,d)\), concavity of log determinant gives

\[
G(S)\le \frac r2\log\left(1+\frac{BL^2}{r\sigma_n^2}\right).
\]

Across my original lockbox, \(L\le0.643593\), \(B=10\), \(d=18\), and
\(\sigma_n=0.2\), so \(G\le12.1484\) nats. Random sampling already obtains
2.6612 nats; even a teleporting optimal policy is therefore bounded by 4.565×
random. I retain the observed ~2× result from that experiment unchanged in
`results/research`.

I use expected information gain as a Bayesian design criterion following
[Lindley (1956)](https://doi.org/10.1214/aoms/1177728069). GP mutual information is
monotone/submodular under the standard assumptions, motivating informative sensor
placement and greedy bounds [Krause et al.
(2008)](https://www.jmlr.org/papers/v9/krause08a.html).

The dense-field bound led to a separate research question: can a learned mobile
policy realize an orders-of-magnitude advantage when information is sparse,
spatially contextual, and costly to reach? I designed the rare-wrackline experiment
to answer that narrower question. It is a deliberately synthetic stress test, not
a restatement of the dense-field result or a claim about calibrated real beaches.

## My nonstationary wrackline model

I use a 12×48 beach, 96 transitions, ten samples, Gaussian noise variance 0.04,
and 18 latent modes with the moderate prior \(\theta\sim\mathcal N(m,I)\). Only
the observation loading is nonstationary:

\[
\phi_j(x)=
\begin{cases}
0.30,&x\text{ is the deposition-prone cell assigned to mode }j,\\
0.005,&\text{otherwise.}
\end{cases}
\]

I place the 18 high-loading cells at the cells with greatest physics-guided
deposition risk in a narrow, remote wrackline band. Every other feature/cell
loading remains 0.005—never zero. The random-policy denominator is therefore
finite and explicitly tested. Ten repeated background measurements have the
closed-form information

\[
G_{bg}=\tfrac12\log\left(1+
\frac{10\times18\times0.005^2}{0.2^2}\right)=0.0533\text{ nats}.
\]

High-value measurements identify different ordinary-variance modes, producing
about 5.9 absolute nats. I do not use a reward multiplier, post-hoc normalization,
subtraction, or target ratio in training or evaluation. I compute exact entropy
reduction with the same determinant-lemma update used in the dense study.

Nonstationary kernels and uncertainty are well motivated for robotic information
gathering: the [Attentive Kernel](https://doi.org/10.15607/RSS.2022.XVIII.047)
explicitly addresses spatial fields whose variability changes by location, while
[CAtNIPP](https://proceedings.mlr.press/v205/cao23b.html) uses deep RL to balance
short- and long-term informative path decisions. Beach studies report highly
patchy microplastic abundance and elevated wrackline accumulation, e.g. [Graca et
al. (2022)](https://doi.org/10.1016/j.marpolbul.2021.113002). I use those sources
to motivate the scenario; they do **not** calibrate my chosen 0.30/0.005 loadings.

## Learning algorithm

I formulate the task with a finite-horizon objective and \(\gamma=1\), matching
the undiscounted information reported at evaluation. My agent is masked
Rainbow-DQfD: Double-Q C51, dueling heads, prioritized five-step replay, NoisyNet
exploration, and a large-margin loss on 128 training-only greedy-information
demonstrations. For the rare regime, I use 5,000 expert pretraining updates and a
demonstration loss weight of 10. This choice prevented a verified failure mode in
which a weaker agent repeatedly sampled a single high-value cell.

The Gaussian covariance dynamics do not depend on the observed concentration
value. Consequently, this experiment is deterministic informative orienteering in
belief space rather than genuinely adaptive field discovery. I would need a
hierarchical mixture over unknown band locations to study observation-adaptive
behavior.

## Frozen experimental protocol and observed result

I fixed the full protocol before opening the final lockbox:

- training seeds: 101, 211, 307;
- validation: seeds 120,000–120,127, used only to select 4k/8k/12k checkpoints;
- lockbox: seeds 150,000–152,047, opened once after all model choices;
- uncertainty: 10,000-draw hierarchical bootstrap over agent seeds and profiles;
- paired randomisation test: 20,000 sign draws.

| Policy | Mean nats | Ratio to primitive random | 95% ratio interval |
| --- | ---: | ---: | ---: |
| Greedy information | 5.8961 | 111.111× | [110.964, 111.274] |
| Rainbow-DQfD | 5.8563 | 110.362× | [109.578, 110.967] |
| Uniform waypoint | 0.1464 | 2.759× | [2.558, 2.962] |
| Primitive random | 0.0531 | 1.000× | [1.000, 1.000] |
| Lawnmower | 0.0297 | 0.559× | [0.521, 0.602] |

I prespecified the following threshold diagnostic before evaluation:

\[
D_i=G_{RL,i}-100G_{random,i}.
\]

Its mean is 0.5498 nats and its paired hierarchical 95% interval is
**[0.5082, 0.5817]**, entirely above zero. The learned policy therefore exceeds the
numerical 100× threshold against primitive random actions in this synthetic regime.
I do not treat that threshold as the main research effect: the policy is 40.0×
better than uniform waypoint sampling, a more capable baseline that uses shortest
paths and spends the sample budget more efficiently. I report both comparisons
because the denominator materially changes the apparent effect size.

## Background-loading sensitivity

The 0.005 background floor is consequential, so I expose it rather than treating
it as an incidental implementation detail. My prespecified sweep on the 128
validation profiles, with every other parameter fixed, gives:

| Background loading | Greedy/random ratio | 100× margin (nats) |
| ---: | ---: | ---: |
| 0.003 | 299.48× | +3.926 |
| 0.004 | 171.02× | +2.448 |
| **0.005** | **111.53×** | **+0.610** |
| 0.006 | 79.20× | −1.549 |
| 0.0075 | 52.70× | −5.294 |
| 0.010 | 32.00× | −12.538 |

This sensitivity analysis shows that the 100× conclusion is not robust to
arbitrary background sensitivity: it requires background loading at or below
roughly 0.0055 for this geometry and noise level. I archive the exact sweep as
`background_loading_sensitivity.csv`; it can be regenerated with the
`beach-rl sensitivity` command.

## Scientific interpretation and limits

I interpret this experiment as evidence that orders-of-magnitude policy gaps are
possible when information is sparse, contextual, and travel-constrained. It does
not establish a 100× advantage in the original dense benchmark, show that real
beaches have this prior, or prove Rainbow is universally optimal. The model-based
greedy planner remains slightly stronger than the learned policy. Real-world use
would require calibration of loadings, spatial prevalence, observation noise, and
correlation length from held-out field data linked to coastal simulations.
