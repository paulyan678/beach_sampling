# Learning algorithm: masked Rainbow-DQfD

I use a compact Rainbow-DQfD agent for the discrete belief MDP. I chose this
combination because off-policy transitions are reusable, the action space is small,
and model-based demonstrations can help bridge the delay between movement and an
informative sample. I do not claim that Rainbow-DQfD is universally best or that
deep Q-learning is globally optimal for this problem.

## Distributional Bellman target

My network represents a categorical return distribution on fixed atoms
$z_i=v_{min}+i\Delta z$, $i=0,\ldots,N-1$:

$$
Z_\theta(b,a)=\sum_i p_{\theta,i}(b,a)\,\delta_{z_i},\qquad
Q_\theta(b,a)=\sum_i z_i p_{\theta,i}(b,a).
$$

For an n-step replay item, the online network selects a valid action and the target
network evaluates it, giving the Double-DQN target

$$
a^*=\arg\max_{a\in\mathcal A(b_{t+n})}Q_\theta(b_{t+n},a),\qquad
Tz_i=\mathrm{clip}\left(R_t^{(n)}+\gamma^n(1-d_t)z_i,v_{min},v_{max}\right).
$$

I linearly project the target probabilities onto adjacent support atoms and minimise
the per-item cross-entropy

$$
\ell_i=-\sum_j [\Phi_zT Z_{\bar\theta}]_j\log p_{\theta,j}(b_t,a_t).
$$

My tests verify that the projection conserves probability, including when a target
falls exactly on an atom. The C51 support must cover plausible returns; when I
change rewards or the horizon, I monitor the support-clipping frequency rather
than assuming the original bounds remain adequate.

## Network and exploration

I encode the eight spatial maps with two convolutional layers, the second using
stride two, and flatten their output. I then concatenate the complete posterior
and budget scalars. Separate dueling streams form

$$
Z(b,a)=V(b)+A(b,a)-|\mathcal A|^{-1}\sum_{a'}A(b,a').
$$

Factorised Gaussian NoisyLinear layers learn the exploration scale. During
evaluation I disable the learned noise and mask infeasible Q-values to
$-\infty$. If demonstrations are disabled, replay warm-up uses uniformly random
valid actions.

## Demonstration pretraining

For the primary dense-field study, my path-aware greedy-information planner
provides 128 training-only demonstration trajectories. In addition to the C51
loss, expert transitions receive the DQfD large-margin term

$$
J_E=\max_{a\in\mathcal A(b)}[Q_\theta(b,a)+l(a_E,a)]-Q_\theta(b,a_E),
$$

where $l(a_E,a_E)=0$ and $l(a_E,a)=0.8$ otherwise. I apply 1,000
pretraining updates and then continue with off-policy Rainbow learning. The replay
capacity is large enough to retain the prescribed demonstration and online
transitions throughout the 30,000-step dense study.

This dependence on a strong model-based demonstrator is part of the method. The
result should be interpreted as planner-guided learning, not as evidence that an
uninformed neural agent discovered the sampling strategy independently. The
rare-wrackline stress test strengthens this dependence further, using 5,000
pretraining updates and demonstration-loss weight 10 to avoid repeated sampling of
one high-value cell. I describe the implementation as a Rainbow-DQfD composite,
not as the canonical Atari Rainbow protocol.

## Replay and optimisation

I accumulate five-step transitions without crossing episode boundaries. The
prioritised replay tree samples item $i$ according to

$$
P(i)=p_i^\alpha/\sum_k p_k^\alpha,\qquad
w_i=\frac{(NP(i))^{-\beta}}{\max_j(NP(j))^{-\beta}},
$$

where $p_i=\ell_i+10^{-6}$. I anneal beta from 0.4 to 1. I store spatial maps
as float16 and convert them to float32 for learning; posterior sufficient
statistics remain float32. I clip the gradient norm at 10 and hard-update the
target network every 1,000 environment steps.

## Deliberate departures and limitations

- Canonical Rainbow combined the same components in a particular Atari protocol.
  I use its algorithmic components without claiming identical hyperparameters or
  benchmark conditions.
- I do not use recurrent R2D2 or IQN because the complete Bayesian belief is
  observed under my assumed model.
- Deep nonlinear off-policy learning has no global convergence guarantee. An exact
  Bayesian reward does not prove that the learned neural policy is optimal.
- The checked dense-field experiment uses three training seeds. Its uncertainty
  estimates therefore include only limited between-agent variation.
- The training and evaluation profiles are generated procedurally from the same
  model family used by the Bayesian belief. Performance on this well-specified
  synthetic problem does not establish robustness to real or out-of-distribution
  beaches.
- With fixed features and homoscedastic Gaussian noise, covariance reduction is a
  function of sampling locations rather than observed concentration values. The
  experiment primarily tests path planning, not observation-adaptive discovery.
- The greedy-information baseline has direct model access and slightly outperforms
  Rainbow-DQfD in the primary experiment. I treat that outcome as a substantive
  result rather than evidence of RL optimality.

Primary sources: [Rainbow](https://doi.org/10.1609/aaai.v32i1.11796),
[C51](https://doi.org/10.48550/arXiv.1707.06887),
[Double DQN](https://doi.org/10.1609/aaai.v30i1.10295),
[prioritised replay](https://doi.org/10.48550/arXiv.1511.05952), and
[NoisyNet](https://openreview.net/forum?id=zgUEeXkag9). My demonstration objective
follows [Deep Q-learning from Demonstrations (Hester et al.,
2018)](https://doi.org/10.1609/aaai.v32i1.11757).
