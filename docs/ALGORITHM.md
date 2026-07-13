# Algorithm: masked Rainbow-DQfD

There is no universally best reinforcement-learning algorithm. This implementation
uses a research-grade but compact Rainbow variant for a discrete belief MDP. Its
components are independently inspectable and covered by unit tests.

## Distributional Bellman target

The network represents a categorical return distribution on fixed atoms
\(z_i=v_{min}+i\Delta z\), \(i=0,\ldots,N-1\):

\[
Z_\theta(b,a)=\sum_i p_{\theta,i}(b,a)\,\delta_{z_i},\qquad
Q_\theta(b,a)=\sum_i z_i p_{\theta,i}(b,a).
\]

For an n-step replay item, the online network selects a valid action and the target
network evaluates it (Double DQN):

\[
a^*=\arg\max_{a\in\mathcal A(b_{t+n})}Q_\theta(b_{t+n},a),\qquad
Tz_i=\mathrm{clip}\left(R_t^{(n)}+\gamma^n(1-d_t)z_i,v_{min},v_{max}\right).
\]

The target probabilities are linearly projected onto adjacent support atoms. The
per-item loss is cross entropy

\[
\ell_i=-\sum_j [\Phi_zT Z_{\bar\theta}]_j\log p_{\theta,j}(b_t,a_t).
\]

Tests verify that projection conserves probability, including the exact-atom case.
The C51 support must cover plausible returns; support clipping frequency should be
monitored when changing reward or horizon.

## Network and exploration

Two convolutions, the second with stride two, encode and flatten the eight maps. The complete posterior
and budgets are concatenated afterward. Dueling streams form

\[
Z(b,a)=V(b)+A(b,a)-|\mathcal A|^{-1}\sum_{a'}A(b,a').
\]

Factorised Gaussian NoisyLinear layers learn exploration scale. Evaluation disables
noise and masks infeasible Q-values to \(-\infty\). Replay warm-up uses uniformly
random valid actions when no demonstrations are configured.

## Demonstration pretraining

The path-aware greedy-information policy supplies 128 training-only demonstrations.
In addition to the C51 loss, expert transitions receive the DQfD large-margin term

\[
J_E=\max_{a\in\mathcal A(b)}[Q_\theta(b,a)+l(a_E,a)]-Q_\theta(b,a_E),
\]

where \(l(a_E,a_E)=0\) and \(l(a_E,a)=0.8\) otherwise. The network receives 1,000
pretraining updates, then continues off-policy Rainbow learning. Replay capacity is
large enough that the prescribed demonstration and online transitions are retained
for the 30,000-step study. This is explicitly a Rainbow-DQfD composite, not the
canonical Atari Rainbow protocol.

## Replay and optimisation

Five-step transitions are accumulated without crossing episode boundaries. The
replay tree samples item \(i\) with

\[
P(i)=p_i^\alpha/\sum_k p_k^\alpha,\qquad
w_i=\frac{(NP(i))^{-\beta}}{\max_j(NP(j))^{-\beta}},
\]

where \(p_i=\ell_i+10^{-6}\). Beta anneals from 0.4 to 1. Float16 map storage is
converted back to float32 for learning; posterior sufficient statistics remain
float32. Gradient norm is clipped at 10. The target network is hard-updated every
1,000 environment steps.

## Deliberate departures and limitations

- Canonical Rainbow also combined the components in a particular Atari training
  protocol; “Rainbow” here describes the same algorithmic components, not a claim
  of identical hyperparameters or benchmark conditions.
- Recurrent R2D2/IQN is not used because the complete Bayesian belief is observed.
- Deep nonlinear off-policy learning has no global convergence guarantee. The exact
  Bayesian reward does not confer a proof that the neural policy is optimal.
- The greedy-information baseline exploits model knowledge and may outperform RL.
  That outcome is scientifically informative, not a failed experiment.

Primary sources: [Rainbow](https://doi.org/10.1609/aaai.v32i1.11796),
[C51](https://doi.org/10.48550/arXiv.1707.06887),
[Double DQN](https://doi.org/10.1609/aaai.v30i1.10295),
[prioritised replay](https://doi.org/10.48550/arXiv.1511.05952), and
[NoisyNet](https://openreview.net/forum?id=zgUEeXkag9).
The demonstration objective follows [Deep Q-learning from Demonstrations
(Hester et al., 2018)](https://doi.org/10.1609/aaai.v32i1.11757).
