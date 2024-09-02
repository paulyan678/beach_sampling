# Research methodology

## 1. Decision problem

I study autonomous microplastic sampling as a finite-horizon decision problem. In
each episode, I draw an unseen beach case \(c\) and give the robot \(T=64\)
transitions. The robot can move in the four cardinal directions, collect a sample,
or wait safely. I mask water and unsafe cells, grid boundaries, and the sample
action after the budget is exhausted before the policy selects an action.

The state contains the case covariates, robot pose, visit and sample maps, elapsed
time and remaining budgets, and the complete Bayesian posterior
\((m_t,\Sigma_t)\). Although the contaminant field is only partially observed, this
posterior is a sufficient information state under my assumed linear-Gaussian model.
I therefore formulate the belief process as the MDP

\[
\mathcal M=(\mathcal B,\mathcal A,P_b,r,\gamma,T),\qquad
J(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

This Markov property is conditional on the model being correctly specified. It
does not establish that the same belief would be sufficient for real microplastic
transport with unmodelled dynamics.

## 2. Physics-guided procedural cases

For the checked experiments, I use a procedural generator rather than claiming to
simulate calibrated hydrodynamics. It constructs a two-dimensional beach elevation
from a cross-shore slope, a Gaussian berm, a meandering shoreline, and smooth
alongshore modes. From this surface I derive explicitly named proxy covariates for
run-up and wrack deposition, low-energy retention, morphology curvature, and
longshore forcing. These cases are controlled stress tests, not substitutes for
field measurements or complete XBeach hydrodynamics.

In the primary dense-field experiment, radial basis functions span the field:

\[
\tilde\phi_j(x)=\exp\left(-\frac{\lVert x-c_j\rVert^2}{2\ell^2}\right),\qquad
\phi_j(x)=\frac{\tilde\phi_j(x)[0.55+0.9q(x)]}
{\lVert\tilde\phi_j(\cdot)[0.55+0.9q(\cdot)]\rVert_2},
\]

where \(q(x)\in[0,1]\) is the deposition-risk covariate. I use ridge
projection of \(-0.7+2q(x)\) to obtain the prior mean \(m_0\), then generate each
held-out field once as

\[
\theta_c\sim\mathcal N(m_0,\sigma_0^2I),\quad
f_c(x)=\phi(x)^T\theta_c,\quad
y_x=f_c(x)+\epsilon,\quad\epsilon\sim\mathcal N(0,\sigma_n^2).
\]

The statistical model represents log concentration, so exponentiating the latent
field produces a positive concentration. My benchmark optimises information about
the latent map rather than captured mass, hotspot yield, or a field-calibrated
environmental endpoint.

## 3. Exact Bayesian update and information reward

I maintain the conjugate posterior
\(\theta\mid D_t\sim\mathcal N(m_t,\Sigma_t)\). For the feature vector at a
candidate cell, I define

\[
s_x^2=\phi_x^T\Sigma_t\phi_x+\sigma_n^2,\quad
K_x=\Sigma_t\phi_x/s_x^2.
\]

After observing \(y_x\), I update the belief exactly:

\[
m_{t+1}=m_t+K_x(y_x-\phi_x^Tm_t),\qquad
\Sigma_{t+1}=\Sigma_t-\frac{(\Sigma_t\phi_x)(\Sigma_t\phi_x)^T}{s_x^2}.
\]

The sampling component of the reward is the entropy reduction in nats:

\[
r_t^{\mathrm{sample}}
=\tfrac12\left(\log\det\Sigma_t-\log\det\Sigma_{t+1}\right)
=\tfrac12\log\left(1+\frac{\phi_x^T\Sigma_t\phi_x}{\sigma_n^2}\right).
\]

I implement the rank-one covariance update, symmetrise the covariance after every
measurement, and test both positive semidefiniteness and the log-determinant
identity. Movement, revisits, waiting, and invalid actions retain explicit costs.

To improve delayed credit assignment, I use the path-aware potential
\(\Phi(b)=\max_x I_b(x)/(1+d(x_b,x))^{0.35}\), where the implementation uses
Manhattan distance as the shaping travel proxy. I add
\(F(b,b')=\gamma\Phi(b')-\Phi(b)\), with terminal potential set to zero. Because
the shaping discount matches the learner discount, this is potential-based shaping
and does not change the set of optimal policies for the specified MDP. The complete
transition reward is

\[
r_t=r_t^{\mathrm{sample}}-\lambda_m1[\mathrm{move}]
-\lambda_r1[\mathrm{revisit}]-\lambda_i1[\mathrm{invalid}]+\lambda_\Phi F(b_t,b_{t+1}).
\]

I never apply reward normalisation, clipping, shaping, or a target ratio to the
separately reported cumulative information.

Under fixed features and homoscedastic Gaussian noise, the covariance update—and
therefore information gain—depends on the locations sampled, not on the observed
concentration values. The present experiment is consequently informative path
planning in belief space, not a demonstration of observation-adaptive hotspot
discovery.

## 4. Observation and action spaces

I provide the agent with eight spatial channels: normalised elevation, deposition
prior, posterior predictive mean, posterior predictive standard deviation, sample
count, robot pose, traversability, and visit count. Scalar features contain
normalised time, sample, and path budgets together with \(m_t\) and every entry of
\(\Sigma_t\). I include the full covariance deliberately because a marginal
variance map alone is not a sufficient statistic for future spatial information
gain.

The action space contains four cardinal movements, one sample action, and one wait
action. The action mask prevents the evaluated policies from selecting infeasible
movements or sampling after the budget is exhausted.

## 5. Baselines

I compare the learned agent with four interpretable policies:

1. **Random:** selects uniformly among the currently valid primitive actions.
2. **Uniform waypoint:** draws reachable sample targets uniformly and follows
   shortest paths to them.
3. **Lawnmower:** follows a serpentine coverage ordering and selects evenly spaced
   targets using shortest-path routing.
4. **Greedy information:** recomputes exact information and chooses the reachable
   target maximising \(I_t(x)/(1+d(x_t,x))^{0.35}\), using graph distance.

I treat the greedy-information policy as the strongest baseline, not as a weak
straw comparison. With a fixed kernel and homoscedastic noise, covariance reduction
depends on locations rather than observed values, so the problem is close to
informative orienteering and model-based planning is naturally competitive. A
learned policy is not substantively useful merely because it beats primitive
random exploration.

## 6. Evaluation

For the primary dense-field study, I train three agents with seeds 11, 29, and 47.
I select checkpoints using 128 validation profiles with seeds 20,000--20,127, then
evaluate the selected policies once on 1,024 lockbox procedural profiles with seeds
50,000--51,023. Training, demonstration, validation, and lockbox seed namespaces
are disjoint, and every policy sees the same evaluation cases.

I treat an entire beach profile as the independent resampling unit. My report
includes:

- the mean, median, standard deviation, and a 10,000-draw hierarchical-bootstrap
  95% interval;
- paired differences and ratios to random, each with a bootstrap interval, and a
  20,000-draw paired sign-randomisation test;
- posterior log-concentration RMSE, unique samples, distance travelled, and invalid
  actions;
- example trajectories and aggregate sampling-frequency heatmaps.

When I evaluate several training seeds under one policy name, point estimates
average the agents within each profile, while confidence intervals resample both
training seeds and profiles hierarchically. The checked study contains only three
training seeds, so the intervals capture limited between-agent variability and
should not be interpreted as evidence of field generalisation. A stronger follow-up
would use at least the documented ten-seed, 2,048-profile extension and a separately
calibrated XBeach or field-backed test set.
