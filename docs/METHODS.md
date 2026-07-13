# Methods

## 1. Decision problem

Each episode draws an unseen beach case \(c\) and lasts \(T=64\) transitions. The
robot has four cardinal movement actions, one sample action, and a safe wait action. Water/unsafe cells,
grid boundaries, and sampling after the budget is exhausted are masked before
action selection. A state contains the case covariates, robot pose, visit/sample
maps, time and budgets, and the complete Bayesian posterior \((m_t,\Sigma_t)\).

The latent field is partially observed, but this posterior is the sufficient
information state. The resulting belief process is an MDP

\[
\mathcal M=(\mathcal B,\mathcal A,P_b,r,\gamma,T),\qquad
J(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

## 2. Physics-guided procedural cases

The fallback backend creates a two-dimensional beach elevation from a cross-shore
slope, a Gaussian berm, a meandering shoreline, and smooth alongshore modes. It
then constructs explicitly named proxy covariates for run-up/wrack deposition,
low-energy retention, morphology curvature, and longshore forcing. These are a
stress-test generator, not a replacement for XBeach hydrodynamics.

Radial basis functions span the field:

\[
\tilde\phi_j(x)=\exp\left(-\frac{\lVert x-c_j\rVert^2}{2\ell^2}\right),\qquad
\phi_j(x)=\frac{\tilde\phi_j(x)[0.55+0.9q(x)]}
{\lVert\tilde\phi_j(\cdot)[0.55+0.9q(\cdot)]\rVert_2},
\]

where \(q(x)\in[0,1]\) is the deposition-risk covariate. Ridge projection of
\(-0.7+2q(x)\) gives prior mean \(m_0\). Each held-out field is generated once as

\[
\theta_c\sim\mathcal N(m_0,\sigma_0^2I),\quad
f_c(x)=\phi(x)^T\theta_c,\quad
y_x=f_c(x)+\epsilon,\quad\epsilon\sim\mathcal N(0,\sigma_n^2).
\]

The model is for log concentration; exponentiating gives a positive concentration.
The benchmark optimises map information, not hotspot yield.

## 3. Exact Bayesian update and reward

With current \(\theta\mid D_t\sim\mathcal N(m_t,\Sigma_t)\), define

\[
s_x^2=\phi_x^T\Sigma_t\phi_x+\sigma_n^2,\quad
K_x=\Sigma_t\phi_x/s_x^2.
\]

After observing \(y_x\),

\[
m_{t+1}=m_t+K_x(y_x-\phi_x^Tm_t),\qquad
\Sigma_{t+1}=\Sigma_t-\frac{(\Sigma_t\phi_x)(\Sigma_t\phi_x)^T}{s_x^2}.
\]

The sampling component of reward is exactly the entropy reduction in nats:

\[
r_t^{\mathrm{sample}}
=\tfrac12\left(\log\det\Sigma_t-\log\det\Sigma_{t+1}\right)
=\tfrac12\log\left(1+\frac{\phi_x^T\Sigma_t\phi_x}{\sigma_n^2}\right).
\]

The implementation uses the rank-one form, symmetrises after each update, and tests
positive semidefiniteness and the log-determinant identity. Travel, revisit, and
invalid-action costs are explicit configuration terms. A path-aware potential
\(\Phi(b)=\max_x I_b(x)/(1+d(x_b,x))^{0.35}\) supplies dense delayed-credit
feedback via \(F(b,b')=\gamma\Phi(b')-\Phi(b)\), with terminal potential zero.
This is potential-based shaping with the same discount as the learner, so it does
not change the set of optimal policies. The transition reward is

\[
r_t=r_t^{\mathrm{sample}}-\lambda_m1[\mathrm{move}]
-\lambda_r1[\mathrm{revisit}]-\lambda_i1[\mathrm{invalid}]+\lambda_\Phi F(b_t,b_{t+1}).
\]

No reward normalisation, clipping, shaping, or target ratio changes the separately
reported cumulative information.

## 4. Observation and action spaces

Eight spatial channels contain normalised elevation, deposition prior, posterior
predictive mean, posterior predictive standard deviation, sample count, robot pose,
traversability, and visit count. Scalar features contain normalised time/sample/path
budgets plus \(m_t\) and all entries of \(\Sigma_t\). Including the full covariance
is deliberate: a marginal variance map alone is not a sufficient statistic for
future spatial information gains.

## 5. Baselines

1. **Random:** uniformly selects among valid actions.
2. **Lawnmower:** follows a serpentine coverage ordering and selects evenly spaced
   targets using shortest-path routing.
3. **Greedy information:** recomputes exact information and chooses the reachable
   target maximising \(I_t(x)/(1+d(x_t,x))^{0.35}\).
4. **Rainbow:** the learned policy in `docs/ALGORITHM.md`.

The third policy is intentionally hard to beat. With fixed kernel and homoscedastic
noise, covariance reduction depends on locations rather than observed values; this
is close to an informative orienteering problem. A learned policy should not be
called useful if it only beats random.

## 6. Evaluation

The prespecified lockbox evaluation cases are seeds 50,000–51,023 and never appear
in training or validation. Every policy sees the same cases. The independent resampling unit is a
whole beach profile. The report includes:

- mean, median, standard deviation, and 10,000-draw hierarchical bootstrap 95% interval;
- paired difference/ratio (each with a bootstrap interval) to random and a
  20,000-draw paired sign-randomisation test;
- posterior log-concentration RMSE, unique samples, distance, and invalid actions;
- trajectories and aggregate sampling-frequency heatmaps.

When several training seeds are evaluated, point estimates average them within a
profile while confidence intervals resample training seeds and profiles
hierarchically. For publication-quality claims, run the 10-seed/2,048-profile
command in the README.
