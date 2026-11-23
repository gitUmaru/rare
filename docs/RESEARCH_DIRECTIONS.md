# Research directions: turning RARE into a modest publication

Working notes on how to grow the current starter result into a small, solid paper.
The aim is a workshop-grade contribution, not a flagship result: a clear mechanism,
one useful diagnostic, an honest validation, and a learned-score replication showing
the story survives model error. Everything here builds on what the figures already
show, so none of it starts from scratch.

## What we already have

From the current experiments on the 18 cases:

1. A fast probability-flow ODE sampler (explicit Euler, few steps) pushes the rare
   region fraction too high.
2. The size of that error is controlled mostly by cluster overlap (separation $s$
   against $\sigma = 0.35$), not by how small the rare weight $w_3$ is.
3. The step-count sweep shows the error shrinking roughly like $\Delta v$, and the
   overlap cases stay far from the target even at 128 steps while separated cases land
   on it.
4. A proposed mechanism: where blobs overlap, the score field turns sharply near the
   Voronoi seam, and long Euler steps overshoot the turn.

That is already a mechanism plus a controlled comparison. The gap to a paper is a
quantitative diagnostic, a decision rule, a proper distribution-quality metric, and
the learned-score check.

## The core claim to defend

> Fast probability-flow ODE sampling inflates the probability of a rare region, and
> the error is governed by score-field curvature along the sampling path, which the
> cluster overlap sets, rather than by the rare weight itself.

Everything below is in service of making that sentence precise, measurable, and
falsifiable.

## Direction 1: a curvature-based failure heuristic (the centerpiece)

This is the "heuristic for when fast sampling fails" you want, made concrete.

The idea. Euler's local error on $dx/dv = -\tfrac12 s(x,v)$ scales with the second
derivative of the trajectory, which is the directional change of the score field
along the path. So a natural per-trajectory error proxy is the integrated bending of
the velocity field, something like

$$
C(x_T) = \int_0^{v_{\max}} \Big\lVert \frac{d}{dv}\, s\big(x(v), v\big) \Big\rVert \, dv,
$$

estimated cheaply from the steps the sampler already takes (finite differences of the
score it is evaluating anyway). High $C$ means a bent path, which means Euler
overshoots, which means error.

Concrete heuristic candidates, in increasing ambition:

- H1 (global): predict the rare-region error of a case from a single scalar, the
  overlap ratio $s/\sigma$ or a pilot estimate of mean curvature. Fit on the
  development cases, test on the final cases. This is a sanity-level result but it is
  honest and easy to report.
- H2 (per-region): the rare region sits near a Voronoi seam, so measure curvature
  only along trajectories that land in or near the rare cell. Claim: the rare-region
  error tracks seam curvature better than it tracks $w_3$.
- H3 (per-trajectory, actionable): use $C(x_T)$ to flag which individual starting
  points will be mis-integrated, and show the flagged points are the ones that cross
  cells incorrectly under coarse steps.

Why it is publishable at a modest level: it converts "overlap causes error" from a
plot into a predictor you can validate out of sample, with a clear mechanism
(curvature) behind it. The novelty is narrow but real, since most fast-sampler work
reports aggregate FID and does not localize error to a rare region.

## Direction 2: a curvature-aware step-allocation rule

The README explicitly invites a step-allocation rule compared at equal score
evaluations. The heuristic from Direction 1 gives one for free.

- Baseline schedules: uniform and quadratic (already in the starter), plus a
  standard log-SNR / EDM-style spacing as a stronger baseline.
- Proposed rule: place more steps where the estimated curvature $\lVert d s/dv\rVert$
  is largest, using a cheap pilot pass. This is a 1D optimal-step-size idea applied to
  the noise axis.
- Fair accounting: charge the pilot-pass score evaluations against the budget, as the
  README demands. Report error at matched total evaluations, not matched step counts.

The honest framing is a comparison, not a promised win. A credible outcome is
"curvature-aware spacing helps in the overlap regime and ties elsewhere," which is a
perfectly reportable result and matches the README's "safe or fails" language.

## Direction 3: distribution quality beyond the rare fraction

The README is firm that a sampler could hit the rare frequency while distorting
everything else, so we need at least one real distance between distributions.

- Sliced Wasserstein distance between generated and reference samples (cheap in 2D,
  defensible, standard).
- Energy distance or MMD with an RBF kernel as a second opinion.
- Keep the existing mean-L2 and covariance-Frobenius as basic checks, but do not lean
  on them as the headline.

Reportable finding to look for: fast sampling can land near the right rare fraction at
some step counts while the Wasserstein distance is still large, i.e. the rare fraction
is a necessary but not sufficient quality signal. That is a useful cautionary result
on its own.

## Direction 4: the learned-score replication (small noise-conditioned network)

This is the second thing you want, and the README spells out the recipe.

- Model: a small MLP $s_\theta(x, v)$, two inputs for $x$ plus a scalar (or Fourier)
  encoding of the noise level $v$. A few hidden layers, trained per case (or one
  network across cases as a variant).
- Objective: denoising score matching. Draw $x_0$ from the training points, a noise
  level $v$, and $\varepsilon \sim \mathcal{N}(0, I)$, form $x_v = x_0 + \sqrt{v}\,\varepsilon$, and
  regress $s_\theta(x_v, v)$ onto the target $-\varepsilon / \sqrt{v}$, with $v$ bounded
  away from zero. Define the smallest-$v$ behavior explicitly (clamp $v \ge v_{\min}$)
  and keep one frozen network across all sampler comparisons.
- The key experiment: repeat the fast-vs-slow and curvature-heuristic comparisons with
  $s_\theta$ in place of the exact score. Two outcomes are both worth reporting:
  - If the overlap-driven error persists, the conclusion is robust to model error,
    which strengthens the whole paper.
  - If the learned score adds its own bias near the rare seam (likely, since training
    points are scarce there), then you get a second, interesting finding: model error
    and sampler error compound in exactly the low-density region that matters.
- Diagnostic tie-in: measure the learned score's error $\lVert s_\theta - s \rVert$ as
  a function of position and show whether it is worst in the rare cell. This connects
  the learned-score section back to the curvature story.

## Direction 5: validation and honesty (what makes it solid rather than flashy)

- Freeze the claim, method, and settings on the development cases before touching the
  final cases, as the README requires. Write the frozen protocol down in the repo.
- Use all three seeds everywhere; report Monte Carlo error bars (the Wilson intervals
  are already there for the rare fraction, add bootstrap bars for the Wasserstein
  numbers).
- Include a falsifier: a regime where the claim should break. Candidate: a case with
  high overlap but a symmetric configuration where the rare seam is short, predicting
  low error despite low $s$. If the heuristic still predicts correctly there, that is
  strong support; if not, report it as the failure boundary.
- Separate sampling-noise error from numerical error explicitly, since the README
  warns that small rare-fraction gaps can be within sampling noise.

## A realistic scope for the paper

A tight, honest package:

1. Mechanism: overlap sets score curvature, curvature sets Euler error, shown with the
   controlled weight-vs-geometry experiment already in hand.
2. Diagnostic: a curvature proxy that predicts rare-region error out of sample
   (develop to final).
3. Rule: curvature-aware step allocation at matched evaluations, reported as a win in
   the overlap regime and a tie elsewhere.
4. Robustness: a second distribution metric (sliced Wasserstein) and the learned-score
   replication showing the story holds, plus where model error compounds it.
5. Limits: one stated falsifier and a clear sampling-noise accounting.

That is four small results and a mechanism, which is plenty for a workshop paper or a
short venue, and none of it requires a large model or heavy compute.

## Literature review options

Grouped by role in the paper. These are starting points; read abstracts and recent
citing work before committing.

### Foundations (method and framing)
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021), *Score-Based Generative
  Modeling through SDEs*, arXiv:2011.13456. The probability-flow ODE the starter uses.
- Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239.
- Karras, Aittala, Aila, Laine (2022), *Elucidating the Design Space of Diffusion
  Models (EDM)*, arXiv:2206.00364. Noise schedule and step spacing as explicit design
  axes; the right place to ground Direction 2.

### Fast samplers / solvers (baselines and related rules)
- Song, Meng, Ermon (2021), *Denoising Diffusion Implicit Models (DDIM)*,
  arXiv:2010.02502. First widely used few-step deterministic sampler.
- Lu, Zhou, Bao, Chen, Li, Zhu (2022), *DPM-Solver*, arXiv:2206.00927, and
  DPM-Solver++ / DPM-Solver-v3 (arXiv:2310.13268). Higher-order solvers; cite as the
  stronger baselines our simple Euler is deliberately below, and as the natural "does a
  better solver remove the rare-region bias" comparison.
- Zhang, Chen (2023), *Fast Sampling of Diffusion Models with Exponential Integrator
  (DEIS)*, arXiv:2204.13902. Another training-free solver with discretization-error
  analysis.

### Rare / minority samples (the phenomenon)
- Um, Lee, Ye (2023), *Don't Play Favorites: Minority Guidance for Diffusion Models*,
  arXiv:2301.12334. Diffusion bias toward high-density regions; the closest prior work
  to our question, though it is model-side rather than sampler-side.
- Um, Ye (2024), *Self-Guided Generation of Minority Samples (ECCV 2024)*,
  arXiv:2407.11555. Follow-up; shows where the low-density-sample literature is now.

### Convergence theory (why error scales as it does, with exact or near-exact score)
- Chen et al. (2022), *Sampling is as easy as learning the score*, arXiv:2209.11215.
- Chen, Chewi, et al., *A Sharp Convergence Theory for the Probability Flow ODEs of
  Diffusion Models*. The on-point theory for our exact-score ODE sampler, in TV
  distance; use to justify the $\Delta v$ error framing.
- *A Simple Analysis of Discretization Error in Diffusion Models* (2025),
  arXiv:2506.08337. Clean, recent Euler-discretization error treatment to cite for the
  curvature-error link.

### Metrics (for Direction 3)
- Standard references for sliced Wasserstein distance and for MMD / energy distance as
  two-sample distribution distances. Pick the canonical citations your venue expects.

## Open questions to settle before writing

- Is the curvature proxy cheap enough to beat its own cost in the step-allocation rule,
  once pilot evaluations are charged?
- Does the rare-region error correlate better with seam curvature or with a plain
  overlap ratio $s/\sigma$? If the plain ratio is nearly as good, the simpler story is
  the better paper.
- Does the learned score fail specifically in the rare cell, and can we show it with
  the $\lVert s_\theta - s \rVert$ heatmap?
- What is the cleanest falsifier case we can construct within the fixed data, or do we
  need one extra synthetic configuration to make the failure boundary sharp?
