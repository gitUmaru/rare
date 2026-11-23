# Literature review: rare-event behavior in diffusion models

Reading notes for RARE. The goal is to place our question (does fast probability-flow
ODE sampling distort how often a rare outcome appears) against what is already known,
and to pull out ideas we can borrow for detecting the same failure in real data.

The core of the review is the two minority-sample papers from the KAIST group, read in
full. The other groups are summarized more briefly, since we covered them in the
research-directions notes.

## How this connects to RARE

RARE studies a sampler-side error: with an exact score, few Euler steps inflate the
probability of a rare Voronoi region, driven by cluster overlap (score-field
curvature). The minority-sample literature studies a model-side error: a trained score
is biased against low-density samples. These are two different error sources, but they
land in the same place (the rare region) and, as shown below, they are measured by
closely related quantities. That overlap is what makes the minority-sample work the
most useful prior art for us, and the source of the detection ideas at the end.

## Foundations (method and framing)

- Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239.
  The DDPM forward/reverse process and the $\varepsilon$-prediction training objective.
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021), *Score-Based Generative
  Modeling through SDEs*, arXiv:2011.13456. The SDE view and the probability-flow ODE
  that the RARE starter integrates.
- Karras, Aittala, Aila, Laine (2022), *Elucidating the Design Space of Diffusion
  Models (EDM)*, arXiv:2206.00364. Separates noise schedule, step spacing, and solver
  as independent design axes. Note for us: EDM observes that under strong noise the
  denoiser output collapses toward the dataset mean, the same effect the minority
  papers build on.

## Fast samplers and solvers (baselines)

- Song, Meng, Ermon (2021), *DDIM*, arXiv:2010.02502. First widely used deterministic
  few-step sampler.
- Lu et al. (2022), *DPM-Solver*, arXiv:2206.00927, plus DPM-Solver++ and
  DPM-Solver-v3 (arXiv:2310.13268). High-order ODE solvers reaching good samples in
  10 to 20 evaluations. These are the stronger baselines our deliberately simple Euler
  sits below, and the natural "does a better solver remove the rare-region bias" test.
- Zhang, Chen (2023), *DEIS*, arXiv:2204.13902. Training-free exponential-integrator
  solver with explicit discretization-error analysis.

## Convergence theory (why the error scales as it does)

- Chen et al. (2022), *Sampling is as easy as learning the score*, arXiv:2209.11215.
  With an $L^2$-accurate score, diffusion samples essentially any distribution
  efficiently.
- Chen, Chewi, et al., *A Sharp Convergence Theory for the Probability Flow ODEs of
  Diffusion Models*. Non-asymptotic convergence for the exact-score ODE sampler in TV
  distance; the right citation for our $\Delta v$ error framing.
- *A Simple Analysis of Discretization Error in Diffusion Models* (2025),
  arXiv:2506.08337. Clean, recent treatment of Euler-discretization error.

## Rare / minority samples (the phenomenon), read in depth

Two papers by Soobin Um and Jong Chul Ye (KAIST), with Suhyeon Lee on the first. Both
attack the fact that diffusion models under-produce low-density ("minority") samples.
Together they give us a mechanism, a cheap metric, and a theoretical link to
score-matching error.

### Um, Lee, Ye (2023/2024), *Don't Play Favorites: Minority Guidance* (ICLR 2024), arXiv:2301.12334

The mechanism. The optimal score is a posterior average over the clean data that could
have produced a noised point:

$$
s^\star_\theta(x_t, t) = \mathbb{E}_{q(x_0 \mid x_t)}\big[\nabla_{x_t} \log q(x_t \mid x_0)\big].
$$

Because it is an average, and because the posterior $q(x_0 \mid x_t)$ is dominated by
the plentiful majority samples, the score points toward the majority manifold. Feed
that score into Tweedie's formula for the posterior mean,

$$
\hat{x}_0 = \mathbb{E}[x_0 \mid x_t] = \frac{1}{\sqrt{\alpha_t}}\big(x_t + (1 - \alpha_t)\, s^\star_\theta(x_t, t)\big),
$$

and the denoiser restores majority structure well but drags minority inputs toward the
dataset mean. The bias is worst at large noise ($t \to T$), exactly where EDM also
notes the denoiser collapses to the mean. This is a clean statement of why generation
is majority-oriented: it is baked into the Bayes-optimal denoiser, not just a training
artifact.

The metric (minority score). Measure how badly a clean sample survives a
noise-then-denoise round trip:

$$
l(x_0) = \mathbb{E}_{q(x_t \mid x_0)}\big[\, d(x_0, \hat{x}_0)\,\big],
$$

with $d$ a squared error or LPIPS. Minority samples lose more information in the round
trip, so they score high. It costs essentially one denoiser evaluation, far cheaper
than the iterative outlier detectors it competes with, and it tracks standard outlier
measures (Average k-NN, Local Outlier Factor) on real data.

The theoretical link we care about most. With squared-error $d$, minority score equals
(up to a timestep weight) the noise-prediction error:

$$
l(x_0) = \tilde{w}_t \, \mathbb{E}_{\varepsilon}\big[\lVert \varepsilon^\star_\theta(x_t, t) - \varepsilon \rVert_2^2\big].
$$

So a sample's "minority-ness" is the same quantity as the model's score-matching error
on it. Scarce regions are both low-density and hard to denoise, and those are the same
places. For RARE this is the bridge: our sampler error and their model error are both
large in the region where the score is hardest to get right.

The fix (minority guidance). Train a light classifier on minority-score levels and use
classifier guidance to steer generation toward a chosen minority level. It needs no
minority labels (label-agnostic), and the paper shows it holds up on medical imaging.

### Um, Ye (2024), *Self-Guided Generation of Minority Samples* (ECCV 2024), arXiv:2407.11555

The problem with the above. Minority score is defined on clean $x_0$, so using it as
guidance during sampling needs either a separately trained classifier (the ICLR paper)
or an expensive external one (Sehwag et al. 2022 report 40 V100-days for an
ImageNet-64 classifier). This paper removes that dependency.

The idea. Define an inference-time metric using only the pretrained model, via two
Tweedie passes. From the current noisy $x_t$, denoise to $\hat{x}_0$, re-noise it to a
lighter level $s$, denoise again to $\hat{\hat{x}}_0$, and measure the gap:

$$
\tilde{\mathcal{L}}(x_t; s) = \mathbb{E}\big[\, d\big(\hat{x}_0(x_t),\ \hat{\hat{x}}_0(\hat{x}_s(x_t))\big)\,\big].
$$

This is minority score transported onto the noisy iterate: the clean surrogate
$\hat{x}_0$ stands in for $x_0$. The gradient of $\tilde{\mathcal{L}}$ is used as
energy-based guidance pushing toward low density. Applying the exact log-likelihood
gradient instead would shove samples off-manifold, so the reconstruction-discrepancy
form is deliberately gentler.

The practical knobs. The guidance is applied intermittently (roughly every 5 steps),
which keeps the cost low, and its strength is scheduled over timesteps. The method
beats the classifier-based approaches on minority metrics (Average k-NN) and on FID,
using only a pretrained model.

### What these two give RARE

- A mechanism (majority-biased Bayes-optimal denoiser) that explains why real diffusion
  under-produces rare samples, which is the model-side analogue of our sampler-side
  overshoot.
- A cheap, label-free, pretrained-only rarity signal (minority score), with an
  inference-time version that works on noisy iterates.
- The identity minority score ≈ score-matching error, which is the key lever for the
  detection ideas below.

One caveat on direction. Their failure is under-production of minorities (majority
pull), while our exact-score experiments show over-production of the rare region
(overshoot into the Voronoi cell). These are not contradictory: theirs is model error
from a biased learned score, ours is discretization error from coarse steps, and in a
real model both act at once. Reconciling the sign in a combined setting is itself an
interesting question for the learned-score extension.

## Rare-event sampling as a target (adjacent)

- *Steering Diffusion Models to Rare Events with Sequential Monte Carlo* (2024).
  Diffusion plus SMC to estimate rare-event probabilities, with the explicit $1/p$
  Monte Carlo cost scaling that our Wilson intervals also run into.

## Most promising ideas for catching this in real data

The RARE setup has a luxury real data never has: a known $p_0$, an exact score, and a
clean reference sample to define the true rare fraction. In real data all three are
gone. The minority-sample papers are useful precisely because their tools assume none
of them. Here are the directions that look most likely to transfer, roughly in order of
promise.

1. Minority score as a drop-in rarity label. Use the noise-then-denoise reconstruction
   gap $l(x_0) = \mathbb{E}[d(x_0, \hat{x}_0)]$ to rank real samples by rarity with no
   labels and one denoiser pass. This replaces our exact Voronoi membership, which does
   not exist in real data. Then measure the rare fraction as the share of generated
   samples above a minority-score threshold, and compare a fast sampler against a slow
   one on that fraction. This is the most direct port of the RARE experiment to real
   images.

2. Score-matching error as a per-region failure map. Since minority score equals the
   score-matching error, the regions where the learned score is least trustworthy are
   the rare regions. Build a heatmap of $\lVert \varepsilon_\theta - \varepsilon \rVert^2$
   (or the minority score) over the data, and predict that fast-sampling distortion
   concentrates exactly there. In RARE we can validate this against ground truth; in
   real data it becomes the detector itself.

3. Tweedie-discrepancy as a curvature proxy. Our mechanism is score-field curvature
   along the path; the ECCV metric is a double-Tweedie discrepancy on noisy iterates.
   Both are cheap and both spike near low-density seams. Worth testing whether the
   double-Tweedie gap $\tilde{\mathcal{L}}(x_t; s)$ correlates with our curvature proxy
   on the synthetic cases, where we can check it against the true error. If it does,
   we get a curvature diagnostic that runs on real pretrained models unchanged.

4. Fast-vs-slow divergence on the rarity-scored tail. Rank generated samples by minority
   score, then compare the fast and slow samplers only on the top-tail (rarest) bucket,
   using a two-sample distance (sliced Wasserstein or MMD) restricted to that bucket.
   The prediction is that fast and slow agree in the bulk and diverge in the tail, which
   would be a clean, real-data-friendly signature of the effect without any ground-truth
   density.

5. Intermittent-guidance cost model as a safety knob. The ECCV paper shows minority
   guidance only needs to fire every few steps to work. The same intermittency could
   gate a "spend extra steps here" rule: run the cheap rarity metric occasionally, and
   only when it flags a trajectory as heading into a low-density region do you refine
   the local step size. This ties the detection idea back to the step-allocation rule
   in the research-directions notes, with a cost model borrowed from a published method.

### The honest risk

Minority score and the double-Tweedie metric both depend on the learned denoiser, so in
real data they measure "where the model thinks density is low," not true density. If the
model's density estimate is itself wrong in the tail (likely, since the tail is
undertrained), the detector inherits that error. RARE can quantify this gap directly,
because we have the exact score to compare against, which is a genuinely useful thing
our synthetic setup can contribute back to the real-data detection question.

## Open questions this review raises

- Does the double-Tweedie discrepancy track our score-field curvature proxy on the
  synthetic cases? If yes, it is a real-data-ready curvature detector.
- In a combined learned-score plus fast-sampler setting, does the majority pull (under-
  production) cancel or compound the Euler overshoot (over-production)? What is the net
  sign?
- Is minority-score-thresholded rare fraction a stable enough target to replace Voronoi
  membership, given it depends on a model and a threshold?

## Second-pass review: sampler-side error (the part RARE actually owns)

The minority papers are model-side. A second search turned up the sampler-side
literature that sits much closer to RARE, and it sharpens what our contribution is.

### Zhang, Liao, Wu, Zou (2025), *On the Collapse Errors Induced by the Deterministic Sampler*, arXiv:2508.16154

The most directly comparable paper. They show deterministic (ODE) samplers over-
concentrate samples into local regions, which they call *collapse error*, and they
measure it with TID (Tail Index Difference): compare the heaviness of the
nearest-neighbor-count tail between sampled and training data using Hill's estimator.
ODE samplers show high TID, stochastic (SDE) samplers stay low. They demonstrate it on
2D toys (chessboard, spiral) and on CIFAR-10 / CelebA.

The crucial detail for us: they attribute collapse to the *interplay* between
deterministic sampling dynamics and score *misfitting in the high-noise regime* (a
"see-saw" effect, where fitting low noise well hurts high-noise fit). In other words,
their cause is a mixture of sampler error and learned-score error, and they explicitly
say they cannot cleanly separate the two. They even write the exact Gaussian-mixture
score to analyze it, then study the learned version.

Why this matters for RARE. We have the exact score, so we hold the learned-score error
at exactly zero. That means we can measure the pure-discretization contribution to
over-concentration and to rare-region distortion, which this paper cannot isolate. Our
mechanism (overshoot where the score field curves, set by cluster overlap) is a
sampler-only statement; theirs needs the see-saw misfitting. RARE is the controlled
experiment their setup is missing.

Direction note: they report ODE *over-concentration* (collapse), RARE reports ODE
*over-production* of the rare Voronoi cell. These are consistent, since pushing mass
across an overlap seam into the rare wedge is a local concentration. Worth checking
whether TID on our generated samples rises in the same cases where our rare-region
error is largest.

### Hurault, Moreau, Peyre (2026), *Geometry-Aware Discretization Error of Diffusion Models*, arXiv:2605.08392

The analytical tool RARE's curvature heuristic has been missing. Instead of loose
worst-case convergence bounds, they derive asymptotically exact small-stepsize
expansions of the Euler discretization error (weak and Frechet), with closed-form
formulas for Gaussian data and a *perturbative extension to Gaussian scale mixtures*.
The error is written in terms of the target's covariance spectrum, and they use it to
pick noise and rescaling schedules.

Why this matters for RARE. Our data is literally a Gaussian mixture, which their
perturbative GSM analysis covers. Their expansion gives a principled, computable error
functional we can evaluate exactly on our cases, rather than our hand-rolled curvature
proxy. We can either validate our proxy against their expansion, or adopt their
expansion directly and specialize it to the rare region.

### Other sampler-side items worth citing

- *A Simple Analysis of Discretization Error in Diffusion Models* (2025),
  arXiv:2506.08337. Clean Euler error treatment for the $\Delta v$ framing.
- *When Higher Order Hurts: Pre-Asymptotic Order Collapse* (ECCV 2026). Shows
  higher-order solvers can underperform Euler at practical step sizes because of a
  discretization-learning interaction term. Relevant if we claim "use a better solver,"
  since with an exact score that interaction term vanishes, which is again a clean
  control only RARE has.
- *Minimizing Trajectory Curvature of ODE-based Generative Models*, arXiv:2301.12003.
  Ties truncation error directly to trajectory curvature, the exact mechanism we
  propose, at the training-objective level rather than per-region.
- Adaptive step-size / PI-control solvers (arXiv:2610.03034 and related). Prior art for
  the step-allocation rule; charge their per-step error estimator against the budget.

## Clear next step

The object of the study is rare-event fidelity: does fast sampling report the rare
outcome at the right rate and shape, by how much is it off, and when is it safe. The
cause (geometry, step placement, stochastic vs deterministic) is left open and reported,
not assumed. Discretization is one candidate cause, not the headline.

The gap the literature leaves open, stated as our question:

> Nobody measures whether a fast sampler preserves the frequency and shape of a known
> rare outcome. The minority-sample work generates more rare samples; the fast-sampler
> work measures overall quality (FID). Neither asks whether cutting steps makes a known
> rare event appear too often or too rarely, and what that error tracks.

RARE answers exactly that, because the rare outcome and its true rate are known by
construction. Concretely, four steps in priority order.

1. Nail the rare-event fidelity curve. For each case, plot the rare-region error (and a
   Wilson interval) against sampling budget, and report the smallest budget at which
   the error is statistically within sampling noise of the true rate. This is the
   headline deliverable: a "when is fast sampling safe for the rare event" boundary per
   case. We already have most of this; what is missing is the explicit safe-budget
   threshold and its uncertainty.

2. Report what the error tracks, cause-agnostic. Regress the rare-event error against
   candidate explanatory variables (overlap ratio $s/\sigma$, rare weight $w_3$, a
   pilot curvature estimate, step count) and report which one it follows. The current
   finding says overlap dominates and rarity barely matters; this step makes that a
   measured ranking rather than an eyeballed one, and leaves room for other causes to
   win. Fit on development cases, confirm on final.

3. Check that fidelity is not just the rate. A sampler could hit the right rare
   frequency while deforming the rare cluster's shape. Add a rare-conditioned
   distribution distance (sliced Wasserstein restricted to points in the rare region,
   or the TID tail metric from the collapse paper computed on the rare subset). Claim
   to test: the rare rate can look correct while the rare region's shape is still
   wrong, so rate alone overstates fidelity.

4. Does model error change the rare-event conclusion? Train the small noise-conditioned
   net (the README extension) and repeat steps 1 and 3 with the learned score. Because
   we can compare against the exact score, we can say whether a learned score makes the
   rare event appear more or less faithfully than the exact-score sampler at the same
   budget. This directly tests whether the fidelity boundary from step 1 survives model
   error.

The framing for a paper: "fast samplers are tuned and judged on overall quality, but
nobody checks whether a known rare outcome survives the speedup; using a mixture with a
known rare event and exact score, we measure the rare-event error against budget, find a
safe-budget boundary, report that the error tracks geometry far more than rarity, and
show whether a learned score shifts the boundary." The rare event is the object; the
cause is a finding, not an assumption.

How the closest papers relate, under this framing. The minority-sample papers share our
object (the rare case) but ask a generation question, not a fidelity one, so they are
the natural relate-and-contrast. The collapse and geometry-aware papers supply tools
(the TID tail metric, a geometry-indexed error) we can borrow for steps 2 and 3, but
their object is overall concentration or total error, not the rare event, so they are
support rather than the template.

### Caveats on the second-pass sources

Some of these carry arXiv dates that read as future (2026), likely version or metadata
artifacts; I read the collapse and geometry-aware papers directly and the methods are as
summarized, but verify publication venue and final numbers before citing.
