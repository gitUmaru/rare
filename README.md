# RARE: Rare-event Analysis of Reverse-diffusion Error

Project 03 starter: do fast diffusion samplers lose rare outcomes?

## Question
A generative model may represent a rare kind of example correctly, yet a fast sampling procedure may produce that kind too often or too rarely. Is the error caused by how rare it is, how close it is to other kinds, or where the sampler spends its steps?

Real images make these explanations hard to separate. Here, points come from a known two-dimensional distribution, so its correct score is available. The score tells the sampler which direction increases probability. Using the exact score first lets you study numerical sampling error without confusing it with a poorly trained network.

## Shared data
Each distribution is a mixture of three circular Gaussians, with standard deviation 0.35. Two components are common; the third has small mixture weight. Development cases cross rare weights 2% and 8% with separations 0.8 and 2.0, using seeds 940–942. Final cases use weight/separation pairs (4%, 1.4) and (1%, 2.8), again with three seeds. All 18 cases are fixed and shared across groups.

Each file includes 6,000 clean training points, 50,000 independent reference points and 5,000 fixed starting points for sampling. It also includes exact distribution parameters. The starting distribution is sampled exactly after adding Gaussian variance 16; it is not approximated by one large Gaussian. Development and final cases use independent random streams.

A generated point has no observed latent component label. The starter therefore measures a concrete event: the point is closest to the rare component's mean. **This event's probability is not generally the rare component's mixture weight**, especially when components overlap. We estimate its correct probability from the reference points. State this distinction in your report.

## First afternoon
Use Python 3.11 or later. In Colab, upload and unzip this entire package, then change into its directory.

    pip install -r requirements.txt
    python check_data.py
    python starter.py --quick
    python starter.py --steps 8 16 32
    python starter.py --steps 8 16 32 --schedule quadratic --output quadratic.json

The starter follows a deterministic probability-flow ordinary differential equation from Gaussian noise variance 16 down to zero. Its derivative is minus one-half the score. Explicit Euler integration is deliberately simple: it uses the current direction to take each step. The supplied formula and code are a bridge from the course's diffusion models; no stochastic-calculus derivation is required.

The two schedules use identical numbers of score evaluations. The quadratic schedule spends more of its steps at small noise variance; it is a comparison, not a promised improvement. All schedules reuse the same starting points. The output includes rare-region frequency, reference and generated Wilson intervals, and basic overall mean/covariance errors. Intervals concern sampling variation; they are not numerical-error bounds. The starter does not provide an interval for the paired difference between schedules.

## Your research work
The starter comparison alone is insufficient. State one explanation of when rare-region error becomes large, and design a controlled experiment distinguishing it from another explanation. For example, change rare weight while holding geometry fixed, then change geometry while holding weight fixed. A result based only on plots of error against step count does not isolate a mechanism.

Develop a step-allocation rule, an error diagnostic, or a supported condition under which faster sampling is safe or fails. Compare methods at the same total number of score evaluations, including extra calls used to choose steps. If pilot samples or reference data inform a schedule, account for that information and its cost. Include at least one overall distribution-quality measure beyond rare frequency; a sampler could hit the target rare frequency while distorting everything else. The supplied mean/covariance errors are only basic checks, not a complete distance between distributions.

After establishing a result with the exact score, train a small noise-conditioned network on the supplied training points and repeat a limited comparison using its learned score. For additive variance v, generate noisy inputs x+sqrt(v)*epsilon and use target -epsilon/sqrt(v) for denoising-score training, with v bounded away from zero. Define how your network handles the smallest variance and keep the same trained network across sampler comparisons. This extension tests whether model error changes your conclusion; it need not reproduce a large image diffusion model.

Freeze your claim, method and settings before using final-case outcomes. Use all three seeds, report Monte Carlo uncertainty, and include a condition that could contradict your claim. A credible limitation or failure boundary is acceptable. Small apparent rare-frequency errors may simply be within sampling noise; quantify that instead of claiming success from a visually close plot.

## Background
Song et al. (2021), *Score-Based Generative Modeling through Stochastic Differential Equations*, for score models and the probability-flow ODE. Course connection: diffusion models, Gaussian noise and numerical sampling. The ODE bridge is provided in this package.

Code and synthetic data were created for this course; see `LICENSE.txt`.
