# Report draft: Introduction and Related Work

> [!NOTE]
> Draft text for the Introduction and Related Work sections of `main.tex`.
> Each sentence is on its own line.
> The Introduction states the problem and its motivation only, with no methods or results.
> The Related Work describes prior work only and does not describe our approach or experiments.
> An earlier draft that includes experimental results is saved as `draft_with_results.md`.

> [!WARNING]
> For most cited works I read the abstract only, so check each claim against the full paper before submitting.
> Several entries have 2026 dates from search results, and their venues are not confirmed.

## 1 Introduction

Diffusion models have become a standard approach to generative modeling, and they now produce images, audio and video of high quality (Ho et al., 2020; Karras et al., 2022).
They are defined by a forward process that gradually adds Gaussian noise to data until only noise remains (Sohl-Dickstein et al., 2015; Song and Ermon, 2019).
Generation reverses this process, starting from pure noise and removing noise step by step.
In the continuous-time view, the reverse process is a differential equation whose drift depends on the score function, which is the gradient of the log density of the noised data (Song et al., 2021b).
This view includes a stochastic version and a deterministic version, called the probability flow ODE.
A neural network approximates the score, and a numerical solver integrates the equation over a finite number of steps.

Each step of the solver requires one evaluation of the network, so the cost of generating a sample is measured in the number of function evaluations (NFE).
Accurate integration can need hundreds to thousands of steps, which makes sampling slow and expensive (Zhang and Chen, 2023).
This cost limits their use wherever latency or compute is constrained.
Reducing the NFE is therefore a central goal of current research.

Three families of methods pursue this goal.
The first designs better solvers that reach good samples in about 10 to 20 NFE (Zhang and Chen, 2023; Lu et al., 2022).
The second chooses where to place the steps along the noise range (Watson et al., 2021; Sabour et al., 2024).
The third trains a model to take fewer steps, for example by distillation or by consistency training (Salimans and Ho, 2022; Song et al., 2023).
All three make few-step sampling practical.
In this regime the numerical error of the solver can no longer be ignored, because each step approximates the trajectory over an interval that grows as the number of steps shrinks (Karras et al., 2022).

Few-step samplers are usually compared with metrics that summarize the whole distribution, such as the Fréchet Inception Distance (FID; Heusel et al., 2017).
A single number cannot distinguish different failure cases (Sajjadi et al., 2018).
It also says little about parts of the distribution that carry little probability.
Convergence guarantees have the same property.
A bound on the total variation distance between the generated and the target distribution limits the error in the probability of every event by that distance (Chen et al., 2022; Li et al., 2024).
When the distance is comparable to the probability of a rare event, the bound allows that probability to double or to vanish.
A sampler can therefore look accurate overall while changing how often a rare outcome occurs.

Rare outcomes are often the reason a dataset is valuable.
Rare conditions matter most in medical diagnosis, under-represented groups matter in fairness studies, and unusual cases improve a classifier when they are added to its training data (Um et al., 2024; Um and Ye, 2024).
In these uses, the frequency of an outcome in the generated data matters as much as the quality of each sample.
A sampler that produces a rare outcome too often can overstate its prevalence and distort the class balance of synthetic training sets.
A sampler that produces it too rarely loses coverage of exactly the cases that were hardest to collect.
Either error can go unnoticed when only aggregate quality is reported.

The effect of fast sampling on a rare outcome is hard to study on real data for three reasons.
First, the true frequency of a rare outcome is usually unknown, so the generated frequency has nothing to be compared with.
Second, generated samples carry no labels, so the frequency has to be estimated through a proxy such as a classifier or a region of the data space.
Third, the score is only approximated by a network, so the error in the samples mixes the error of the model with the error of the solver.
Analyses in simple settings list several separate sources of error, namely generalization and optimization in score matching, the discretization of the diffusion, and the minimal noise level (Hurault et al., 2025).
On a trained model these sources cannot be told apart from the samples alone.

We therefore study the following problem.
Let $p$ be a data distribution and let $A$ be a rare event, a set of outcomes with a small probability $p(A)$.
Let a sampler with a budget of $N$ NFE generate samples from a distribution $q_N$.
The question is whether $q_N(A)$ is close to $p(A)$.
If it is not, what determines the difference $q_N(A) - p(A)$?
The project brief names three candidates, which are how rare the event is, how close its outcomes are to those of other classes, and where along the noise range the sampler places its steps.
A further question is for which budgets fast sampling is safe for the rare event, in the sense that the difference is smaller than the uncertainty of estimating $q_N(A)$ from a finite number of samples.
The last question is whether the answers change when the exact score is replaced by a score learned from data.

## 2 Related Work

#### Diffusion models and the probability flow ODE.
Diffusion models learn to reverse a process that adds Gaussian noise to data (Sohl-Dickstein et al., 2015; Ho et al., 2020).
Score-based models generate samples with Langevin dynamics driven by scores estimated at several noise levels (Song and Ermon, 2019).
Song et al. (2021b) unify these views with stochastic differential equations, and show that every reverse-time diffusion has a deterministic counterpart, the probability flow ODE, whose only randomness is the starting noise.
Karras et al. (2022) separate the design into independent choices, namely the noise schedule, the time discretization and the solver, and study each on its own.
Their analysis treats the sampler as a numerical integrator of an ODE.
The explicit Euler method is first order, with a local error that grows with the square of the step size, and higher-order solvers reduce this error at the cost of extra NFE per step.

#### Reducing the number of steps.
Training-free samplers use the structure of the ODE to reduce the NFE.
DDIM (Song et al., 2021a) gives a deterministic sampler that works with fewer steps.
DEIS (Zhang and Chen, 2023) applies an exponential integrator to the semilinear structure of the ODE, and DPM-Solver (Lu et al., 2022) solves it with an exact formulation of the solution and high-order methods.
Both reach good samples in about 10 to 20 NFE.
Karras et al. (2022) combine a second-order method with a discretization that places smaller steps at low noise.
The placement of the steps is itself a design choice.
Watson et al. (2021) choose the steps by dynamic programming on the evidence lower bound, and Sabour et al. (2024) optimize the schedule for a given solver, model and dataset.
Adaptive solvers choose the step size during sampling (Jolicoeur-Martineau et al., 2021; Kemperman and Ambrogioni, 2026).
Lee et al. (2023) relate the truncation error of the solver to the curvature of the generative trajectories.
A second line trains a network to take fewer steps, by distillation (Salimans and Ho, 2022), by consistency models (Song et al., 2023), or by evolving the output distribution during training so that a single step suffices (Deng et al., 2026).
These works typically report FID, likelihood or perceptual quality as a function of the NFE.

#### Error analysis.
Convergence theory bounds the distance between the generated and the target distribution.
Chen et al. (2022) show that, with an $L^2$-accurate score, diffusion models sample from essentially any distribution with a number of steps that is polynomial in the problem parameters.
Li et al. (2024) show that about $d/\varepsilon$ steps suffice for the probability flow ODE sampler to reach total variation distance $\varepsilon$ in dimension $d$.
Hurault et al. (2025) analyze the Wasserstein error for Gaussian data and keep the errors of score matching and of discretization separate.
Hurault et al. (2026) derive expansions of the Euler-Maruyama error for small step sizes, with explicit formulas for Gaussian data and an extension to Gaussian scale mixtures.

#### Gaussian mixtures as a testbed.
Low-dimensional Gaussian mixtures are a common setting for studying diffusion models, because their score is available in closed form.
Biroli et al. (2024) use high-dimensional Gaussian mixtures with an optimally trained score, and identify a speciation transition, where the gross structure of the data appears, followed by a collapse onto memorized samples.
Raya and Ambrogioni (2023) show that the generative dynamics have two phases separated by a symmetry breaking, and that starting the sampler later with a Gaussian initialization improves fast samplers.
Aithal et al. (2024) use one- and two-dimensional Gaussians to study samples that lie outside the support of the training data.
Zhang et al. (2025) use a mixture of Gaussians and two-dimensional toy distributions to study samples that are over-concentrated under deterministic samplers.

#### Rare and minority samples.
Sehwag et al. (2022) observe that uniform sampling from a diffusion model mostly draws from high-density regions, and they guide the sampler toward low-density regions.
Um et al. (2024) attribute a bias toward majority features to the denoiser of Tweedie's formula at high noise.
They define a minority score from its reconstruction error and use it to guide sampling.
Later work removes the external classifier (Um and Ye, 2024) and the guidance itself (Um et al., 2025).
Few-step generators are also reported to lose diversity.
Gandikota and Bau (2025) find that distilled models commit to the image structure at the first step, and that using the base model for that step restores diversity.
Lin et al. (2026) report that few-step flow matching models concentrate on dominant modes and neglect rare ones.
Zhang et al. (2025) report that deterministic ODE samplers produce more concentrated samples than stochastic ones, and attribute this to the dynamics together with a misfit of the score at high noise.

#### Evaluation of generative models.
FID (Heusel et al., 2017) is the standard metric for comparing samplers.
Sajjadi et al. (2018) define precision and recall for distributions, which separate the quality of the samples from the coverage of the target.
Kynkäänniemi et al. (2019) refine these with nonparametric estimates of the data manifold, and Naeem et al. (2020) propose density and coverage metrics that address their reliability problems.

## References

Aithal, S. K., Maini, P., Lipton, Z. C., and Kolter, J. Z. (2024). Understanding hallucinations in diffusion models through mode interpolation. NeurIPS 2024. arXiv:2406.09358.

Biroli, G., Bonnaire, T., de Bortoli, V., and Mézard, M. (2024). Dynamical regimes of diffusion models. Nature Communications 15, 9957. arXiv:2402.18491.

Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., and Zhang, A. R. (2022). Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. arXiv:2209.11215.

Deng, M., Li, H., Li, T., Du, Y., and He, K. (2026). Generative modeling via drifting. arXiv:2602.04770.

Gandikota, R. and Bau, D. (2025). Distilling diversity and control in diffusion models. arXiv:2503.10637.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. NeurIPS 2017. arXiv:1706.08500.

Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS 2020. arXiv:2006.11239.

Hurault, S., Terris, M., Moreau, T., and Peyré, G. (2025). From score matching to diffusion: a fine-grained error analysis in the Gaussian setting. arXiv:2503.11615.

Hurault, S., Moreau, T., and Peyré, G. (2026). Geometry-aware discretization error of diffusion models. arXiv:2605.08392.

Jolicoeur-Martineau, A., Li, K., Piché-Taillefer, R., Kachman, T., and Mitliagkas, I. (2021). Gotta go fast when generating data with score-based models. arXiv:2105.14080.

Karras, T., Aittala, M., Aila, T., and Laine, S. (2022). Elucidating the design space of diffusion-based generative models. NeurIPS 2022. arXiv:2206.00364.

Kemperman, E. and Ambrogioni, L. (2026). Adaptive second-order solvers for fast stochastic diffusion sampling. arXiv:2610.03034.

Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., and Aila, T. (2019). Improved precision and recall metric for assessing generative models. NeurIPS 2019. arXiv:1904.06991.

Lee, S., Kim, B., and Ye, J. C. (2023). Minimizing trajectory curvature of ODE-based generative models. ICML 2023. arXiv:2301.12003.

Li, G., Wei, Y., Chi, Y., and Chen, Y. (2024). A sharp convergence theory for the probability flow ODEs of diffusion models. arXiv:2408.02320.

Lin, Y., Shi, J., Ye, S., Wang, W., Yao, Y., and Liu, T. (2026). SubFlow: sub-mode conditioned flow matching for diverse one-step generation. arXiv:2604.12273.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. (2022). DPM-Solver: a fast ODE solver for diffusion probabilistic model sampling in around 10 steps. NeurIPS 2022. arXiv:2206.00927.

Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., and Yoo, J. (2020). Reliable fidelity and diversity metrics for generative models. ICML 2020. arXiv:2002.09797.

Raya, G. and Ambrogioni, L. (2023). Spontaneous symmetry breaking in generative diffusion models. NeurIPS 2023. arXiv:2305.19693.

Sabour, A., Fidler, S., and Kreis, K. (2024). Align your steps: optimizing sampling schedules in diffusion models. arXiv:2404.14507.

Sajjadi, M. S. M., Bachem, O., Lucic, M., Bousquet, O., and Gelly, S. (2018). Assessing generative models via precision and recall. NeurIPS 2018. arXiv:1806.00035.

Salimans, T. and Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. ICLR 2022. arXiv:2202.00512.

Sehwag, V., Hazirbas, C., Gordo, A., Ozgenel, F., and Canton Ferrer, C. (2022). Generating high fidelity data from low-density regions using diffusion models. CVPR 2022. arXiv:2203.17260.

Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. ICML 2015. arXiv:1503.03585.

Song, J., Meng, C., and Ermon, S. (2021a). Denoising diffusion implicit models. ICLR 2021. arXiv:2010.02502.

Song, Y. and Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. NeurIPS 2019. arXiv:1907.05600.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021b). Score-based generative modeling through stochastic differential equations. ICLR 2021. arXiv:2011.13456.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. (2023). Consistency models. ICML 2023. arXiv:2303.01469.

Um, S., Lee, S., and Ye, J. C. (2024). Don't play favorites: minority guidance for diffusion models. ICLR 2024. arXiv:2301.12334.

Um, S. and Ye, J. C. (2024). Self-guided generation of minority samples using diffusion models. ECCV 2024. arXiv:2407.11555.

Um, S., Kim, B., and Ye, J. C. (2025). Boost-and-Skip: a simple guidance-free diffusion for minority generation. arXiv:2502.06516.

Watson, D., Ho, J., Norouzi, M., and Chan, W. (2021). Learning to efficiently sample from diffusion probabilistic models. arXiv:2106.03802.

Zhang, Q. and Chen, Y. (2023). Fast sampling of diffusion models with exponential integrator. ICLR 2023. arXiv:2204.13902.

Zhang, Y., Liao, Z., Wu, J., and Zou, D. (2025). On the collapse errors induced by the deterministic sampler for diffusion models. arXiv:2508.16154.
