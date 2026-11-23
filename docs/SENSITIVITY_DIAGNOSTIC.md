# Can we tell early which "rare" samples are fake?

> [!NOTE]
> Every number here can be regenerated with
> `PYTHONPATH=src python -m experiments.sensitivity_experiment`. It takes about 30
> seconds on a laptop CPU and gives the same answer every time. Output goes to
> `results/sensitivity/`.

> [!WARNING]
> This is an early, exploratory note. It uses the 12 development cases only. The final
> cases have not been touched, and the code refuses to run on them unless you
> explicitly switch on `frozen=True`.

## The short version

We generate 2D points with a fast sampler. A small fraction of the true data is "rare".
When we cut the sampler down to 8 steps, it reports far too many rare points. We found
that:

1. Almost all of the sampler's mistakes are the same kind: it calls a point rare when a
   slower, more careful run says it is not. It almost never misses a truly rare point.
2. A cheap test, "nudge the point a little and see whether the answer flips", can spot
   many of these mistakes early in the run, while the points are still mostly noise.
3. Even so, the test is not worth its cost. Simply re-running every suspicious point
   with more steps is cheaper and works better. Just spacing the 8 to 128 steps
   differently, with more of them near the end, removes most of the errors with no test
   at all.

So the honest result is a partly negative one. The early-warning idea works as a
signal, and it loses to simpler baselines at equal cost.

## Background you need

You can skip this section if you already read `figures/README.md`.

**The data.** Each dataset is 2D points drawn from a mixture of three Gaussian blobs
(round clumps). Two blobs are common. One blob is rare, and only 1% to 8% of points
come from it. The blobs are centered at

$$
\mu_1 = (-s, 0), \qquad \mu_2 = (+s, 0), \qquad \mu_3 = (0, s),
$$

where $s$ is the separation. Small $s$ means the blobs overlap. Each blob has spread
$\sigma = 0.35$. The rare one is $\mu_3$.

**The "rare label".** A generated point has no tag saying which blob made it. So we
label a point *rare* when it lies closer to $\mu_3$ than to $\mu_1$ or $\mu_2$. We write
this label as $\ell(x)$, and $\ell(x) = 1$ means "rare".

**The sampler.** To make a point, we start from pure noise and walk it toward the data.
The walk is controlled by a number $v$, the *noise level*, which starts at $v = 16$
(lots of noise) and ends at $v = 0$ (clean data). At each step the sampler asks
"which way is the data from here?" and moves a bit that way. The answer to that question
is a function called the *score*, written $s(x, v)$. In this project the score is a
known formula, so there is no neural network and no training.

**Steps.** The walk from $v = 16$ to $v = 0$ is cut into a fixed number of straight
moves. The *fast sampler* uses 8 moves. The *fine sampler* uses 1,024. The fast one is
cheap and inexact. The fine one is slow and close to exact. The cost of a run is the
number of times we compute the score, called *score evaluations*. Our fast sampler costs
8 evaluations per starting point.

**Two kinds of mistake.** For one starting point, run both samplers and compare the rare
label each gives:

| Fast says | Fine says | We call it |
|---|---|---|
| rare | not rare | **spurious rare** (a false alarm) |
| not rare | rare | **missed rare** |
| same as fine | same as fine | agreement |

We use the fine sampler as the stand-in for the truth. It is not the true distribution.
That is a limitation, and it is listed at the end.

## The idea: sensitivity

Imagine a ball rolling down a hill with a ridge in the middle. A ball that starts far
from the ridge ends up on the same side no matter how you jiggle it. A ball that starts
right on the ridge can end up on either side after a tiny jiggle. We want to find the
second kind of ball.

For a sample sitting at position $x$ when the noise level is $v_0$, we do this:

1. Finish the walk from $x$ down to $v = 0$ and record the rare label.
2. Make $K$ slightly shifted copies of $x$ by adding a small random nudge $\delta_k$.
3. Finish the walk for each copy and record its label.
4. The *sensitivity score* is the fraction of copies whose label differs from the
   original.

As a formula, with $\Phi$ standing for the final rare label of a finished sample:

$$
\Phi_{v_0}(x) = \ell(\text{finish}(x)), \qquad
S(x, v_0) = \frac{1}{K} \sum_{k=1}^{K} \big|\, \Phi_{v_0}(x) - \Phi_{v_0}(x + \delta_k) \,\big|
$$

The function $\text{finish}(\cdot)$ means "walk from $v_0$ down to $0$". The symbol
$\Phi$ is the downstream evaluation metric, as in the critical period paper. There it is
classification accuracy, and the sensitivity is $\Phi$ of the clean run minus $\Phi$ of
the damaged run. Here $\Phi$ is 0 or 1, so we take the absolute difference, which counts
a flip in either direction. It equals 1 exactly when the label changes, so this is the
same count as before and none of the numbers below change. The nudge is Gaussian noise
whose size is 20% of the noise already present at that level:

$$
\delta_k \sim \mathcal{N}\big(0,\ 0.2^2\,(\sigma^2 + v_0)\, I\big).
$$

A score of 0 means no nudge ever changed the answer. A score of 1 means every nudge did.
High scores should mark points sitting on a ridge, and those are the points a fast,
sloppy sampler is most likely to get wrong.

We try this at several early noise levels $v_0 = 14, 12, 10, 8, 6, 4, 2$, using the
position the fast sampler has reached at that point. The idea is early warning: can we
predict the final mistake before the run is over?

## Where the idea comes from

Lee, Dias, Chiang and Lunga, *Understanding the Learning Phases in Self-Supervised
Learning via Critical Periods* (ICLR 2026). That paper is about training neural
networks, not sampling, and it never mentions diffusion or rare events. I read its main
text and not its appendix. We borrowed two of its measurement ideas and moved the time
axis from "training epoch" to "noise level":

| In the paper | In this note |
|---|---|
| Downstream evaluation metric $\Phi$, such as classification accuracy | $\Phi$, the rare label of the finished sample (0 or 1) |
| Sensitivity $S(t_0) = \Phi(f_\theta) - \Phi(f_{\theta'})$, clean minus damaged | $S(x, v_0)$, the average of $\lvert \Phi(x) - \Phi(x + \delta_k) \rvert$ over $K$ nudges |
| Damage the training data at epoch $t_0$ and see how much it hurts later | Nudge the sample at noise level $v_0$ and see whether the final label changes |
| Damage the data for a short window of epochs | Replace a window of the walk with one big step (Result 4) |
| Fisher information: how sensitive the model is to small changes in its parameters | The squared length of the score arrow, $\lVert s(x, v)\rVert^2$, which measures how sensitive the density is to a small shift in $x$ |

Our exact score has no parameters to take the paper's Fisher information over. That
comparison has to wait until we train the small network.

## How we measured

- **Data.** 12 development cases: 4 geometries (rare weight 2% or 8%, separation 0.8 or
  2.0) times 3 random seeds. Each has 5,000 starting points, so 60,000 trajectories in
  total.
- **Fast sampler.** 8 equal steps. **Fine sampler.** The same starting points through
  1,024 equal steps.
- **Cheap sensitivity score.** $K = 4$ nudged copies, each finished with 16 steps. That
  is $5 \times 16 = 80$ score evaluations per point scored. **Expensive version.**
  $K = 8$ copies, 128-step finishes.
- **Other scores to compare against:**
  - *Entropy* of the cluster shares. For a point $x$, let $p_k$ be how much of $x$
    "belongs" to blob $k$ (the shares sum to 1). Entropy $-\sum_k p_k \log p_k$ is high
    when the point is torn between blobs and low when it clearly belongs to one.
  - *Score norm.* The squared length of the score arrow at $x$.
  - *Radius.* How far $x$ is from the origin.
- **How we grade a score: AUROC.** Pick one spurious point and one genuine point at
  random. AUROC is the chance that the score ranks the spurious one higher. 0.5 means the
  score is no better than a coin flip. 1.0 is perfect. Below 0.5 means the score points
  the wrong way.
- **Error bars: bootstrap.** To see how much an AUROC would wobble with different data,
  we resample the points with replacement 100 times and look at the middle 95% of the
  results. We call this the *interval*. If the whole interval sits above 0.5, we trust
  that the score beats chance for that case.

## Result 1: the sampler invents rare points, it does not lose them

Totals over all 60,000 trajectories:

| | Count | Share of all points |
|---|---|---|
| Fine sampler says rare | 4,425 | 7.4% |
| Fast sampler says rare | 9,834 | 16.4% |
| Spurious rare (fast yes, fine no) | 5,410 | 9.0% |
| Missed rare (fast no, fine yes) | 1 | 0.002% |

So 55% of everything the fast sampler calls rare is a false alarm, and it lost just one
true rare point out of 60,000. By geometry the false-alarm share is 65% (2% rare,
overlapping), 46% (8% rare, overlapping), 79% (2% rare, separated) and 40% (8% rare,
separated).

This changes what we are trying to detect. The job is "which of the sampler's rare
points are fake", and it is not "which rare points will it miss".

## Result 2: overall AUROC is misleading

First attempt: how well does each score predict a spurious label across all points?
Mean AUROC over the 12 cases:

| Noise level $v_0$ | Cheap sensitivity | Expensive sensitivity | Entropy | Score norm | Radius |
|---|---|---|---|---|---|
| 14 | 0.862 | 0.836 | 0.913 | 0.428 | 0.407 |
| 8 | 0.847 | 0.820 | 0.906 | 0.439 | 0.401 |
| 2 | 0.789 | 0.817 | 0.883 | 0.457 | 0.371 |

> [!IMPORTANT]
> These numbers look good and mean little. The fast sampler's own final label, "did I
> end in the rare region?", scores AUROC 0.959 on this task and costs nothing. Spurious
> points can only exist among the points the fast sampler called rare, so anything that
> predicts "ends up rare" also predicts "is spurious". Entropy tops the table because
> of that, and for no other reason.

## Result 3: the real test is inside the "fast-rare" group

Once the fast sampler has called a point rare, the only question left is whether to
trust it. So we grade the scores only on those points, where 55% are fake and 45% are
real. Now a coin flip is a fair baseline. Mean AUROC over 12 cases, with the number of
cases whose interval clears 0.5 in parentheses:

| Noise level $v_0$ | Cheap sensitivity | Expensive sensitivity | Entropy | Score norm | Radius |
|---|---|---|---|---|---|
| 14 | **0.789** (12/12) | 0.537 (8/12) | 0.096 (0/12) | 0.277 (0/12) | 0.282 (0/12) |
| 8 | **0.736** (12/12) | 0.511 (6/12) | 0.086 (0/12) | 0.276 (0/12) | 0.284 (0/12) |
| 6 | 0.707 (9/12) | 0.513 (6/12) | 0.089 (0/12) | 0.275 (0/12) | 0.285 (0/12) |
| 4 | 0.678 (9/12) | 0.518 (6/12) | 0.104 (0/12) | 0.275 (0/12) | 0.287 (0/12) |
| 2 | 0.664 (11/12) | 0.588 (6/12) | 0.169 (0/12) | 0.286 (0/12) | 0.292 (0/12) |

Cheap sensitivity at three noise levels, split by geometry (mean over 3 seeds):

| Geometry | $v_0 = 14$ | $v_0 = 8$ | $v_0 = 2$ |
|---|---|---|---|
| 2% rare, overlapping ($s = 0.8$) | 0.776 | 0.773 | 0.618 |
| 8% rare, overlapping ($s = 0.8$) | 0.832 | 0.824 | 0.723 |
| 2% rare, separated ($s = 2.0$) | 0.747 | 0.599 | 0.567 |
| 8% rare, separated ($s = 2.0$) | 0.800 | 0.746 | 0.747 |

What this shows:

- **Cheap sensitivity works, and works best early.** It beats chance in all 12 cases at
  the earliest checkpoint, and it fades as the run goes on. That fits the early-warning
  idea.
- **The expensive version is nearly useless**, at 0.51 to 0.59, and it costs 14 times
  more. One guess is that a careful 128-step finish rarely changes a label, so it
  carries little signal. We have not tested that guess.
- **Entropy, score norm and radius point the wrong way.** At $v_0 = 14$ and $v_0 = 8$,
  all 12 intervals lie entirely below 0.5. Fake rare points have lower entropy, a
  shorter score arrow and a smaller radius than real ones. We do not know why. If we
  flipped the sign, entropy would score about 0.90 to 0.91 and the other two about 0.72.
  We did not flip any sign, because picking the sign after seeing the results is what
  the README's "freeze your method first" rule is meant to prevent.
- **A possible catch.** The cheap score finishes each nudged copy in only 16 steps,
  which is itself a sloppy sampler. It may partly measure how fragile a coarse sampler
  is on that point, and not how close the point is to a ridge. That would also explain
  why it beats the 128-step version. We have not tested this.

## Result 4: the damage happens late

The early warning works, but when does the damage actually happen? To find out, we took
the 1,024-step walk and replaced one stretch of noise levels with a single big step,
leaving everything else fine. Then we counted how many of the 5,000 labels changed.
Mean over 3 seeds:

| Geometry | $v$: 0 to 0.25 | 0.25 to 0.5 | 0.5 to 1 | 1 to 2 | 2 to 4 | 4 to 8 | 8 to 16 |
|---|---|---|---|---|---|---|---|
| 2% rare, overlapping | **169** | 22 | 23 | 20 | 17 | 15 | 15 |
| 8% rare, overlapping | **132** | 25 | 25 | 21 | 17 | 15 | 13 |
| 2% rare, separated | 9 | 3 | 12 | 20 | 24 | 23 | 22 |
| 8% rare, separated | 5 | 2 | 10 | 21 | 30 | 30 | 29 |

When the blobs overlap, coarsening just the last stretch ($v < 0.25$) changes 5 to 7
times more labels than coarsening any other stretch, even though it is the narrowest.
When the blobs are far apart, the last stretch barely matters and the damage is spread
over the noisy early part.

For overlapping blobs, then, the warning is available early (Result 3) but the mistake
is made late. The paper found the opposite pattern for network training, where early
damage matters most.

The 8-step walk takes its last step from $v = 2$ straight to $v = 0$, which contains
this sensitive stretch. Our checkpoints stop at $v_0 = 2$, so we never looked inside it.

## Result 5: is the detector worth its cost?

This is the test the README asks for: compare methods at the same total number of score
evaluations, and count the detector's own evaluations.

**The gate.** Run the 8-step sampler. At $v_0 = 8$, compute the cheap score for every
trajectory the sampler called rare. If the score is at least a threshold $\tau$, throw
away that trajectory's answer and finish it again from $v_0 = 8$ with 128 steps. Keep
the sampler's answer for everything else. The total cost per starting point is

$$
\text{cost} = 8 + f \times 80 + p_\text{flag} \times 128,
$$

where $f = 0.164$ is the average fraction of points the sampler calls rare (each of
those pays 80 for scoring), and $p_\text{flag}$ is the fraction that get refined.

**The comparisons**, all measured against the 1,024-step answer, as means over 12
cases. "Disagreements" counts labels that differ, out of 5,000 per case. "Rate error" is
the sampler's rare percentage minus the fine sampler's, in percentage points.

| Strategy | Cost (evaluations) | Disagreements | Rate error |
|---|---|---|---|
| Fast sampler, 8 even steps | 8.0 | 450.9 | 9.02 |
| 32 even steps | 32.0 | 205.8 | 4.12 |
| 128 even steps | 128.0 | 62.9 | 1.26 |
| 8 steps, spaced with more near the end | 8.0 | 227.7 | 4.55 |
| 32 steps, spaced with more near the end | 32.0 | 56.8 | 1.14 |
| 128 steps, spaced with more near the end | 128.0 | 7.1 | 0.14 |
| **Refine every fast-rare point, no detector** | **29.0** | **34.5** | **0.69** |
| Gate, $\tau = 0.25$ | 32.0 | 138.4 | 2.77 |
| Refine the same number of points, picked at random | 32.0 | 235.8 | 4.71 |
| Even steps with the same total cost as the gate | 32.2 | 199.8 | 3.99 |
| Gate, $\tau = 0.5$ | 26.7 | 279.6 | 5.59 |
| Gate, $\tau = 0.75$ | 23.2 | 385.6 | 7.71 |

("Spaced with more near the end" is the starter's quadratic schedule, where the
steps bunch up as $v$ approaches 0.) At $v_0 = 8$, $\tau = 0.25$ flags 53% of the
fast-rare points, $\tau = 0.5$ flags 27%, and $\tau = 0.75$ flags 11%.

What this shows:

- **The score is a real selector.** The gate beats refining at random in 11 of 12 cases
  and beats the equal-cost even grid in 10 of 12.
- **But the detector does not pay for itself.** Refining *every* fast-rare point, with
  no detector, costs less (29.0 against 32.0) and leaves far fewer errors (34.5 against
  138.4). The reason is plain arithmetic. Scoring costs about 13 evaluations per
  starting point and saves only about 10 evaluations of refinement. It also beats a
  128-step even run in all 12 cases, at about a quarter of the cost.
- **Changing the schedule beats everything.** With the same step count, moving steps
  toward the end removes 50% of the false alarms at 8 steps, 72% at 32 steps and 89% at
  128 steps. A 32-step run spaced this way (56.8 disagreements) already beats a 128-step
  even run (62.9) at a quarter of the cost. This agrees with Result 4: the end of the
  walk is where the errors come from.
- **When would a detector help?** When refinement is expensive or most points are flagged.
  Here refinement is cheap and only 16% of points need it, so the brute-force baseline
  wins.

## What we can and cannot claim

What the development results support:

- The fast sampler's rare-rate error is almost entirely false alarms.
- A cheap nudge test at high noise ranks fake rare points above real ones, better early
  than late.
- Entropy, score norm and radius point the wrong way on the fast-rare group.
- For overlapping blobs, the damage builds up in the final stretch of the walk.
- Using the score to choose what to refine beats choosing at random.

What they do not support:

- That the score is worth using. Refining everything is cheaper and better, and a
  better step schedule beats both.
- Anything about the final cases, other nudge sizes, other gate checkpoints, or a
  learned score.
- That we found a "critical period" like the paper's. We never computed the paper's
  Fisher information, and the cheap score fades gradually as the run goes on (Result 3)
  with no sharp cutoff where it suddenly stops working.

## Next steps

1. **Try the gate on the better schedule.** On a good schedule the remaining errors are
   rarer and harder to find by brute force, which is where a detector might finally win.
2. **Test the "sloppy finisher" explanation.** Keep $K$ fixed and change the finish
   length: 4, 8, 16, 32, 64 steps.
3. **Look inside the sensitive stretch.** Use a grid with finer steps near the end, so we
   have checkpoints below $v_0 = 2$.
4. **Bring in the trained network.** Compute the paper's Fisher information over
   training epochs and compare it with this sampling-time score.
5. **Freeze the protocol, then run the final cases once** with `frozen=True`.

This fits step 2 of the plan in `LITERATURE_REVIEW.md` (report what the rare-event
error tracks), at the level of single points instead of whole datasets. Result 5 also
gives a measured answer to the step-allocation question in `RESEARCH_DIRECTIONS.md`.

## Limits of this note

- "Spurious" means "differs from the 1,024-step run". That run is close to the truth but
  not the truth.
- The gate was tested at one checkpoint ($v_0 = 8$), one finish length (128 steps) and
  one nudge size (20%). The nudge size was picked once and never varied. Other choices
  could change the gate's standing.
- The detector is charged 80 evaluations for every fast-rare point it scores, including
  points it decides not to refine.
- The 12 cases are 4 geometries with 3 seeds each, so they are not 12 independent
  experiments. AUROC gaps of a couple of hundredths should not be read as real.
