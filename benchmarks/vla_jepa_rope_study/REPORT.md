# Correct RoPE does not imply better recursive prediction

AI-assisted component study, 10 October 2026. This report accompanies
[LeRobot PR #4835](https://github.com/huggingface/lerobot/pull/4835).
The correction predates this study; the contribution here is reproducible
validation and a frozen-checkpoint diagnosis of mixed learning results.

## Original paired CUDA experiment

The predictor is the actual `ActionConditionedVideoPredictor` from LeRobot
commit `b9cb121cb7d4e3e26ec5c906d08dda68d151ba52`, reduced to two layers,
width 96, four heads and 153,808 parameters. Old and corrected implementations
are imported into separate namespaces; only the two frequency expansion lines
differ. This does not test the complete current PR, real video representations,
pretrained VLA weights, language conditioning or a physical robot.

There are 120 generated trajectories with 16 transitions, four spatial patches
and 16 fixed features per patch. Three trajectory folds contain 64 train,
16 validation and 40 held-out test episodes each. Seeds 11, 22 and 33 produce
nine old/new pairs. Initial weights, batch order and trajectory splits are
identical within each pair. The validation minimum selects a checkpoint;
test data do not select checkpoints. Each of the 18 formal runs has 1,000
optimizer steps. Smoke and pilot runs are preserved but excluded from means.
All selected checkpoints occur at the final step, so convergence is not shown.
The primary training/evaluation context contains eight frames, starting at
the beginning of each trajectory rather than a random temporal crop.

| Original CUDA metric: feature MSE, lower is better | Old | Corrected | Relative change |
|---|---:|---:|---:|
| 8-frame teacher-forced one-step | 0.0043982141 | 0.0041329792 | -6.03% |
| First 7 recursive predictions | 0.0455416205 | 0.0448665963 | -1.48% |
| All 15 recursive predictions | 0.1901469578 | 0.2003823585 | +5.38% |
| 16-frame teacher-forced one-step | 0.0174597090 | 0.0172765193 | -1.05% |

All nine fold/seed pairs improve the primary metric. Seven of nine pairs
worsen the 15-step mean, and all three cross-fold seed aggregates worsen it.
The corrected per-horizon mean is better at horizons 1–6 and already worse
at horizon 7, whose input still contains only eight frames. It would therefore
be incorrect to attribute the entire degradation to contexts longer than eight.
The 8- and 16-frame teacher-forced metrics also include the first transition,
where a single observed position cannot identify the hidden initial velocity;
the recursive evaluation starts from two true frames and predicts times 2–16.

The selected CUDA mathematics checks pass 8/8 for corrected and 2/8 for old:
closed-form rotation, norm, same-position inner product, joint position shift,
dimension-two broadcasting, reference input gradients, finite attention
forward/backward and causal-prefix preservation. These are selected numerical
checks, not exhaustive backend/mixed-precision coverage or finite-difference
gradcheck. Formal training took 535.48 seconds on an RTX 3070 Laptop 8 GB,
Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8. Peak PyTorch
allocated/reserved memory was 92.90/102 MiB; the device-wide sampled 1,422 MiB
includes the desktop and other programs.

## What the rotation correction guarantees

For adjacent coordinates, a correct pair uses one angle:

$$R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix},
\qquad R(\theta)^T R(\theta)=I.$$

The old frequency repetition can give the two coordinates different angles:

$$M=\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\beta&\cos\beta\end{pmatrix},
\qquad MM^T=\begin{pmatrix}1&\sin(\beta-\alpha)\\\sin(\beta-\alpha)&1\end{pmatrix}.$$

Thus it generally is not orthogonal. `repeat_interleave(2, dim=-1)` restores
the adjacent pairing. Same-position inner products are preserved; at different
positions, the rotated inner product depends on relative position, rather than
equaling the original unrotated inner product. These are the geometric
properties of [RoPE](https://arxiv.org/abs/2104.09864).

Orthogonality constrains the rotation suboperation, not the entire trained
predictor. Attention weights, projections, feed-forward networks and the
training solution can all differ. A smaller error on true histories does not
order the error on histories containing earlier predictions.

## Frozen-checkpoint diagnostic results

All 18 original checkpoints were evaluated on Mac CPU without retraining.
The same audited generator/configuration was used, but the full regenerated
tensor hash differs across platforms. Every original CUDA recursive horizon
was reproduced within a maximum absolute MSE difference of $2.37\times10^{-7}$,
below the diagnostic rejection threshold $10^{-5}$. This is
numerical reproduction, not bit-identical original data. No original result
was replaced. Diagnostic script/data/archive hashes and per-episode metrics
are included in the diagnostic archive.

The full-context 15-step feature MSE has this exact decomposition (means over
three folds and three initialization seeds):

| Term | Old | Corrected | Corrected minus old |
|---|---:|---:|---:|
| True-history residual squared, $b^2$ | 0.01831509 | 0.01812903 | -0.00018606 |
| Feedback shift squared, $d^2$ | 0.15572543 | 0.16550559 | +0.00978016 |
| Cross term, $2bd$ | 0.01610647 | 0.01674775 | +0.00064128 |
| Sum: recursive feature MSE | 0.19014698 | 0.20038236 | +0.01023538 |

On true histories the corrected model's residual is slightly smaller; after
earlier predictions replace true history, the feedback-shift term is larger
and dominates the difference. The exact identity is checked per episode and
horizon in float64 on the computed predictions. This locates the measured
difference in feedback behavior, without uniquely identifying which learned
weights or feature channels generate it.

Decoded physical position error has a complementary exact decomposition:

| Kernel-weighted defect term | Old | Corrected | Corrected minus old |
|---|---:|---:|---:|
| Diagonal: weighted individual defect squares | 0.09775340 | 0.09100579 | -0.00674760 |
| Off-diagonal: cross-time defect products | 0.00007954 | 0.01317196 | +0.01309242 |
| Sum: mean 15-step position MSE | 0.09783294 | 0.10417775 | +0.00634482 |

The unweighted one-step dynamics-defect MSE is also smaller for corrected:
0.00077126 → 0.00071822 (-6.88%). Nevertheless, positive cross-time products
increase enough to outweigh the smaller diagonal energy. Thus **smaller
individual defects do not ensure smaller accumulated error**: in these frozen
models and trajectories, temporal reinforcement is larger. This is an exact
quadratic accounting of the decoded position error; it does not by itself
explain every redundant feature channel or prove a universal RoPE mechanism.
The kernel reconstruction's maximum absolute residual is approximately
$1.24\times10^{-14}$. The known-dynamics oracle's maximum absolute position
error over 15 predictions is $5.02\times10^{-7}$ against the float32 generator.

Controlled context/re-encoding comparisons are mixed:

| Recursive evaluation, all 15 predictions | Old feature MSE | Corrected feature MSE | Change |
|---|---:|---:|---:|
| Growing context, raw proposals fed back | 0.19014698 | 0.20038236 | +5.38% |
| Last 8 frames, raw proposals fed back | 0.17974132 | 0.18512627 | +3.00% |
| Last 2 frames, raw proposals fed back | 0.20507674 | 0.19694352 | -3.97% |
| Growing context, re-encoded feedback, proposal metric | 0.18419390 | 0.18143503 | -1.50% |
| Growing context, re-encoded feedback, feedback metric | 0.18420129 | 0.18036188 | -2.08% |

The eight-frame cap does not eliminate the old/new degradation. Two-frame
context flips the relative ranking but has worse absolute feature error than
eight-frame context. Re-encoding also flips the relative feature ranking, yet
its **absolute position MSE is worse for both variants**: old 0.09783 → 0.14356,
corrected 0.10418 → 0.13664. It therefore is not an established accuracy fix.
These context and feature-consistency interventions change multiple factors,
as described below. Original negative findings remain part of the contribution.

## Exact teacher-forced / feedback decomposition

Let $z_t$ be a true feature frame, $\hat z_t$ a recursively generated frame,
and $F_t$ the frozen model's last output with the same action sequence and
context policy. The first two input frames are true observations. Define

$$b_{t+1}=F_t(z_{0:t},a_{0:t})-z_{t+1},$$
$$d_{t+1}=F_t(\hat z_{0:t},a_{0:t})-F_t(z_{0:t},a_{0:t}).$$

Then the recursive error is **exactly**

$$e_{t+1}=\hat z_{t+1}-z_{t+1}=b_{t+1}+d_{t+1},$$
$$\operatorname{MSE}(e)=\operatorname{MSE}(b)+\operatorname{MSE}(d)
 +2\operatorname{mean}(b\odot d).$$

Here $b$ means realized teacher-forced residual, not statistical estimator
bias. The cross term may be negative and cannot be discarded. A local
linearization is $d\approx\sum_{s\leq t}J_{t,s}e_s$; a Transformer reads the
whole history, so a last-frame derivative alone is not the full feedback map.
The computed finite difference $d$ requires no linearization assumption.
This resembles the distribution shift studied in sequential prediction by
[Ross, Gordon and Bagnell](https://proceedings.mlr.press/v15/ross11a.html),
but their imitation-learning theorem is not a guarantee for this predictor.

## Exact position dynamics and accumulation kernel

The synthetic generator uses $v_{t+1}=\rho v_t+a_t$ and
$p_{t+1}=p_t+\eta v_{t+1}$, with $\rho=0.85$ and $\eta=0.15$.
Eliminating velocity gives

$$p_{t+1}=(1+\rho)p_t-\rho p_{t-1}+\eta a_t.$$

Two true positions therefore permit a physics oracle; the first-transition
hidden-velocity uncertainty no longer applies to the 15-step rollout. The
two-frame companion matrix has characteristic roots $1$ and $0.85$. The true
process is marginal in position and damped in velocity, not an exponentially
unstable system. Its derivative with respect to the last position is 1.85,
which is an explicit counterexample to interpreting a one-frame gain above
one as proof of instability.

Decode any model frame by averaging its first two coordinates over patches.
For an arbitrary predicted position sequence, define its dynamics defect

$$r_{t+1}=\hat p_{t+1}-(1+\rho)\hat p_t+\rho\hat p_{t-1}-\eta a_t.$$

With two exact initial positions, the position error obeys

$$e^p_{t+1}=(1+\rho)e^p_t-\rho e^p_{t-1}+r_{t+1}.$$

Ignoring only floating-point generator roundoff, its exact convolution is

$$e^p_n=\sum_{j=2}^n h_{n-j}r_j,\qquad
h_k=\sum_{m=0}^k\rho^m=\frac{1-\rho^{k+1}}{1-\rho}.$$

A single defect has bounded asymptotic position influence $1/(1-\rho)=6.67$.
A sustained aligned defect is repeatedly added and can produce linear error
growth; its squared error can grow quadratically. Defect size alone is not
sufficient: temporal alignment and correlations enter

$$\|e^p_n\|^2=\sum_{i,j=2}^n h_{n-i}h_{n-j}\langle r_i,r_j\rangle.$$

The diagnostic implementation subtracts the true sequence's small numerical
dynamics defect before reconstruction, so the identity is exact on the stored
floating-point tensors. This formula is a decomposition of measured errors,
not a claim that the learned model internally implements Newtonian dynamics.

## Feature consistency intervention and limits

The known pointwise observation map is $z=\phi(p)$: it contains position,
position squared, trigonometric terms, patch-local position and a radial
feature. Arbitrary 4×16 predictions need not satisfy these relationships.
The intervention decodes a shared position by averaging patch coordinates
and feeds $\phi(\hat p)$ back into the model. It uses predicted positions,
never the true future position. This is a deterministic re-encoding onto the
known feature set, **not** a nearest-point projection or a method available
for arbitrary real-video representations. It also averages patch noise and
changes feature channels simultaneously; it cannot isolate a single causal
feature. Proposal error is measured before this intervention, and feedback
error after it, so their metrics must not be confused.

Growing context, clipped eight-frame context and clipped two-frame context
are compared on all frozen checkpoints. Clipping changes history and resets
local position indices at once. Later trajectory states are also evaluated
beyond the prefix used in training; the finite training-position envelope is
a descriptive support check, not a probabilistic out-of-distribution test.
The diagnostics were chosen after the original mixed result was observed.
They are exploratory, use the same held-out trajectories, and do not supply
independent confirmation of a newly tuned method.

## Compatibility and contribution boundary

Old-trained and corrected-trained checkpoints are evaluated with their own
implementations. Swapping the correction into a released old-trained model
is not evaluated. Upstream V-JEPA 2 explicitly retained its old behavior for
[released-model compatibility](https://github.com/facebookresearch/vjepa2/pull/75).
The correction and original candidate authors should be credited separately
from this component study. AI assistance was used for code, analysis and
independent automated checks; independent human review is not claimed.
