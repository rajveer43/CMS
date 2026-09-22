# Consultation prompt — improving point-cloud SR results

**How to use this file.** Paste everything between the `>>> BEGIN` and
`>>> END` markers into ChatGPT (or another strong model) as a single message.
It asks for a diagnosis and a ranked set of interventions, and then asks it to
emit an implementation prompt you can bring straight back to Claude Code.

Keep the two steps separate on purpose: the consult is where disagreement is
useful, and the implementation prompt is where precision is. Do not let the
consulting model write code — this repo has conventions and hard constraints it
cannot see.

---

>>> BEGIN PROMPT

# Role

You are advising on a computer-vision / set-prediction research problem in
high-energy physics. I need your best technical judgement on **why a model is
failing and what to change**, not encouragement. Treat this as a design review
where I have already spent four weeks and produced a negative result. If my
reasoning below is wrong, say so directly.

# The task

Super-resolve CMS calorimeter jet images: given a low-resolution 64×64×3 image,
produce the 128×128×3 high-resolution version. The 3 channels are separate
calorimeter subsystems.

There are two pipelines in the project:

1. **A dense-grid CNN GAN** — works. This is the baseline to beat.
2. **A point-cloud set-to-set transformer** — the subject of this consult. It
   treats a jet as an unordered list of hits `(channel, x, y, energy)` and never
   materializes a dense image internally.

## Why a point cloud was attempted

The data is **98.5% empty pixels** (measured: mean HR occupancy 1.51%, 744 hits
per jet on a 49,152-cell grid). On that data a plain pixel-wise L1 loss is
minimized by predicting a dim blob everywhere, because empty pixels dominate
the average. The grid pipeline works around this with energy-weighted L1
(`w = 1 + 5.0 * energy`). The hypothesis was that a loss over a *list of actual
hits* has no empty-pixel dilution to exploit, removing the degenerate optimum
by construction rather than by tuning.

# How success is measured (this matters — read carefully)

A **jet tagger** (small CNN binary classifier, signal vs background) is trained
once on real HR images and frozen. It is then scored (AUC) on three inputs:

- HR ground truth — the ceiling
- LR bicubic-upsampled — the floor, i.e. "no model at all"
- SR model output — what we produce

Then:
- `tagging_efficiency = AUC_SR / AUC_HR`
- `recovery_fraction = (AUC_SR − AUC_LR) / (AUC_HR − AUC_LR)`

**Recovery fraction is the number that matters.** +100% means SR is as useful as
ground truth; 0% means the model added nothing over doing nothing; negative
means the model actively destroyed information.

Both pipelines are scored by the *same frozen tagger code*, deliberately, so the
numbers are comparable.

# Results

| model | tagging efficiency | recovery fraction |
|---|---|---|
| **Grid CNN GAN (10 epochs)** | **89.7%** | **+44.8%** |
| Point cloud, position-only Chamfer (10 ep) | 69.7% | −30.9% |
| Point cloud, + energy-value loss term (10 ep) | 70.9% | −25.6% |
| Point cloud, + Stage A/B/C fixes (5 ep) | 74.6% | **−0.8%** |

The point cloud is at "no better than doing nothing" after four rounds of work.

# Architecture (current state)

```
LR image (3,64,64)
  → dense_to_hits (threshold > 0)
  → LR hit list (B, 600, 4)         # padded + boolean mask
  → SetGenerator
  → HR hit list (B, 1300, 4) + per-slot existence logit
  → keep slots with sigmoid(existence) > 0.5
  → hits_to_dense
  → SR image (3,128,128)            # only here does it become an image
```

**SetGenerator:**
- **Encoder** (`PermInvariantEncoder`): Deep-Sets style. Per-point MLP over
  `(channel, x_normalized, y_normalized, log1p(energy))`, then masked
  mean-pool concatenated with masked max-pool → `(B, 2*d_model)`.
- **Queries**: 1300 output slots. Each is seeded from an actual LR hit's `(x,y)`
  position (round-robin tiled since HR has ~2.2× more hits than LR at this
  scale), plus a learned per-repeat perturbation so repeats of the same seed
  can specialize.
- **Decoder**: 3-layer `nn.TransformerDecoder`, 4 heads, `d_model=128`,
  `dim_feedforward=256`, dropout 0. Queries cross-attend to the **full 600-hit
  LR per-point embedding sequence** plus one pooled-summary token, with padding
  masked.
- **Per-slot head** → `x = sigmoid(·)*(grid−1)`, `y` likewise,
  `energy = softplus(·)`, `existence = raw logit`, `channel = argmax(3-way)`.
- **Discriminator** (`SetDiscriminator`): encodes LR and HR/SR sets with a shared
  encoder, uses only the **pooled** output, classifies the concatenation. LSGAN,
  weight 0.1, 3-epoch warmup then 5-epoch ramp.

# Loss (five terms)

1. **Masked Chamfer** (the workhorse, weight 1.0), bidirectional:
   ```
   L = mean_p min_t D(p,t)  +  Σ_t w_t · min_p D(p,t) / Σ_t w_t
   D(p,t) = ||xy_p − xy_t||²  +  λ_E · |E_p − E_t|     with λ_E = 1.0
   w_t    = 1 + 5.0 · E_t                              (target-side only)
   ```
   Padded targets are excluded from the nearest-neighbour search. `pred` takes
   no mask — every one of the 1300 slots is a real prediction.
2. **Existence** — BCE-with-logits, weight 1.0.
3. **Channel** — cross-entropy masked to real hits, weight 0.1.
4. **Energy conservation** — `|Σ E_pred / Σ E_true − 1|`, weight 1.0.
5. **Adversarial** — LSGAN, weight 0.1.

A Sinkhorn (entropic OT) alternative to Chamfer exists and is wired in but is
off by default (~3× compute, benefit never isolated).

# What has already been tried — do not re-propose these

**Fix 1 — energy-value term (worked as designed).** Position-only Chamfer
produced flat energies (top-5: 10.9, 9.2, 7.4, 6.2, 6.1) vs HR's peaked
(27.6, 16.6, 6.2, 6.0, 4.0). Root cause: nearest-neighbour matching on `(x,y)`
gives zero gradient pressure on whether the matched energy *value* is right.
Added `λ_E·|E_p − E_t|` to the pair distance and energy-weighted the target
side. Peaks returned. Tagging: 69.7% → 70.9%.

**Fix 2 (Stage A) — attention bottleneck.** The decoder cross-attended to a
*single* pooled token, so softmax attention weight was mathematically always
exactly 1.0 and no slot could attend differently from any other. Now attends to
the full 600-hit sequence + summary token.

**Fix 3 (Stage B) — input-independent queries.** All 1300 queries were the same
learned vectors in every sample. Now seeded from real LR hit positions.

**Fix 4 (Stage C) — Hungarian matching.** `existence_loss` and `channel_loss`
compared predicted slot *i* to target row *i*, but a point cloud is unordered,
so row order is arbitrary storage detail — the same jet with shuffled rows gave
a different loss. Now uses `scipy.optimize.linear_sum_assignment` on squared
`(x,y)` distance before those two losses. Chamfer was never affected (it does
its own matching), which is why this hid for so long.

# The key measurement — and where I need your judgement

Peak hit energy, same jet, same seed, across every checkpoint:

| checkpoint | SR peak | SR mean | SR total | val Chamfer |
|---|---|---|---|---|
| position-only Chamfer (10 ep) | 6.46 | 0.31 | 195.0 | — |
| **+ energy-value, Fix 1 (10 ep)** | **43.25** | 0.29 | 244.7 | 21.64 |
| + Stage A (3 ep) | **0.38** | 0.28 | 210.7 | — |
| + Stage B (3 ep) | 0.41 | 0.35 | 251.9 | — |
| + Stage C (5 ep) | 0.46 | 0.30 | 231.6 | 19.90 |
| **HR ground truth** | **27.56** | 0.28 | 206.7 | — |

**Read this carefully:** Fix 1 produced real peaks (43.25, overshooting HR's
27.56). From Stage A onward the peak collapses to ~0.4 — i.e. **every one of
~750 predicted hits carries nearly identical energy**. Mean energy and total
energy stay correct throughout. Only the *distribution* is destroyed.

**The loss configuration is byte-identical across these runs** (`λ_E = 1.0`,
energy weighting on, `alpha = 5.0`, same hit budgets, same seed). The only
differences are architectural (Stage A/B/C) and epoch count.

Meanwhile, conditioning is *excellent*: using a cross-Chamfer matrix (compute
Chamfer from each SR output to every HR target in a batch of 16, check whether
the closest is its own), Stage C scores **16/16 with a 68% margin**, chance
being 1/16. Verified on three disjoint batches. So the model knows exactly which
jet it is looking at — and outputs a flat blob.

The tagger scores SR at **AUC 0.5117** vs **0.5130** for plain LR upsampling.
A difference of 0.0013 — noise at n=2000.

## Known confound, stated honestly

Fix 1 ran **10 epochs**; Stage A and B ran **3**; Stage C ran **5**. Peaks might
simply need more epochs. Two things argue against that being the whole story:
Stage C at 5 epochs reaches a *better* val Chamfer (19.90) than Fix 1 at 10
(21.64) while still flat, and the peak does not grow at all between Stage C
epoch 3 and epoch 5 (0.44 → 0.46). **A 10-epoch Stage A run would settle it and
has not been done** (~9 hours on the available hardware).

## One asymmetry I noticed but have not tested

The encoder consumes `log1p(energy)`, but the generator **outputs raw energy via
softplus**, and **the Chamfer loss operates on raw energy**. So the
`|E_p − E_t|` term is in raw units, where it competes against a squared-pixel
position term, and it is averaged over ~750 hit pairs of which the vast majority
are genuinely dim. I do not know whether this matters.

# Hard constraints on any proposal

- The **grid pipeline must not be touched**. Only
  `multiscale_sr/pointcloud/*`, `train_pointcloud.py`,
  `evaluate_pointcloud_tagging.py`, and the point-cloud tests may change.
- Evaluation **must keep using the same frozen tagger**. A point-cloud-native
  tagger would make results incomparable to the 89.7% baseline. If
  re-rasterization is losing information, the fix is better re-rasterization,
  not a different tagger.
- **No rasterize-then-L1 loss.** That reintroduces the exact empty-pixel
  dilution the point-cloud approach exists to escape.
- Hardware: Apple M-series, MPS backend. **A 10-epoch run costs ~9 hours.**
  Compute is the binding constraint — proposals must be rankable by cost.
- Every behavioural change needs a regression test that provably fails against
  the old behaviour and passes against the new.
- The project's discipline: **never claim an improvement without a measured
  tagging-efficiency number.** Chamfer improving is not evidence.

# What I want from you

## Part 1 — Diagnosis

Explain, mechanistically, **why peak energy collapses when the decoder gains a
full-length attention memory**. My working hypothesis: attending over ~600 LR
hits lets each output slot average over many inputs, and an averaged prediction
is a flat one — where the old single-token memory forced differentiation through
the decoder weights instead. Is that right? If not, what is actually happening?
Consider specifically:

- Attention entropy / over-smoothing in cross-attention over a long memory
- Whether softplus + raw-energy Chamfer creates a bad gradient landscape for
  large values (note: `softplus` saturates to linear, but its gradient near
  typical init is small, and most targets are dim)
- Whether the `|E_p − E_t|` L1-style term has a median-seeking solution that is
  actively *correct* to be flat, given ~750 pairs where most targets are dim
- Whether the bidirectional Chamfer's `mean_p` direction (unweighted, over all
  1300 slots) dominates the energy-weighted target direction
- Whether query seeding from LR positions biases toward reproducing LR's
  smoother energy profile

## Part 2 — Ranked interventions

Give **5–8 concrete interventions**, ranked by *expected tagging-efficiency gain
per GPU-hour*. For each:

- What to change, specifically enough to implement
- The mechanism by which it should help
- Estimated cost (loss-reweighting ≈ free; architecture change = a retrain)
- **What measurement would falsify it** — I want to be able to kill bad ideas in
  under an hour
- Expected effect size on recovery fraction, and your confidence

Consider at least: loss reweighting and rebalancing; energy-space
parameterization (log vs raw, and where); a top-K / peak-specific loss term;
local/hierarchical encoding (PointNet++ style neighbourhoods) vs the current
global Deep-Sets pooling; attention temperature or sparsification; Sinkhorn vs
Chamfer; whether the existence threshold is keeping the wrong *set*; and whether
the whole set-prediction framing should be abandoned for this problem.

**Include an explicit "stop" recommendation if you think the point-cloud
approach cannot reach +44.8% recovery within a reasonable budget.** That is a
legitimate answer and I would rather hear it now than after another month. If
so, say what the salvageable transferable finding is.

## Part 3 — Generate an implementation prompt

After Parts 1 and 2, produce a **single self-contained implementation prompt**,
in a fenced code block, that I will hand to a coding agent (Claude Code) working
directly in this repository. That prompt must:

- State the **top 2–3 interventions only** — the ones worth the compute — with
  enough specificity to implement without further consultation
- Give **exact file and function targets** from the architecture described above
- Specify the **regression test** for each change: what must fail before and
  pass after
- Specify the **measurement protocol**: which diagnostic to run, at which epoch
  count, and what number constitutes success vs failure
- Restate the hard constraints verbatim (untouched grid pipeline, same frozen
  tagger, no rasterize-then-L1, test-per-change, measured numbers only)
- Include an explicit **kill criterion**: the result at which the agent should
  stop and report failure rather than continue tuning
- Instruct the agent to **report negative results plainly**, without softening

Write that implementation prompt for an agent that has full repository access
and can run training, but that should not be trusted to re-derive the
motivation — bake the reasoning in.

>>> END PROMPT
