# Point-Cloud Super-Resolution: Results So Far

**Status:** Stage C (Hungarian matching) measured at 5 epochs. The approach does
not yet beat the grid pipeline, and this document explains what was tried, what
each change achieved, and where the remaining gap actually is.

Read [`POINT_CLOUD_EXPLORATION.md`](POINT_CLOUD_EXPLORATION.md) first for why a
point-cloud pipeline exists at all, and
[`POINTCLOUD_AUDIT.md`](POINTCLOUD_AUDIT.md) for the data measurements that set
`lr_n_max`/`hr_n_max`.

## The short version

We built a set-to-set model that takes a list of LR calorimeter hits and
predicts a list of HR hits, never touching a dense image internally. Four
problems were found and fixed. Each fix was correct and verified.
**The model got substantially better at the point-cloud objective it is trained
on, and essentially no better at the task we actually care about.**

Visualizing the output explains why: the model now places hits in the right
places (16/16 on a conditioning test) but gives **every hit nearly the same
energy** — right total, no peaks. Real jets concentrate energy in a few bright
cores, and that concentration is exactly what a jet tagger reads. See
"The core problem" below and the figures in `experiments_pointcloud/<run>/figures/`.

## How we measure success

Two different numbers, and the difference between them turns out to be the
whole story.

**Chamfer distance** is what the model trains on. It compares two lists of hits
by matching each predicted hit to its nearest true hit and measuring how far
apart they are. Lower is better.

**Tagging efficiency** is what actually matters. A jet tagger is a classifier
that looks at a calorimeter image and decides "signal or background". We train
one on real HR images, freeze it, then show it three things:

- the real HR images (the ceiling — the best any model could do)
- the LR images, simply upscaled (the floor — what you get with no model)
- our SR output

`tagging efficiency = AUC_SR / AUC_HR` — how much of the tagger's skill survives
when it looks at our output instead of the truth.

`recovery fraction = (AUC_SR − AUC_LR) / (AUC_HR − AUC_LR)` — of the distance
between "no model" and "perfect", how much did we close. **100% means our output
is as good as the truth. 0% means our model added nothing over doing nothing.
Negative means our model made things worse than doing nothing.**

Recovery fraction is the honest number, because tagging efficiency looks
respectable (70%) even when the model is actively destroying information.

## Where things stand

| model | tagging efficiency | recovery fraction |
| --- | --- | --- |
| **Grid GAN (the target to beat, 10 epochs)** | **89.7%** | **+44.8%** |
| Point cloud, original Chamfer loss (10 ep) | 69.7% | −30.9% |
| Point cloud, + energy-value loss (10 ep) | 70.9% | −25.6% |
| Point cloud, + Stage A/B/C (5 ep) | 74.6% | **−0.8%** |

Three rounds of architectural fixes moved recovery from −30.9% to −0.8%. That
is real progress, but read it precisely: we went from *actively harmful* to
*no better than doing nothing*. The grid pipeline is at +44.8%.

Reproduce the last row with:

```bash
.venv/bin/python -u evaluate_pointcloud_tagging.py \
  --checkpoint experiments_pointcloud/2026-09-21_datasets_20pct_64x_stageC_hungarian_5ep/checkpoints/best.pt \
  --data-dir <path>/datasets_20pct --max-samples 4000 --eval-seed 0
```

## The fixes, and what each one bought

### Fix 1 — the model was smearing energy instead of making peaks

**Problem.** The original loss only compared hit *positions*. Nothing checked
whether a hit's *energy value* was right. So the model learned to spread energy
thinly across many dim hits rather than produce a few bright ones — real HR jets
peak at 27.6, our output peaked at 10.9. This is the same "predict a dim blob"
failure the grid pipeline hit, in a new disguise.

**Fix.** Two additions to `masked_chamfer_loss` and `masked_sinkhorn_loss`
(`multiscale_sr/pointcloud/losses.py`): weight the matching by hit energy
(so getting a bright hit wrong costs more than getting a dim one wrong), and add
an explicit penalty on the energy *difference* at every matched pair.

**Result.** Worked exactly as designed — peaks came back (29.4 vs HR's 27.6).
Tagging efficiency moved 69.7% → 70.9%. Essentially nothing.

### Fix 2 (Stage A) — the model could barely see its own input

**Problem.** The model squeezed the entire LR hit list (up to 600 hits) into one
128-number summary vector, then every output slot looked at that single vector.
When attention has only one thing to look at, its weight is always exactly 1.0 —
a mathematical certainty. So the model could not focus on different input regions
for different outputs. All per-sample information had to fit through 128 numbers.

**Fix.** The decoder now attends to the full per-hit LR sequence, plus one
summary token for cheap global context (`SetGenerator.forward`,
`multiscale_sr/pointcloud/models.py`). Padded slots are masked out.

### Fix 3 (Stage B) — output slots had no idea where they were

**Problem.** Every output slot started from the same fixed learned vector, in
every sample. The model had to work out "which slot handles which region"
entirely from attention, with no starting hint.

**Fix.** `_build_queries` seeds each output slot from an actual LR hit position,
tiling hits round-robin to fill all 1300 slots, with a distinct learned nudge per
repeat so slots seeded from the same hit can specialize.

### Fix 4 (Stage C) — two loss terms were matching hits by row number

**Problem.** A point cloud is an unordered set. But `existence_loss` and
`channel_loss` compared predicted slot #5 against target row #5 — and target row
order is an arbitrary storage detail. Feeding the same physical jet with its rows
shuffled produced a different loss. That is not a tuning issue; it means two of
the loss terms were supervising against noise.

Chamfer was unaffected, because it does its own nearest-neighbour matching. That
is exactly why the problem stayed hidden: one loss term was doing real matching
while another was not.

**Fix.** `multiscale_sr/pointcloud/matching.py` uses the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) to pair each predicted slot with its
geometrically closest target hit before computing those two losses. Slots with no
match are supervised toward "does not exist". Targets are canonically ordered
before solving so the solver's tie-breaking cannot depend on storage order
either.

Enabled by default; `--target-assignment index` restores the old behaviour for
A/B comparison.

**Cost.** 2.99 s/batch versus ~2.0 s/batch without — about 50% slower per step.

## The core problem: good point clouds ≠ good images for tagging

Stage C improved the point-cloud objective clearly, and moved the tagger not at
all:

| checkpoint | own-HR Chamfer | identifies own HR | SR AUC |
| --- | --- | --- | --- |
| Stage A+B (3 ep) | 18.01 | 14/16 | — |
| Stage C (3 ep) | 17.73 | 14/16 | — |
| **Stage C (5 ep)** | **12.08** | **16/16** | **0.5117** |

The "identifies own HR" column comes from
[`scripts/cross_chamfer_diagnostic.py`](../scripts/cross_chamfer_diagnostic.py):
for a batch of 16 jets, compute the Chamfer distance from each SR output to
*every* HR target, and check whether the closest one is its own. Random guessing
scores 1/16. Stage C at 5 epochs scores **16/16 with a 68% margin** — the model
is near-perfect at producing output that matches its specific input.

Meanwhile the tagger scores our SR output at **AUC 0.5117**, against **0.5130**
for simply upscaling the LR image. A difference of 0.0013, which is noise.

**So: Chamfer distance improved ~40% while tagger AUC did not move.** The model
is genuinely reconstructing the right jet *positions*. The tagger cannot tell.

### The figures show why: peak smearing came back at Stage A

Measuring the brightest predicted hit across every checkpoint on the same jet
(same seed, same batch) locates exactly where it happened:

| checkpoint | SR peak hit | SR mean hit | hits kept | SR total | val Chamfer |
| --- | --- | --- | --- | --- | --- |
| Original Chamfer (10 ep) | 6.46 | 0.31 | 631 | 195.0 | — |
| **+ energy value, Fix 1 (10 ep)** | **43.25** | 0.29 | 850 | 244.7 | 21.64 |
| + Stage A (3 ep) | **0.38** | 0.28 | 750 | 210.7 | — |
| + Stage B (3 ep) | 0.41 | 0.35 | 715 | 251.9 | — |
| + Stage C (3 ep) | 0.44 | 0.32 | 699 | 227.1 | 25.41 |
| + Stage C (5 ep) | 0.46 | 0.30 | 761 | 231.6 | 19.90 |
| **HR truth** | **27.56** | 0.28 | 744 | 206.7 | — |

**Fix 1 worked — and Stage A undid it.** The energy-value term produced a real
peak (43.25, overshooting HR's 27.56 but unmistakably peaked). Every
Stage-A-and-later checkpoint has a peak of ~0.4, i.e. no peak at all: mean hit
energy and total energy are right, and the distribution is flat.

Loss configuration is **identical** across the Fix 1 and Stage A runs
(`lambda_energy_value: 1.0`, `chamfer_energy_weighted: true`, `chamfer_alpha:
5.0`, same `lr_n_max`). The change is architectural.

**Caveat on this comparison:** Fix 1 ran 10 epochs, Stage A/B and Stage C (3 ep)
ran 3. Peaks could in principle need more epochs to form. Two things argue
against that being the whole story — Stage C at 5 epochs reaches a *better*
val Chamfer (19.90) than Fix 1 at 10 (21.64) while still having no peak, and the
peak does not grow at all across Stage C epochs 3→5 (0.44 → 0.46). A 10-epoch
Stage A/B/C run would settle it definitively and has not been done.

### What the figures show

Visualizing the output ([`scripts/pointcloud_figures.py`](../scripts/pointcloud_figures.py))
gives a concrete mechanism, and it is not a subtle one.

Measured directly on the Stage C 5-epoch checkpoint:

| jet | SR top-5 hit energies | HR top-5 hit energies | SR total | HR total |
| --- | --- | --- | --- | --- |
| 0 | 0.46, 0.45, 0.45, 0.45, 0.44 | 27.56, 16.63, 6.19, 5.95, 3.97 | 231.6 | 206.7 |
| 1 | 0.46, 0.46, 0.45, 0.45, 0.45 | 9.94, 9.88, 7.69, 6.08, 5.86 | 211.4 | 225.8 |

**The model gets total energy almost exactly right and then spreads it
perfectly flat** — every one of ~750 predicted hits carries ~0.45 energy, where
HR concentrates energy in a handful of bright cores. `jets_scatter.png` shows
this immediately: SR is a uniform dim haze, while HR (and even the LR *input*)
has visible bright structure.

This is the **same failure mode Fix 1 was built to eliminate**, reintroduced by
Stage A and still present at Stage C. It also explains the tagger result mechanically rather than
by hypothesis: a jet tagger classifies on shower shape and energy concentration,
and a flat blob has none. The residual panel in `jets_raster.png` confirms the
shape of the error — SR is missing energy precisely at the jet core (blue) and
adding a faint halo everywhere else (orange).

Why Chamfer does not punish this: the energy-value penalty is one term averaged
over ~750 hit pairs, and the overwhelming majority of HR hits really are dim.
Getting 750 dim hits approximately right while missing 5 bright ones is a good
average, and a catastrophic jet.

This supersedes the earlier hypothesis that the gap was "fine substructure the
tagger keys on that Chamfer under-constrains". That may still contribute, but it
is not needed to explain the result: the peaks are simply gone.

### A note on a diagnostic that was wrong

An earlier check rasterized SR and HR to images and compared them by cosine
similarity, and reported near-total conditioning failure. That diagnostic was
invalid. On data that is 98.5% empty, two point sets differing by a 2-pixel shift
per hit share no exact pixel overlap and score **exactly 0.0** cosine similarity,
despite being structurally near-identical. It was guaranteed to read near-zero
regardless of model quality.

The cross-Chamfer check above replaced it, and reverses the conclusion.
Conclusions drawn from the cosine diagnostic should not be trusted.

## What to try next

Ordered by cost, cheapest first.

1. **Work out why Stage A's architecture kills the peak, and fix that.** This is
   now the clearest defect and it has a precise location: peaks existed before
   Stage A and not after, with an unchanged loss. A plausible mechanism worth
   checking first is that attending over the full ~600-hit LR sequence lets each
   output slot average over many input hits, and an averaged prediction is a
   flat one — where the old single-summary memory forced slots to differentiate
   through the decoder instead. Cheap diagnostics: inspect the decoder's
   attention entropy per slot, and run Stage A for 10 epochs to remove the
   epoch-count confound noted above.

2. **In parallel, strengthen the energy-value signal.** Raise
   `--lambda-energy-value` above its current 1.0 (it competes against a position
   term averaged over ~750 pairs), weight it by target energy the way the
   position term already is, or add an explicit top-K peak term so the brightest
   hits cannot be traded against the dim majority. Loss-weighting only, so cheap
   to try.

3. **Check the rasterization boundary.**
   Take the *ground-truth* HR hit list, convert it to an image with
   `hits_to_dense`, and score the tagger on that. If a lossless round-trip of
   real data already drops AUC toward 0.51, then the bottleneck is the
   point-cloud representation itself — or the existence threshold that decides
   which hits to keep — and no amount of generator improvement can fix it.
   About 20 minutes of compute, and it splits the remaining hypotheses cleanly.
   Two known suspects live here: real HR carries a small negative energy
   pedestal (~0.91% of total magnitude) that a strictly-positive point cloud
   cannot represent, and the existence threshold may keep the wrong *set* of
   hits even when each individual hit is accurate.

4. **Revisit the encoder if the above come back clean.** `PermInvariantEncoder` is
   Deep-Sets style: per-point MLP plus a global pool. Both that and Chamfer's
   aggregate matching under-represent *local* structure, which is exactly what a
   CNN tagger reads. A hierarchical encoder (PointNet++ style neighbourhoods)
   was deliberately scoped out early as unjustified complexity; the evidence now
   justifies revisiting it.

5. **More training alone is probably not the answer**, though a 10-epoch
   Stage A/B/C run is needed to close the confound above. Epochs 3→5 bought a large
   Chamfer improvement and zero tagger movement. Extrapolating, a 10-epoch run
   should land near 75–78% efficiency with recovery still near zero. Worth
   confirming eventually; not where the gap is.

## Figures

Generated by [`scripts/pointcloud_figures.py`](../scripts/pointcloud_figures.py)
into each run's `figures/` directory. **All six runs have figures**, so the
stage-by-stage progression above can be seen directly:

| file | what it shows |
| --- | --- |
| `jets_scatter.png` | LR / SR / HR as actual point clouds. SR is a uniform dim haze; HR has bright cores. |
| `jets_raster.png` | The same jets as dense images — the tagger's actual view — plus the SR−HR residual. |
| `energy_profile.png` | Sorted per-hit energies, log scale. HR falls from ~28 to ~1; SR is flat at ~0.45. |
| `peak_smearing.png` | Total energy (correct) next to peak energy (collapsed). The clearest single slide. |
| `chamfer_vs_tagging.png` | Efficiency and recovery across all stages vs the grid GAN target. Cross-run summary, written only with `--summary`, so it lives in the Stage C run only. |

Compare `peak_smearing.png` in `..._pointcloud_64x_20pct_fixed/` (peaks present)
against the same file in `..._pc3_fullattn_3ep/` (peaks gone) to see the
regression described above.

Energy is plotted on a log scale throughout. On a linear scale every panel is
almost entirely black, because a few hits carry most of the energy — which hides
the very difference these figures exist to show.

```bash
.venv/bin/python -u scripts/pointcloud_figures.py \
  --checkpoint <run>/checkpoints/best.pt \
  --data-dir <path>/datasets_20pct --out-dir <run>/figures
```

## Honest limitations

- **The Stage C result is 5 epochs, not 10.** The grid baseline's best
  checkpoint was epoch 7 of 10, so this is not a like-for-like comparison. The
  direction is clear; the exact number would move.
- **Stage C's contribution has not been isolated.** A matched run with
  `--target-assignment index` would separate "Stage C helped" from "Stage A/B
  helped". It was skipped for compute budget (~3 h), and because it isolates a
  contribution to a metric now shown not to track the target.
- **Item 1 under "What to try next" is a proposal, not a result.** It has not
  been run.
- **HR AUC varies slightly between evaluations** (0.6858 / 0.6692 / 0.6888)
  because the tagger is retrained per run. Ratios absorb some of this, but small
  differences between table rows are not meaningful.
