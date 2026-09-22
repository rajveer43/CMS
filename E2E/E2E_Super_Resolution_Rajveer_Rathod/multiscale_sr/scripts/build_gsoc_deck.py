"""Build a self-contained HTML deck for the GSoC point-cloud SR update.

Figures are base64-embedded, so the output is ONE file that works offline and
survives being emailed or moved. Print-to-PDF gives one slide per page (see
the @page / .slide print rules in the template).

Usage:
    python scripts/build_gsoc_deck.py --out gsoc_update.html
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments_pointcloud"

RUN_FIX1 = EXP / "2026-09-20_datasets_20pct_64x_pointcloud_64x_20pct_fixed" / "figures"
RUN_A = EXP / "2026-09-20_datasets_20pct_64x_pc3_fullattn_3ep" / "figures"
RUN_C = EXP / "2026-09-21_datasets_20pct_64x_stageC_hungarian_5ep" / "figures"


def img(path: Path) -> str:
    """Inline a PNG as a data URI so the deck has no external file deps."""
    if not path.exists():
        raise FileNotFoundError(f"figure missing: {path} - regenerate with scripts/pointcloud_figures.py")
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GSoC Update - Point-Cloud Super-Resolution</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e;
    --line:#30363d; --accent:#58a6ff; --good:#3fb950; --bad:#f85149;
    --warn:#d29922; --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;}
  html,body{height:100%;overflow:hidden}
  /* Filmstrip: all slides in one row, translated horizontally. Transform
     (not scroll) so transitions are GPU-composited and cannot be nudged
     out of alignment by a stray trackpad gesture mid-presentation. */
  .deck{
    display:flex;flex-wrap:nowrap;height:100vh;width:100vw;
    transition:transform .38s cubic-bezier(.4,0,.2,1);will-change:transform;
  }
  .slide{
    flex:0 0 100vw;width:100vw;height:100vh;
    background:var(--panel);border:0;border-radius:0;
    padding:clamp(28px,4.2vh,52px) clamp(34px,5vw,86px);margin:0;
    display:flex;flex-direction:column;overflow:hidden;
  }
  /* Scale type to viewport so a slide never overflows on a projector. */
  .slide{font-size:clamp(13px,1.05vw,17px)}
  /* Compact the vertical rhythm: at 16:9 a slide has far less height than
     the old scrolling layout, and overflow is now clipped rather than
     scrollable, so margins are the first thing to give. */
  h1{font-size:clamp(30px,3.6vw,46px);margin-bottom:.22em}
  .sub{margin-bottom:clamp(10px,1.6vh,20px);font-size:clamp(12.5px,1.05vw,16px)}
  h3{margin:clamp(9px,1.5vh,18px) 0 .35em}
  ul{margin:.4em 0 .4em 20px}
  li{margin:clamp(3px,.55vh,7px) 0}
  p{margin:.45em 0}
  .note{padding:clamp(8px,1.1vh,13px) 16px;margin:clamp(8px,1.2vh,15px) 0;
    font-size:clamp(12.5px,1.02vw,15.5px)}
  .eq{padding:clamp(8px,1.2vh,14px) 16px;margin:clamp(6px,.9vh,11px) 0;
    font-size:clamp(11.5px,.95vw,14.5px)}
  .card{padding:clamp(10px,1.4vh,16px)}
  .cols,.cols3{gap:clamp(12px,1.6vw,26px)}
  table{margin:clamp(8px,1.2vh,14px) 0}
  th,td{padding:clamp(5px,.75vh,9px) 12px}
  figure{margin:clamp(6px,.9vh,10px) 0}
  .stat{padding:clamp(10px,1.5vh,18px) 12px}
  h2{font-size:clamp(21px,2.3vw,31px)}
  h3{font-size:clamp(13px,1.15vw,17px)}
  p,li{font-size:clamp(13px,1.08vw,17px)}
  table{font-size:clamp(12px,1vw,15.5px)}
  .big{font-size:clamp(30px,3.4vw,46px)}
  figure img{max-height:44vh;object-fit:contain}
  .slide.title{align-items:center;justify-content:center;text-align:center}
  h1{font-size:40px;margin:0 0 10px;letter-spacing:-.02em;line-height:1.15}
  h2{font-size:27px;margin:0 0 6px;letter-spacing:-.01em;color:var(--ink)}
  h3{font-size:17px;margin:20px 0 8px;color:var(--accent);
     text-transform:uppercase;letter-spacing:.06em}
  .sub{color:var(--muted);font-size:16px;margin:0 0 22px}
  .kicker{color:var(--accent);font-size:13px;font-weight:650;
    text-transform:uppercase;letter-spacing:.11em;margin-bottom:10px}
  p,li{font-size:16.5px;line-height:1.62;color:#c9d1d9}
  ul{margin:8px 0 8px 20px;padding:0} li{margin:7px 0}
  strong{color:#fff;font-weight:650}
  code{font-family:var(--mono);font-size:.9em;background:#0d1117;
    border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:#79c0ff}
  table{width:100%;border-collapse:collapse;margin:14px 0;font-size:15px}
  th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left}
  th{color:var(--muted);font-weight:600;font-size:12.5px;
     text-transform:uppercase;letter-spacing:.06em}
  td.num{font-family:var(--mono);text-align:right}
  tr.hl td{background:rgba(88,166,255,.09)}
  .good{color:var(--good);font-weight:650}
  .bad{color:var(--bad);font-weight:650}
  .warn{color:var(--warn);font-weight:650}
  figure{margin:10px 0;text-align:center}
  figure img{max-width:100%;border-radius:8px;border:1px solid var(--line);background:#fff}
  figcaption{color:var(--muted);font-size:13.5px;margin-top:9px;line-height:1.5}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:26px}
  .cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  .card{background:#0d1117;border:1px solid var(--line);border-radius:9px;padding:16px}
  .card h4{margin:0 0 7px;font-size:15px;color:var(--accent)}
  .card p{font-size:14.5px;margin:0;line-height:1.55}
  .big{font-size:44px;font-weight:700;font-family:var(--mono);line-height:1}
  .stat{text-align:center;padding:18px 12px}
  .stat .lab{color:var(--muted);font-size:12.5px;text-transform:uppercase;
    letter-spacing:.07em;margin-top:8px}
  .note{border-left:3px solid var(--accent);background:rgba(88,166,255,.06);
    padding:13px 17px;margin:15px 0;font-size:15.5px;border-radius:0 7px 7px 0}
  .note.bad{border-left-color:var(--bad);background:rgba(248,81,73,.07)}
  .note.warn{border-left-color:var(--warn);background:rgba(210,153,34,.07)}
  .flow{display:flex;align-items:center;justify-content:center;
    flex-wrap:wrap;gap:9px;margin:18px 0}
  .box{background:#0d1117;border:1px solid var(--line);border-radius:8px;
    padding:12px 15px;text-align:center;min-width:112px}
  .box .t{font-size:14px;font-weight:650;color:var(--ink)}
  .box .d{font-size:11.5px;color:var(--muted);font-family:var(--mono);margin-top:3px}
  .box.hl{border-color:var(--accent);background:rgba(88,166,255,.08)}
  .arw{color:var(--muted);font-size:19px}
  .eq{background:#0d1117;border:1px solid var(--line);border-radius:8px;
    padding:14px 18px;margin:11px 0;font-family:var(--mono);
    font-size:14.5px;color:#79c0ff;overflow-x:auto}
  .eq .gloss{color:var(--muted);font-family:inherit;font-size:13.5px;
    display:block;margin-top:7px;font-family:-apple-system,sans-serif}
  .foot{margin-top:auto;padding-top:16px;color:var(--muted);font-size:12.5px;
    border-top:1px solid var(--line);display:flex;justify-content:space-between}
  /* toolbar */
  #prog{position:fixed;top:0;left:0;height:3px;width:0;background:var(--accent);
    z-index:100;transition:width .38s cubic-bezier(.4,0,.2,1)}
  #hint{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);
    background:rgba(22,27,34,.94);border:1px solid var(--line);border-radius:20px;
    padding:8px 17px;font-size:13px;color:var(--muted);z-index:99;
    transition:opacity .8s;pointer-events:none}
  #hint b{color:var(--accent)}
  #bar{position:fixed;bottom:16px;right:16px;display:flex;gap:8px;z-index:99;
    align-items:center;opacity:.28;transition:opacity .2s}
  #bar:hover{opacity:1}
  #bar button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
    border-radius:7px;padding:9px 15px;font-size:13.5px;cursor:pointer;font-weight:550}
  #bar button:hover{border-color:var(--accent);color:var(--accent)}
  #pos{background:var(--panel);border:1px solid var(--line);border-radius:7px;
    padding:9px 13px;font-size:13px;color:var(--muted);font-family:var(--mono)}
  @media print{
    @page{size:1280px 780px;margin:0}
    html,body{background:#fff;height:auto;overflow:visible}
    #bar,#hint,#prog{display:none!important}
    /* Unwind the horizontal filmstrip: for paper the slides must stack
       vertically again, one per page, with no transform offset. */
    .deck{display:block;height:auto;width:auto;max-width:none;padding:0;
      transform:none!important;transition:none}
    .slide{flex:none;width:auto;font-size:15px}
    figure img{max-height:none}
    .slide{page-break-after:always;break-after:page;margin:0;border:0;
      border-radius:0;min-height:780px;height:780px;padding:44px 52px;
      background:#fff;overflow:hidden}
    .slide:last-child{page-break-after:auto}
    h1,h2,strong{color:#0d1117}
    p,li{color:#24292f}
    .card,.eq,.box{background:#f6f8fa;border-color:#d0d7de}
    .sub,.foot,figcaption,th,.box .d,.stat .lab{color:#57606a}
    figure img{border-color:#d0d7de}
    table td,table th{border-bottom-color:#d0d7de}
    code{background:#f6f8fa;border-color:#d0d7de}
  }
</style>
</head>
<body>
<div class="deck" id="deck">

<!-- 1 -->
<section class="slide title">
  <div class="kicker">GSoC 2026 &middot; ML4Sci &middot; Weekly Update</div>
  <h1>Point-Cloud Super-Resolution<br>for Calorimeter Jet Images</h1>
  <p class="sub" style="font-size:18px;max-width:730px">
    Building a set-to-set model that treats a jet as a list of hits rather than a
    dense image &mdash; and what measuring it honestly revealed.
  </p>
  <div style="margin-top:8px;color:var(--muted);font-size:15px">
    Rajveer Rathod &middot; Week of 15&ndash;22 September 2026
  </div>
  <div class="note" style="margin-top:34px;max-width:790px;text-align:left">
    <strong>Headline:</strong> the pipeline is built, tested and measured end to end.
    It does <em>not</em> yet beat the grid baseline &mdash; and the figures produced this
    week locate exactly which change broke it.
  </div>
</section>

<!-- 2 -->
<section class="slide">
  <div class="kicker">Slide 2</div>
  <h2>Where this fits</h2>
  <p class="sub">Two parallel pipelines, one repository.</p>
  <div class="cols">
    <div class="card">
      <h4>Grid pipeline &mdash; shipped, working</h4>
      <p>A CNN GAN on dense <code>(3,128,128)</code> images. Tagging efficiency
      <span class="good">89.7%</span>, recovery <span class="good">+44.8%</span>.
      <strong>Untouched by this work</strong> and its tests stay green.</p>
    </div>
    <div class="card">
      <h4>Point-cloud pipeline &mdash; this update</h4>
      <p>A from-scratch set-to-set transformer: LR hit list &rarr; HR hit list,
      never touching a dense image internally. A second, parallel research
      effort &mdash; not an incremental change.</p>
    </div>
  </div>
  <h3>Why attempt a point cloud at all</h3>
  <p>Calorimeter data is <strong>98.5% empty pixels</strong> (measured, not assumed).
  On that data a plain pixel-wise L1 loss is minimised by outputting a dim blob,
  because empty pixels dominate the average. The grid pipeline fixed this with an
  engineered workaround (energy-weighted L1, <code>alpha=5.0</code>).</p>
  <div class="note">
    A loss defined over a <strong>list of actual hits</strong> has no empty-pixel
    dilution to exploit, so the degenerate optimum disappears <em>by
    construction</em> rather than by tuning. That was the hypothesis worth testing.
  </div>
  <div class="foot"><span>Motivation</span><span>2</span></div>
</section>

<!-- 3 -->
<section class="slide">
  <div class="kicker">Slide 3</div>
  <h2>What shipped this week</h2>
  <p class="sub">One commit, <code>18bd840</code> &mdash; 67 files, ~4,950 insertions.</p>
  <div class="cols3">
    <div class="card stat"><div class="big" style="color:var(--accent)">2,477</div>
      <div class="lab">lines of pipeline</div></div>
    <div class="card stat"><div class="big" style="color:var(--accent)">1,037</div>
      <div class="lab">lines of tests</div></div>
    <div class="card stat"><div class="big good">108</div>
      <div class="lab">tests passing, 1 skipped</div></div>
  </div>
  <h3>Delivered</h3>
  <ul>
    <li><strong>Stage C &mdash; Hungarian target matching</strong>
        (<code>pointcloud/matching.py</code>), the week's main feature, with 10 new tests.</li>
    <li><strong>Two 5&ndash;10 epoch training runs</strong> plus the tagging evaluation
        that produces the headline number.</li>
    <li><strong>A corrected diagnostic</strong> &mdash; the previous one was
        measuring the wrong thing and reversed a conclusion.</li>
    <li><strong>25 figures across all 6 runs</strong>, which surfaced the key finding.</li>
    <li><strong>Documentation</strong>: <code>docs/POINTCLOUD_RESULTS.md</code>, written
        to be readable without this codebase in front of you.</li>
  </ul>
  <div class="foot"><span>Progress summary</span><span>3</span></div>
</section>

<!-- 4 -->
<section class="slide">
  <div class="kicker">Slide 4</div>
  <h2>Architecture: end to end</h2>
  <p class="sub">Dense images enter and leave; everything in between is a point set.</p>
  <div class="flow">
    <div class="box"><div class="t">LR image</div><div class="d">(3,64,64)</div></div>
    <span class="arw">&rarr;</span>
    <div class="box"><div class="t">dense_to_hits</div><div class="d">threshold &gt; 0</div></div>
    <span class="arw">&rarr;</span>
    <div class="box"><div class="t">LR hit list</div><div class="d">(B,600,4)</div></div>
    <span class="arw">&rarr;</span>
    <div class="box hl"><div class="t">SetGenerator</div><div class="d">transformer</div></div>
    <span class="arw">&rarr;</span>
    <div class="box"><div class="t">HR hit list</div><div class="d">(B,1300,4)</div></div>
    <span class="arw">&rarr;</span>
    <div class="box"><div class="t">hits_to_dense</div><div class="d">existence &gt; 0.5</div></div>
    <span class="arw">&rarr;</span>
    <div class="box"><div class="t">SR image</div><div class="d">(3,128,128)</div></div>
  </div>
  <h3>Hit representation</h3>
  <p>Every hit is 4 numbers: <code>(channel, x, y, energy)</code>. Lists are padded to a
  fixed budget with a boolean mask, so a batch is one tensor. Budgets come from a
  data audit &mdash; HR carries <strong>744 hits/jet on average</strong>, so
  <code>hr_n_max=1300</code> is the 99th percentile, not a guess.</p>
  <div class="note">
    <strong>Fixed output budget + learned existence score</strong>, the DETR pattern:
    the model emits 1300 candidate slots and learns which are real, rather than
    generating a variable-length list. Variable-length generation is substantially
    harder to train; the cost of this choice is the matching problem on slide 8.
  </div>
  <div class="foot"><span>Architecture</span><span>4</span></div>
</section>

<!-- 5 -->
<section class="slide">
  <div class="kicker">Slide 5</div>
  <h2>The generator</h2>
  <p class="sub"><code>SetGenerator</code> &mdash; encoder, queries, cross-attention decoder.</p>
  <div class="cols">
    <div>
      <h3>1 &middot; Encoder</h3>
      <p><code>PermInvariantEncoder</code>, Deep-Sets style: a per-point MLP followed by
      a masked mean+max pool. Permutation-invariant by construction &mdash; reordering
      the input hit list cannot change the output.</p>
      <h3>2 &middot; Queries</h3>
      <p>1300 output slots, each <strong>seeded from a real LR hit position</strong>
      (round-robin tiled, with a learned per-repeat perturbation so repeats of the
      same seed can specialise).</p>
      <h3>3 &middot; Decoder</h3>
      <p>3-layer transformer decoder, 4 heads, <code>d_model=128</code>. Queries
      cross-attend to the <strong>full per-hit LR sequence</strong> plus one global
      summary token.</p>
    </div>
    <div>
      <h3>Per-slot output head</h3>
      <div class="eq">x, y = sigmoid(&middot;) &times; (grid&minus;1)
<span class="gloss">Always a valid HR coordinate, by construction.</span></div>
      <div class="eq">energy = softplus(&middot;)
<span class="gloss">Non-negative &mdash; calorimeter deposits cannot be negative.</span></div>
      <div class="eq">existence = raw logit
<span class="gloss">Thresholded only at final rasterisation, never inside the loss.</span></div>
      <div class="eq">channel = argmax(3-way softmax)
<span class="gloss">Discrete; gradient flows via the cross-entropy term instead.</span></div>
      <div class="note" style="font-size:14.5px">
        <strong>Discriminator</strong> (<code>SetDiscriminator</code>) judges
        &ldquo;is this a plausible HR set given this LR set&rdquo; from the
        <em>pooled</em> encoding only &mdash; a deliberately global judgement,
        since a point cloud has no patches to run a PatchGAN over.
      </div>
    </div>
  </div>
  <div class="foot"><span>Architecture &middot; generator</span><span>5</span></div>
</section>

<!-- 6 -->
<section class="slide">
  <div class="kicker">Slide 6</div>
  <h2>Loss functions</h2>
  <p class="sub">Five terms. The first is the reconstruction workhorse.</p>
  <h3>Masked Chamfer &mdash; the main reconstruction term</h3>
  <div class="eq">L_chamfer = mean<sub>p</sub> min<sub>t</sub> D(p,t) &nbsp;+&nbsp; &Sigma;<sub>t</sub> w<sub>t</sub> &middot; min<sub>p</sub> D(p,t) / &Sigma;<sub>t</sub> w<sub>t</sub>
<span class="gloss">Both directions: every prediction must be near some target, and every target must be covered by some prediction. Padded targets are excluded from the search entirely.</span></div>
  <div class="eq">D(p,t) = ||xy<sub>p</sub> &minus; xy<sub>t</sub>||&sup2; &nbsp;+&nbsp; &lambda;<sub>E</sub> &middot; |E<sub>p</sub> &minus; E<sub>t</sub>| &nbsp;&nbsp;&nbsp; w<sub>t</sub> = 1 + &alpha;&middot;E<sub>t</sub>
<span class="gloss">Position distance PLUS an explicit energy-value gap, and target-side weighting by energy (same alpha=5.0 as the grid pipeline's weighted L1). Both were added after a measured failure &mdash; see slide 7.</span></div>
  <div class="cols" style="margin-top:6px">
    <div class="card"><h4>Existence &mdash; BCE</h4><p>Is this slot a real hit? Teaches
      cardinality instead of imposing it. Weight 1.0.</p></div>
    <div class="card"><h4>Channel &mdash; cross-entropy</h4><p>Which of 3 calorimeter
      channels, masked to real hits. Weight 0.1.</p></div>
    <div class="card"><h4>Energy conservation</h4><p><code>|&Sigma;E_pred / &Sigma;E_true &minus; 1|</code>
      &mdash; identical to the grid pipeline's physics loss. Weight 1.0.</p></div>
    <div class="card"><h4>Adversarial &mdash; LSGAN</h4><p>Weight 0.1, with a 3-epoch
      warm-up then a 5-epoch ramp, so reconstruction settles first.</p></div>
  </div>
  <div class="note" style="font-size:14.5px"><strong>Sinkhorn</strong> (entropic optimal
  transport) is implemented and wired as an alternative to Chamfer &mdash; it penalises
  many-to-one collapses that Chamfer permits &mdash; but is off by default at ~3&times;
  the compute. Its benefit has not been isolated; that remains an open experiment.</div>
  <div class="foot"><span>Loss functions</span><span>6</span></div>
</section>

<!-- 7 -->
<section class="slide">
  <div class="kicker">Slide 7</div>
  <h2>Fix 1: the loss had no opinion about energy</h2>
  <div class="cols">
    <div>
      <h3>Symptom, found by measurement</h3>
      <p>A position-only Chamfer loss produced SR jets whose top-5 energies were
      <strong>flat</strong> (10.9, 9.2, 7.4, 6.2, 6.1) against HR's
      <strong>peaked</strong> profile (27.6, 16.6, 6.2, 6.0, 4.0) &mdash; while matching
      total energy closely.</p>
      <h3>Root cause</h3>
      <p>Nearest-neighbour matching on <code>(x,y)</code> gives <strong>zero gradient
      pressure</strong> on whether the matched energy <em>value</em> is right. So
      &ldquo;spread the same total energy across more, dimmer hits&rdquo; was a valid
      low-loss solution &mdash; the Chamfer analogue of the grid pipeline's
      &ldquo;predict small everywhere&rdquo;.</p>
      <h3>Fix</h3>
      <p>Add the <code>|E_p &minus; E_t|</code> term and energy-weight the target side
      (slide 6). The same bug was found and fixed independently in Sinkhorn, whose
      transport masses are normalised per sample and therefore see only each hit's
      <em>relative share</em>, never its absolute value.</p>
    </div>
    <div>
      <figure>
        <img src="__F_f1_peak__" alt="Peak energy after Fix 1">
        <figcaption>After Fix 1: peaks are present and in fact overshoot
        (SR/HR = 2.56&times;). Total energy tracks HR closely. <strong>The fix
        worked.</strong></figcaption>
      </figure>
      <div class="note warn" style="font-size:14.5px">
        Tagging efficiency moved <strong>69.7% &rarr; 70.9%</strong>. The fix was
        correct and verified &mdash; and the metric that matters barely moved. This
        pattern repeats, and is the story of this update.
      </div>
    </div>
  </div>
  <div class="foot"><span>Fix 1 &middot; energy-value term</span><span>7</span></div>
</section>

<!-- 8 -->
<section class="slide">
  <div class="kicker">Slide 8</div>
  <h2>Fixes 2&ndash;4: conditioning and correspondence</h2>
  <div class="cols3">
    <div class="card">
      <h4>Fix 2 (Stage A) &mdash; attention bottleneck</h4>
      <p>The model pooled the entire ~600-hit LR set into <strong>one 128-number
      vector</strong>, then every output slot attended to that single token.
      Softmax over a length-1 memory has weight <strong>exactly 1.0, always</strong>
      &mdash; so no slot could attend differently from any other.</p>
      <p style="margin-top:8px"><strong>Fix:</strong> cross-attend to the full
      per-hit LR sequence + a summary token, with padding masked.</p>
    </div>
    <div class="card">
      <h4>Fix 3 (Stage B) &mdash; blank queries</h4>
      <p>All 1300 output slots started from the <strong>same fixed vector in every
      sample</strong>, carrying zero information about that jet.</p>
      <p style="margin-top:8px"><strong>Fix:</strong> seed each query from an actual
      LR hit position, with a learned per-repeat perturbation so slots seeded from
      one hit can specialise to different nearby HR hits.</p>
    </div>
    <div class="card">
      <h4>Fix 4 (Stage C) &mdash; this week</h4>
      <p>Existence and channel losses compared predicted slot <em>i</em> to target
      row <em>i</em>. But a point cloud is <strong>unordered</strong> &mdash; row order
      is an arbitrary storage detail. The same jet with shuffled rows gave a
      <strong>different loss</strong>.</p>
      <p style="margin-top:8px"><strong>Fix:</strong> Hungarian matching before those
      two losses.</p>
    </div>
  </div>
  <div class="note">
    <strong>Why Stage C's bug stayed hidden:</strong> Chamfer does its <em>own</em>
    nearest-neighbour matching, so it was never affected. One loss term was doing
    real matching while another was supervising against noise &mdash; and the
    healthy-looking Chamfer curve masked it.
  </div>
  <div class="foot"><span>Fixes 2&ndash;4</span><span>8</span></div>
</section>

<!-- 9 -->
<section class="slide">
  <div class="kicker">Slide 9</div>
  <h2>Stage C in detail &mdash; Hungarian matching</h2>
  <div class="cols">
    <div>
      <h3>The algorithm</h3>
      <ul>
        <li>Per sample, build a cost matrix between all <strong>1300 predicted
            slots</strong> and all real target hits, using squared
            <code>(x,y)</code> distance.</li>
        <li>Solve with <code>scipy.optimize.linear_sum_assignment</code> &mdash;
            an optimal one-to-one assignment, not a greedy nearest-neighbour pass.</li>
        <li>Targets are <strong>canonically ordered</strong> before solving, so the
            solver's tie-breaking cannot depend on storage order either.</li>
        <li>Unmatched predictions &rarr; supervised toward
            <strong>non-existence</strong>. Unmatched targets contribute nothing here
            but are still covered by Chamfer's independent matching.</li>
        <li>Detached and solved on CPU, up to 4 samples in parallel. Gradients still
            come from Chamfer &mdash; matching only assigns targets.</li>
      </ul>
      <h3>Cost</h3>
      <p><strong>2.99 s/batch</strong> vs ~2.0 s/batch without &mdash; about 50% slower
      per step, measured rather than estimated.</p>
    </div>
    <div>
      <h3>How it is tested</h3>
      <p>The regression test mirrors how Fix 1 was pinned &mdash; it must
      <strong>fail against the old behaviour</strong> and pass against the new:</p>
      <ul>
        <li>Shuffle the target rows while keeping the underlying set identical.
            Hungarian matching still recovers the correct correspondence; naive
            index-order matching does not.</li>
        <li>Existence and channel loss values are
            <strong>invariant to target row order</strong> &mdash; false before the
            fix by construction, true after.</li>
        <li>No target is ever double-matched.</li>
        <li>Edge cases: zero real targets, more targets than slots, coincident
            positions across channels.</li>
        <li>Parallel and serial solves are bit-identical.</li>
      </ul>
      <div class="note" style="font-size:14.5px">10 new tests. Suite:
      <strong>108 passed, 1 skipped</strong>, with the grid pipeline's own tests
      green throughout.</div>
    </div>
  </div>
  <div class="foot"><span>Stage C</span><span>9</span></div>
</section>

<!-- 10 -->
<section class="slide">
  <div class="kicker">Slide 10</div>
  <h2>How success is measured</h2>
  <p class="sub">Two numbers, and the gap between them is the whole story.</p>
  <div class="cols">
    <div>
      <h3>Chamfer distance &mdash; the proxy</h3>
      <p>What the model trains on. Lower is better. Easy to improve, and
      <strong>not what anyone cares about</strong>.</p>
      <h3>Tagging efficiency &mdash; the real metric</h3>
      <p>A jet tagger classifies signal vs background from a calorimeter image. We
      train one on <strong>real HR images</strong>, freeze it, then score it on three
      inputs:</p>
      <ul>
        <li><strong>HR</strong> &mdash; the ceiling</li>
        <li><strong>LR upscaled</strong> &mdash; the floor, i.e. doing nothing</li>
        <li><strong>SR</strong> &mdash; our output</li>
      </ul>
      <div class="note" style="font-size:14.5px">The point-cloud pipeline reuses the
      grid pipeline's <strong>exact frozen tagger code</strong>. A parallel tagger
      would make the two approaches incomparable &mdash; a hard constraint, not a
      convenience.</div>
    </div>
    <div>
      <div class="eq">tagging efficiency = AUC<sub>SR</sub> / AUC<sub>HR</sub>
<span class="gloss">How much of the tagger's skill survives when it sees our output instead of the truth.</span></div>
      <div class="eq">recovery = (AUC<sub>SR</sub> &minus; AUC<sub>LR</sub>) / (AUC<sub>HR</sub> &minus; AUC<sub>LR</sub>)
<span class="gloss">Of the distance between "no model" and "perfect", how much did we actually close.</span></div>
      <table>
        <tr><th>Recovery</th><th>Meaning</th></tr>
        <tr><td class="num good">+100%</td><td>SR is as good as ground truth</td></tr>
        <tr><td class="num">0%</td><td>SR adds nothing over doing nothing</td></tr>
        <tr><td class="num bad">negative</td><td>SR is <strong>worse</strong> than doing nothing</td></tr>
      </table>
      <div class="note bad" style="font-size:14.5px">
        <strong>Recovery is the honest number.</strong> Tagging efficiency looks
        respectable at 70% even while the model is actively destroying information.
        Reporting efficiency alone would have hidden this for weeks.
      </div>
    </div>
  </div>
  <div class="foot"><span>Evaluation methodology</span><span>10</span></div>
</section>

<!-- 11 -->
<section class="slide">
  <div class="kicker">Slide 11</div>
  <h2>Results</h2>
  <table>
    <tr><th>Model</th><th style="text-align:right">Tagging efficiency</th>
        <th style="text-align:right">Recovery fraction</th></tr>
    <tr class="hl"><td><strong>Grid GAN &mdash; the target (10 ep)</strong></td>
        <td class="num good">89.7%</td><td class="num good">+44.8%</td></tr>
    <tr><td>Point cloud, original Chamfer (10 ep)</td>
        <td class="num">69.7%</td><td class="num bad">&minus;30.9%</td></tr>
    <tr><td>Point cloud, + energy value / Fix 1 (10 ep)</td>
        <td class="num">70.9%</td><td class="num bad">&minus;25.6%</td></tr>
    <tr class="hl"><td><strong>Point cloud, + Stage A/B/C (5 ep)</strong></td>
        <td class="num warn">74.6%</td><td class="num warn">&minus;0.8%</td></tr>
  </table>
  <div class="note bad">
    <strong>Read this precisely.</strong> Three rounds of correct, verified fixes moved
    recovery from &minus;30.9% to &minus;0.8%. That is progress from
    <em>actively harmful</em> to <em>no better than doing nothing</em>. It is not
    progress toward the grid baseline's +44.8%.
  </div>
  <figure>
    <img src="__F_c_summary__" alt="Tagging efficiency and recovery across stages">
    <figcaption>Left: efficiency against the 89.7% target. Right: recovery, where
    anything below zero means the model is subtracting value.</figcaption>
  </figure>
  <div class="foot"><span>Results</span><span>11</span></div>
</section>

<!-- 12 -->
<section class="slide">
  <div class="kicker">Slide 12</div>
  <h2>A diagnostic that was measuring the wrong thing</h2>
  <div class="cols">
    <div>
      <h3>The broken check</h3>
      <p>Conditioning was previously assessed by rasterising SR and HR to images and
      comparing them with <strong>cosine similarity</strong>. It reported near-total
      conditioning failure.</p>
      <div class="note bad">
        On 98.5%-empty data, two point sets differing by only a <strong>2-pixel shift
        per hit</strong> &mdash; structurally near-identical, Chamfer distance ~0 &mdash;
        score <strong>exactly 0.0</strong> cosine similarity, because sparse pixel
        overlap needs <em>exact</em> coincidence to register anything.
      </div>
      <p>It was guaranteed to read near-zero regardless of model quality. Conclusions
      drawn from it were discarded.</p>
    </div>
    <div>
      <h3>The replacement</h3>
      <p>Measure conditioning in the metric the model actually trains under. For a
      batch of 16 jets, compute Chamfer from each SR output to <em>every</em> HR
      target, and ask: <strong>is the closest one its own?</strong> Chance is 1/16.</p>
      <table>
        <tr><th>Checkpoint</th><th style="text-align:right">Own HR</th><th style="text-align:right">Margin</th></tr>
        <tr><td>Stage A (3 ep)</td><td class="num">14/16</td><td class="num">39&ndash;42%</td></tr>
        <tr><td>Stage A+B (3 ep)</td><td class="num">14/16</td><td class="num">39&ndash;53%</td></tr>
        <tr class="hl"><td><strong>Stage C (5 ep)</strong></td>
            <td class="num good">16/16</td><td class="num good">68%</td></tr>
      </table>
      <div class="note">
        Verified across <strong>three disjoint batches</strong>, not one. Conditioning
        is <strong>excellent</strong> &mdash; the opposite of what the old diagnostic
        said. The architecture fixes are not broken.
      </div>
    </div>
  </div>
  <div class="foot"><span>Corrected diagnosis</span><span>12</span></div>
</section>

<!-- 13 -->
<section class="slide">
  <div class="kicker">Slide 13</div>
  <h2>So why is tagging still flat?</h2>
  <p class="sub">The model identifies its own jet 16/16 &mdash; and the tagger sees nothing.</p>
  <div class="cols3">
    <div class="card stat"><div class="big good">16/16</div>
      <div class="lab">conditioning (chance 1/16)</div></div>
    <div class="card stat"><div class="big good">19.90</div>
      <div class="lab">val Chamfer, best yet</div></div>
    <div class="card stat"><div class="big bad">0.5117</div>
      <div class="lab">SR AUC &mdash; vs 0.5130 for LR</div></div>
  </div>
  <div class="note bad">
    Chamfer improved roughly <strong>40%</strong> while tagger AUC moved by
    <strong>0.0013</strong> &mdash; noise at n=2000. The trained objective and the real
    metric have come fully apart.
  </div>
  <figure>
    <img src="__F_c_scatter__" alt="LR, SR and HR jets as point clouds">
    <figcaption>The same three jets as point clouds (log energy scale). SR sits in
    the right place with roughly the right hit count &mdash; and is a
    <strong>uniform dim haze</strong>. HR has bright, structured cores. Even the LR
    <em>input</em> has more visible peak structure than the model's output.</figcaption>
  </figure>
  <div class="foot"><span>The core problem</span><span>13</span></div>
</section>

<!-- 14 -->
<section class="slide">
  <div class="kicker">Slide 14</div>
  <h2>The finding: peaks died at Stage A</h2>
  <p class="sub">Brightest predicted hit, same jet, same seed, across every checkpoint.</p>
  <table>
    <tr><th>Checkpoint</th><th style="text-align:right">SR peak</th>
        <th style="text-align:right">SR mean</th><th style="text-align:right">SR total</th>
        <th style="text-align:right">val Chamfer</th></tr>
    <tr><td>Original Chamfer (10 ep)</td><td class="num">6.46</td><td class="num">0.31</td>
        <td class="num">195.0</td><td class="num">&mdash;</td></tr>
    <tr class="hl"><td><strong>+ energy value, Fix 1 (10 ep)</strong></td>
        <td class="num good">43.25</td><td class="num">0.29</td>
        <td class="num">244.7</td><td class="num">21.64</td></tr>
    <tr><td>+ Stage A (3 ep)</td><td class="num bad">0.38</td><td class="num">0.28</td>
        <td class="num">210.7</td><td class="num">&mdash;</td></tr>
    <tr><td>+ Stage B (3 ep)</td><td class="num bad">0.41</td><td class="num">0.35</td>
        <td class="num">251.9</td><td class="num">&mdash;</td></tr>
    <tr><td>+ Stage C (5 ep)</td><td class="num bad">0.46</td><td class="num">0.30</td>
        <td class="num">231.6</td><td class="num">19.90</td></tr>
    <tr class="hl"><td><strong>HR truth</strong></td><td class="num">27.56</td>
        <td class="num">0.28</td><td class="num">206.7</td><td class="num">&mdash;</td></tr>
  </table>
  <div class="note bad">
    <strong>Fix 1 worked &mdash; and Stage A undid it.</strong> Peaks were present
    (43.25, overshooting HR's 27.56) and collapse to ~0.4 from Stage A onward.
    <strong>Loss configuration is identical across these runs</strong>
    (<code>lambda_energy_value: 1.0</code>, <code>chamfer_energy_weighted: true</code>,
    <code>alpha: 5.0</code>). The change is architectural.
  </div>
  <p>Mean and total energy stay correct throughout. Only the <em>distribution</em>
  collapses &mdash; and distribution is exactly what a jet tagger reads.</p>
  <div class="foot"><span>Key finding</span><span>14</span></div>
</section>

<!-- 15 -->
<section class="slide">
  <div class="kicker">Slide 15</div>
  <h2>Peak collapse, visually</h2>
  <div class="cols">
    <figure>
      <img src="__F_f1_peak__" alt="Fix 1 peaks present">
      <figcaption><strong>After Fix 1</strong> &mdash; peak SR/HR = 2.56&times;.
      Overshooting, but unmistakably peaked.</figcaption>
    </figure>
    <figure>
      <img src="__F_a_peak__" alt="Stage A peaks collapsed">
      <figcaption><strong>After Stage A</strong> &mdash; peak collapsed to
      ~0.01&times;, while total energy stays correct.</figcaption>
    </figure>
  </div>
  <figure style="margin-top:4px">
    <img src="__F_c_profile__" alt="Energy profile SR vs HR">
    <figcaption>Sorted per-hit energies at Stage C, log scale. HR (red) falls from
    ~28 to ~1 across its brightest 40 hits. SR (blue) is <strong>flat at ~0.45</strong>
    &mdash; every hit carries the same energy.</figcaption>
  </figure>
  <div class="note warn">
    <strong>Confound, stated plainly:</strong> Fix 1 ran 10 epochs; Stage A/B ran 3.
    Peaks might need more epochs. Two things argue otherwise &mdash; Stage C at 5
    epochs reaches a <em>better</em> Chamfer (19.90) than Fix 1 at 10 (21.64) while
    still flat, and the peak does not grow at all from Stage C epoch 3 to 5
    (0.44 &rarr; 0.46). <strong>A 10-epoch Stage A run would settle it and has not
    been done.</strong>
  </div>
  <div class="foot"><span>Key finding &middot; evidence</span><span>15</span></div>
</section>

<!-- 16 -->
<section class="slide">
  <div class="kicker">Slide 16</div>
  <h2>What the tagger actually sees</h2>
  <figure>
    <img src="__F_c_raster__" alt="LR, SR, HR and residual as dense images">
    <figcaption>Re-rasterised to dense images (log energy). <strong>SR</strong> is a
    diffuse cloud in roughly the right place; <strong>HR</strong> has sharp bright
    cores. The residual (right) shows the error's shape: <strong>blue at the
    centre</strong> &mdash; SR is missing energy exactly where the jet core is &mdash;
    with a faint orange halo of spurious energy spread around it.</figcaption>
  </figure>
  <div class="note">
    This is the mechanism behind the flat AUC, stated concretely rather than
    hypothesised: a jet tagger classifies on <strong>shower shape and energy
    concentration</strong>. A blob with the right total energy and no concentration
    gives it nothing to work with. Chamfer does not punish this, because the
    energy-value term is one penalty averaged over ~750 hit pairs and the vast
    majority of real HR hits genuinely are dim &mdash; getting 750 dim hits
    approximately right while missing 5 bright ones is a good average and a
    catastrophic jet.
  </div>
  <div class="foot"><span>Mechanism</span><span>16</span></div>
</section>

<!-- 17 -->
<section class="slide">
  <div class="kicker">Slide 17</div>
  <h2>Next steps</h2>
  <p class="sub">Reordered by this week's finding &mdash; cheapest and most diagnostic first.</p>
  <div class="card" style="margin:8px 0">
    <h4>1 &middot; Why does Stage A's attention flatten the output?</h4>
    <p>Now the clearest defect, with a precise location: peaks existed before Stage A
    and not after, with an unchanged loss. Leading hypothesis &mdash; attending over
    ~600 LR hits lets each output slot <strong>average over many inputs</strong>, and an
    averaged prediction is a flat one, where the old single-token memory forced slots
    to differentiate through the decoder. Cheap checks: per-slot attention entropy,
    and a 10-epoch Stage A run to close the confound.</p>
  </div>
  <div class="card" style="margin:8px 0">
    <h4>2 &middot; Strengthen the energy-value signal (parallel, cheap)</h4>
    <p>Raise <code>lambda_energy_value</code> above 1.0; weight it by target energy as
    the position term already is; or add an explicit top-K peak term so bright hits
    cannot be traded against the dim majority. Loss-weighting only &mdash; no
    architecture change.</p>
  </div>
  <div class="cols" style="margin-top:4px">
    <div class="card"><h4>3 &middot; Check the rasterisation boundary</h4>
      <p>Score the tagger on <em>ground-truth</em> HR hits round-tripped through
      <code>hits_to_dense</code>. If that alone drops AUC, the representation &mdash;
      not the generator &mdash; is the bottleneck. ~20 minutes.</p></div>
    <div class="card"><h4>4 &middot; Revisit the encoder</h4>
      <p>Deep-Sets pooling and Chamfer both under-represent <em>local</em> structure,
      which is what a CNN tagger reads. A hierarchical encoder was scoped out early;
      the evidence now justifies revisiting.</p></div>
  </div>
  <div class="foot"><span>Next steps</span><span>17</span></div>
</section>

<!-- 18 -->
<section class="slide">
  <div class="kicker">Slide 18</div>
  <h2>Summary &amp; open questions</h2>
  <h3>What went well</h3>
  <ul>
    <li>A complete parallel pipeline &mdash; data, model, five losses, training,
        evaluation, diagnostics, figures &mdash; built and tested to
        <strong>108 passing tests</strong>, with the shipped grid pipeline untouched.</li>
    <li>Stage C fixed a <strong>real correctness bug</strong>: two loss terms were
        supervising against an arbitrary storage order.</li>
    <li>A misleading diagnostic was caught and replaced, <strong>reversing</strong> an
        earlier conclusion about conditioning.</li>
    <li>The figures located the regression to a <strong>specific architectural
        change</strong> &mdash; something four weeks of scalar metrics had not surfaced.</li>
  </ul>
  <h3>What did not</h3>
  <ul>
    <li>Recovery is <strong>&minus;0.8%</strong>. The approach does not yet beat doing
        nothing, and the grid baseline remains far ahead at +44.8%.</li>
    <li>Three correct fixes bought ~5 points of efficiency. That rate is itself a signal.</li>
  </ul>
  <div class="note">
    <strong>Questions for mentors:</strong> (1) Is the Stage A attention-averaging
    hypothesis worth a 10-epoch confirmation run, given ~9 h compute? (2) Should effort
    continue here, or fold the energy-value lesson back into the grid pipeline? (3) Is
    a hierarchical (PointNet++ style) encoder in scope for the remaining timeline?
  </div>
  <div class="foot"><span>Summary</span><span>18</span></div>
</section>

<!-- 19 -->
<section class="slide">
  <div class="kicker">Appendix</div>
  <h2>Reproducing everything</h2>
  <h3>Train (Stage C, matching on by default)</h3>
  <div class="eq">python train_pointcloud.py --data-dir &lt;data&gt; --scale 64 \
  --epochs 10 --batch-size 32 --lr-n-max 600 --hr-n-max 1300 \
  --seed 123 --target-assignment hungarian --run-name stageC_10ep</div>
  <h3>Evaluate &mdash; the number that actually matters</h3>
  <div class="eq">python evaluate_pointcloud_tagging.py \
  --checkpoint &lt;run&gt;/checkpoints/best.pt --data-dir &lt;data&gt; \
  --max-samples 4000 --eval-seed 0</div>
  <h3>Conditioning diagnostic and figures</h3>
  <div class="eq">python scripts/cross_chamfer_diagnostic.py --checkpoint &lt;ckpt&gt; \
  --data-dir &lt;data&gt; --batch-size 16 --batches 0 3 7

python scripts/pointcloud_figures.py --checkpoint &lt;ckpt&gt; \
  --data-dir &lt;data&gt; --out-dir &lt;run&gt;/figures</div>
  <table style="margin-top:14px">
    <tr><th>Setting</th><th>Value</th></tr>
    <tr><td>Data</td><td>20% subset &mdash; 27,860 jets, byte-identical to the full set</td></tr>
    <tr><td>Scale / seed / batch</td><td>64 &rarr; 128 &middot; seed 123 &middot; batch 32</td></tr>
    <tr><td>Hardware</td><td>Apple M-series, MPS backend</td></tr>
    <tr><td>Throughput</td><td>2.99 s/batch with matching, ~2.0 without</td></tr>
    <tr><td>Full 10-epoch run</td><td>~8.9 h measured</td></tr>
  </table>
  <div class="foot"><span>Appendix &middot; reproduction</span><span>19</span></div>
</section>

</div>

<div id="prog"></div>
<div id="hint">&larr; &rarr; to navigate &middot; <b>F</b> fullscreen &middot; click edges to advance</div>
<div id="bar">
  <button onclick="go(window.__cur()-1)" title="Previous">&larr;</button>
  <span id="pos">1 / 19</span>
  <button onclick="go(window.__cur()+1)" title="Next">&rarr;</button>
  <button onclick="toggleFull()" title="Fullscreen (F)">Full</button>
  <button onclick="window.print()">PDF</button>
</div>

<script>
(function(){
  var deck   = document.getElementById('deck');
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var pos    = document.getElementById('pos');
  var prog   = document.getElementById('prog');
  var hint   = document.getElementById('hint');
  var n      = slides.length;
  var cur    = 0;

  function render(){
    deck.style.transform = 'translateX(' + (-cur * 100) + 'vw)';
    pos.textContent = (cur + 1) + ' / ' + n;
    prog.style.width = ((cur) / (n - 1) * 100) + '%';
    // Keep the URL in sync so a given slide can be linked or reloaded into.
    if (history.replaceState) history.replaceState(null, '', '#' + (cur + 1));
  }
  function go(i){
    var t = Math.max(0, Math.min(n - 1, i));
    if (t === cur) return;
    cur = t;
    render();
  }
  window.go = go;
  window.__cur = function(){ return cur; };   // used by the toolbar arrows

  document.addEventListener('keydown', function(e){
    if (e.metaKey || e.ctrlKey || e.altKey) return;   // leave Cmd+P etc. alone
    var k = e.key;
    if (k === 'ArrowRight' || k === 'ArrowDown' || k === ' ' || k === 'PageDown' || k === 'n'){
      e.preventDefault(); go(cur + 1);
    } else if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp' || k === 'p'){
      e.preventDefault(); go(cur - 1);
    } else if (k === 'Home'){ e.preventDefault(); go(0); }
    else if (k === 'End'){ e.preventDefault(); go(n - 1); }
    else if (k === 'f'){ e.preventDefault(); toggleFull(); }
    else if (/^[0-9]$/.test(k)){ e.preventDefault(); go(parseInt(k, 10) - 1); }
  });

  function toggleFull(){
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  }
  window.toggleFull = toggleFull;

  // Touch / trackpad swipe, for presenting from a laptop without a clicker.
  var x0 = null;
  document.addEventListener('touchstart', function(e){ x0 = e.touches[0].clientX; }, {passive:true});
  document.addEventListener('touchend', function(e){
    if (x0 === null) return;
    var dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 55) go(cur + (dx < 0 ? 1 : -1));
    x0 = null;
  }, {passive:true});

  // Clicking the left/right edge advances, like a normal presentation tool.
  document.addEventListener('click', function(e){
    if (e.target.closest('#bar') || e.target.closest('a')) return;
    var third = window.innerWidth / 3;
    if (e.clientX > third * 2) go(cur + 1);
    else if (e.clientX < third) go(cur - 1);
  });

  // Deep link: open on #7 and land on slide 7.
  var h = parseInt((location.hash || '').replace('#',''), 10);
  if (h >= 1 && h <= n) cur = h - 1;

  // Printing needs the transform cleared, then restored afterwards --
  // otherwise the first page carries the current slide's offset.


  // Safety net. CSS clamps handle the common cases, but an unusual window
  // aspect (or a browser with larger default fonts) can still overflow a
  // slide -- and overflow is clipped, which would silently hide content
  // mid-presentation. Measure each slide and scale it down if needed.
  function fit(){
    slides.forEach(function(sl){
      sl.style.transform = '';
      var pad = parseFloat(getComputedStyle(sl).paddingTop) || 0;
      var avail = sl.clientHeight - pad * 2;
      // scrollHeight includes padding; compare like for like.
      var need = sl.scrollHeight - pad * 2;
      if (need > avail + 1){
        var k = Math.max(0.62, avail / need);
        sl.style.transformOrigin = 'top center';
        sl.style.transform = 'scale(' + k.toFixed(4) + ')';
      }
    });
  }
  var rt;
  window.addEventListener('resize', function(){
    clearTimeout(rt); rt = setTimeout(function(){ fit(); render(); }, 120);
  });
  // Print: clear BOTH the filmstrip offset and any per-slide fit scaling,
  // then restore both afterwards -- otherwise the first page carries the
  // current slide's offset, or slides come back unscaled.
  window.addEventListener('beforeprint', function(){
    deck.style.transform = 'none';
    slides.forEach(function(sl){ sl.style.transform = ''; });
  });
  window.addEventListener('afterprint', function(){ fit(); render(); });

  setTimeout(function(){ if (hint) hint.style.opacity = '0'; }, 4200);
  fit();
  render();
})();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "gsoc_update_pointcloud.html"))
    args = ap.parse_args()

    figs = {
        "c_scatter": img(RUN_C / "jets_scatter.png"),
        "c_raster": img(RUN_C / "jets_raster.png"),
        "c_peak": img(RUN_C / "peak_smearing.png"),
        "c_profile": img(RUN_C / "energy_profile.png"),
        "c_summary": img(RUN_C / "chamfer_vs_tagging.png"),
        "f1_peak": img(RUN_FIX1 / "peak_smearing.png"),
        "a_peak": img(RUN_A / "peak_smearing.png"),
    }
    html = TEMPLATE
    for k, v in figs.items():
        html = html.replace("__F_%s__" % k, v)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = len(html.encode()) / 1e6
    print(f"[deck] wrote {out} ({mb:.1f} MB, {html.count('<section')} slides)")


if __name__ == "__main__":
    main()
