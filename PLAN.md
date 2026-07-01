# Plan: A population-pretrained foundation representation for cuffless blood pressure

*Single source of truth for this project — context, hypotheses, architecture, experiment
matrix, models, metrics, empirical findings to date, known issues, and the operational
playbook for running everything. Supersedes the former EXPERIMENTS.md / MODELS.md /
PERFORMANCE_DIAGNOSIS.md / RUNLOG.md, whose content is folded in below.*

---

## 1. Context and motivation

### 1.1 The problem we're actually solving

We have wearable electrical-impedance-tomography (EIT) "PVI ring" data from ~91 human
subjects (216 sessions: baseline, Valsalva, cold-pressor) with continuous arterial BP
recorded simultaneously. The applied goal is **cuffless blood pressure estimation** —
predicting continuous BP from bioimpedance signals without a cuff.

We already have direct evidence that the straightforward approach — train a supervised
model end-to-end per architecture, pooled across subjects — **does not generalize across
subjects**. In `ml-experiments` (a separate, more mature codebase), the same CRT
architecture trained with a `split_mode="local"` protocol (test *sequences* held out, but
from subjects that were also in the training set) reaches cc_abs **0.928**. The identical
architecture family evaluated under `split_mode="disjoint"` (entire subjects held out) — the
protocol that actually tests generalization to a new person — collapses to **~0.46 peak,
0.21 by the end of training** (`abl-crtsin-bioz-to-waveform`). That's a ~0.5 Pearson-r gap
between "the model has seen this person before" and "the model has not," and it is the
central empirical fact motivating this project: **naively pooling data across subjects and
training one model does not produce a representation that transfers to new people.**

### 1.2 Why "foundation model," specifically

The hypothesis we're testing is not "a bigger/better model will fix this." It's a specific
architectural bet, borrowed from Wang et al. 2025 (*Foundation model of neural activity
predicts response to new stimulus types*, Nature, `s41586-025-08829-y.pdf` — the MICrONS
mouse-visual-cortex foundation model): **separate a shared, population-pretrained *core*
from a small, cheap, per-subject *readout*.** The core is trained once across the whole
cohort; adapting to a new subject means fitting only the readout (or a minimal correction on
top of it), not retraining the whole model on that subject's limited data. If this works, it
directly attacks the generalization failure in §1.1 (the core carries what's shared across
people; the readout carries what's specific to one person) and it directly attacks the
practical deployment problem (a new user of a cuffless BP device needs the calibration
burden — how much of *their own* data you need before the device is accurate — to be as
small as possible).

That second point is the applied hook: **cuffless BP calibration burden is a recognized,
unsolved, clinically important problem** (every cuffless BP method on the market today
requires an individual cuff-calibration step; reducing how much subject-specific data/effort
that takes is a real translational contribution, independent of any neuroscience analogy).

### 1.3 What we are and are not claiming, and why (read this before designing new experiments)

Wang et al.'s framing works because neural population activity is very high-dimensional
(thousands of neurons) and the stimulus domain is unbounded (arbitrary natural video), so "one
shared core generalizes across an open-ended response/stimulus space" is a strong,
information-rich claim. **We do not have that.** BP is a 1–2 dimensional physiological
signal (SBP/DBP, or a stereotyped cardiac waveform dominated by a handful of physiological
parameters — stroke volume, vascular resistance, arterial stiffness, HR). This is a real,
load-bearing difference from the source paper and it must shape what we claim:

- **We are not claiming** that a low-dimensional target makes the problem easy or that
  accuracy should saturate quickly. See the hypothesis in §3.3 for why this specific
  intuition is likely wrong, and what to test instead.
- **We are not claiming** BP prediction is a rich enough readout target to support
  Wang-et-al.-scale "digital twin" / "functional barcode" interpretability claims (Exp D/E)
  as *primary* results. They're demoted to secondary/exploratory (§5).
- **We are claiming**, and building the experiment matrix to test: (a) population
  pretraining reduces the amount of person-specific data needed for accurate calibration
  (Exp A/B/G — the calibration-burden story), and (b) a population-pretrained core
  generalizes across *physiological states* (baseline → Valsalva/pressor) better than a
  model that has only ever seen resting data (Exp C — the OOD story). These two are the
  primary pillars of the paper. Everything else is supporting or exploratory.

---

## 2. Data

- **91 subjects, 216 masked HDF5 sessions** under `$PVI_DATA_ROOT/main/`
  (`{subject}_{session}_masked.h5`): baseline (91), Valsalva (74), pressor (51); ~51
  subjects have all three maneuvers. ~30h of impedance data total.
- **A dedicated, never-yet-used 5-subject holdout set exists** (`branch="holdout"` /
  `TrainingBranch.HOLDOUT`, pre-built by the original pvi-ml data engineering — see
  `src/README.md` Appendix A). It is currently **not wired into any `src/foundation/`
  entry point** — every number reported so far comes from `branch="main"` with an in-run
  `disjoint` re-split. **This must change before any number is paper-bound** (§7.1).
- **Ground truth is present in all 3 maneuvers**, `data/bp/signal` = continuous BP (mmHg).
  Example ranges (subject001): baseline SBP≈109/DBP≈59; Valsalva swings (DBP→35,
  SBP→144); pressor sustained (SBP 130±14). OOD is measurable — the maneuvers really do
  produce different BP regimes, which is what makes Exp C meaningful.
- **Cardiac-cycle-normalized time**: each period = exactly `period_length=50` frames
  (≈40–50 Hz; cycle ≈1.0–1.2s); sessions ≈9 min; `num_periods` ≈440–870/session. Absolute
  timing/HR lives in `stats`, not in the waveform itself — the model sees cardiac *phase*,
  not absolute time (feed HR/timing via stats if timing-sensitive tasks need it).
- **Ring native outputs**: `data/{pviHP,pviLP}/{resistance,reactance}` each shape `(32,T)`;
  `signal` `(1,T)`; `img` `(40,40,T)` = the existing 1-step-Newton EIT reconstruction, fixed
  circular FOV (228/1600 px NaN, constant over time).
- **Masks** (`masks/mask{01,05,10,15}`, 1-based inclusive period indices) = clean QC
  sequences; SQI ≈80%. Leakage-safe train/test partitioning is handled by
  `GraphBipartitePartitioner` (approximate minimum-vertex-cover on a bipartite overlap
  graph — see `src/README.md` Appendix C for the algorithm), which is sound, already
  implemented, and should not need to change.

---

## 3. Approach

### 3.1 Core + readout architecture

- **Core**: a shared backbone, `forward_core(input) -> features`, trained once across the
  pooled cohort. Implemented as `BasePviLearner.forward_core` across all architectures
  (`src/models/base_model.py`); every model in `src/models/` and `src/foundation/` follows
  this split so any core is transferable via `transfer_core()` /
  `load_core_from_state_dict()`.
- **Readout**: `forward_readout(features) -> BP`, one instance per subject
  (`SubjectReadout`, `src/foundation/readout.py`). During pretraining, all subjects share
  one `SHARED_READOUT`. During transfer, the core is frozen and a fresh readout is fit on
  the target subject's (small) data.
- **Two cores, trained differently, compared head-to-head (Exp G)**:
  - **Core S (supervised)**: pretrained end-to-end with BP labels, pooled across the
    cohort (`src/foundation/pretrain.py`).
  - **Core U (self-supervised)**: pretrained with *no BP labels* via a dual pretext —
    masked reconstruction (hide random channel×time patches, reconstruct them) + causal
    forecasting (predict the next window from the past) — `src/foundation/ssl.py`,
    `ssl_pretrain.py`. Motivation: if useful representations can be learned without BP
    labels, pretraining can use *all* available sessions (including ones without reliable
    BP ground truth) and the resulting core may transfer better to OOD physiological
    states, since it wasn't fit to a specific state's BP mapping.

### 3.2 Calibration mechanism — the actual central design question

This is the piece most worth getting right, and where the plan changed most based on
critical review (§8). We will test **three calibration mechanisms**, not just one, along
the same budget curve:

1. **Frozen core + linear readout** (current default, `readout_hidden=0`). Cheapest,
   already implemented, but may under-sell the core (a single linear layer has limited
   capacity to correct for anything the core didn't already linearize).
2. **Frozen core + small MLP readout** (`readout_hidden>0`) or **partial fine-tune**
   (unfreeze the last N core layers). Tests whether more calibration capacity closes the
   gap to a fully fine-tuned model, and at what data budget it stops being worth it.
3. **Frozen core + shape/bias decomposition + per-subject affine correction** (new,
   proposed in §3.3): fit only a 1–2 parameter (scale + offset) correction per subject on
   top of a *fixed, population-shared* shape prediction. This is the cheapest possible
   calibration mechanism and — if it works — the most clinically deployable, since it
   mirrors how real cuffless-BP calibration protocols already work (a couple of cuff
   readings to fit an offset).

We do not assume which of these wins. Comparing them *is* one of the experiments (folded
into Exp B, see §5).

### 3.3 Hypothesis: population pretraining separates cleanly into a "shape" problem and a
"calibration" problem — and only the shape problem benefits from pooling

**Do not assume "BP is low-dimensional, so we should get high accuracy fast."** This
intuition conflates two different things and is already contradicted by one of our own
results (§6.3): Core U at the 64-min budget reaches `cc_abs = 0.307` (a plausible-looking
correlation) alongside `AMAE = 98 mmHg` (a physiologically impossible absolute error). Low
dimensionality made a *correlation-shaped* metric look reasonable quickly, while the
*absolute* number that actually matters clinically (and is what AAMI/BHS standards grade)
was nowhere close.

The reason this isn't a fluke: Wang et al.'s foundation-model argument doesn't come from
"neural activity is high-dimensional, therefore hard, therefore benefits from pretraining."
It comes from "there is a lot of *shared structure* across animals in how visual features
drive responses, so pooling data is statistically efficient." Output dimensionality and
cross-subject *shared structure* are different properties. For BP specifically, absolute
level depends heavily on subject-specific factors that plausibly do **not** transfer
(arterial stiffness, vessel geometry, electrode contact/skin impedance) — which is exactly
why every existing cuffless-BP product requires an individual cuff-calibration step. Whereas
the *waveform shape* — how BP changes relative to itself over a cardiac cycle, and how it
responds to a maneuver — plausibly *is* shared across subjects and should be exactly the
kind of thing population pretraining is good at.

**Testable hypothesis**: population pretraining substantially improves the *shape/relative*
component of BP prediction quickly (small data), but the *absolute/calibration* component
remains subject-specific and does not improve much with more pretraining — it needs a
small amount of per-subject correction regardless of core quality.

**How we test it** (a new, explicit analysis axis, not just reporting cc_abs and AMAE
side by side as unrelated numbers): decompose every evaluation into —
- **Shape term**: correlation between predicted and true BP trajectory (extend beyond the
  current min/max-only `cc_abs` to a full-waveform correlation where useful).
- **Bias/calibration term**: mean signed error per subject (`mean(predicted − true)`).
- **Corrected accuracy**: fit a per-subject affine correction (scale + offset, 2
  parameters, using only a handful of calibration points from the target subject) on top
  of the core+readout prediction, then re-report AMAE/BHS/AAMI.

**Potential findings and what each would mean:**
- *If the hypothesis holds* (shape term improves fast with pretraining; a 2-parameter
  affine correction closes most of the remaining absolute-error gap): this is a strong,
  clean, clinically deployable result — "a population-pretrained shape prior + minimal
  per-subject calibration" is a *better* story than "train a small readout NN per
  subject," because it needs less data, is simpler to justify to a clinical audience, and
  mirrors real calibration protocols. It also reframes the readout: the affine correction
  might replace or supplement `SubjectReadout`.
- *If it fails* (bias errors are not well-explained by a simple affine term — i.e.,
  subject-specific errors are structured/nonlinear, not just an offset): that's still an
  important, reportable negative result. It tells us the individual-vs-foundation gap is
  about something deeper than a fixed per-subject calibration constant, and motivates
  either richer per-subject readouts (mechanism 2 above) or additional per-subject
  covariates (e.g., anthropometrics, resting HR) as calibration inputs.

Either outcome is publishable and informative; the point is to test it explicitly rather
than assume the favorable case.

### 3.4 Intermediate representation: reconstructing our own EIT image (candidate second
architectural pillar, conditional on data availability)

`src/models/eit_recon.py` implements `EITReconstructor` (channels → conductivity image)
and `EITForwardOperator` (differentiable image → boundary-measurement operator, for
physics data-consistency). The motivating idea: a physically-grounded image bottleneck
(what's actually happening inside the ring, in physical units) should generalize better
across subjects than raw channel amplitudes, which are confounded by subject-specific
electrode contact and geometry — the same "shared structure vs. idiosyncratic nuisance"
argument as §3.3, but applied to the *input* representation instead of the *output*.

**Honest caveat, load-bearing for how much weight this idea can carry**: `EITForwardOperator`
currently defaults to a *learnable random matrix*, not the ring's real electrode
geometry/drive pattern. Without the true forward model, the "data-consistency" loss does
not confer physical grounding — it's just another learned nonlinear layer with an extra
loss term, no more principled than a plain conv core. **Getting the real ring
geometry/drive pattern is the actual prerequisite for this to be a legitimate second
pillar**, not further architecture work. Decision point, to resolve early:
- If the geometry is obtainable → build the real forward operator, and this becomes a
  first-class comparison: does a physics-constrained learned-image representation
  transfer better (cross-subject, cross-maneuver) than raw channels or the existing
  1-step-Newton image? This would be genuinely novel.
- If not obtainable → keep `eit_recon.py` as an empirically-motivated (not
  physically-validated) intermediate representation, evaluate it purely on downstream
  transfer performance, and do not claim physical grounding in the paper — let the
  generalization numbers carry the argument instead.

---

## 4. Models

All architectures implement `forward_core` / `forward_readout` on `BasePviLearner`
(`src/models/base_model.py`), so any of them can serve as a foundation core and is
transferable via the same mechanism. Selected via `arch` tag through
`src/foundation/model_factory.py` / `src/models/_model_mapper.py`.

| Tag | Class | Best input | Role | Priority |
|-----|-------|------------|------|----------|
| `crt` | `PviCNNTransformer` | impedance (64ch) | **Primary Core S** — conv + LSTM positional encoder + transformer. Matches the architecture family with the best existing (non-foundation) results (PW15, 0.928 local / ~0.46 disjoint), so it's the most de-risked choice for the supervised core. | **Primary** |
| `mae` | `PviMaskedTransformer` | impedance, image patches | **Primary Core U** — tokenized MAE-style encoder, natural fit for the masked-reconstruction pretext. | **Primary** |
| `linear` | `PviLinearRegression` | signal | Sanity lower bound. | Diagnostic |
| `mlp` | `PviMLP` / `PviCore` | signal, impedance | Scaffold used to validate the core/readout + transfer wiring cheaply before scaling up. Not a production model — do not report scaffold numbers as final results (see §6.2, the checkpoint-export lesson). | Scaffold only |
| `cnn` | `PviCNN` | signal, impedance | Local temporal-conv baseline / ablation point (how much does the recurrent/attention piece of `crt` actually buy). | Ablation |
| `dnclstm` | `PviDenseNetConvLSTM` | image (40×40 EIT) | Paper-faithful 3D-DenseNet + Conv-LSTM + spatial bilinear readout — the most literal replication of Wang et al.'s actual core. Pairs with §3.4's learned-image track. High scientific fidelity, high cost/risk. | Stretch goal |
| `samba` | `PviSamba` (WIP) | long raw sequences | SSM/Mamba for high-rate impedance. | Deprioritized (WIP, not required for the primary pillars) |

**Why `crt`/`mae` and not a from-scratch architecture**: `crt` already has a directly
comparable non-foundation reference point (PW15) in the same codebase family, which lets us
cleanly attribute gains/losses to the foundation *paradigm* rather than to an unfamiliar
architecture. `mae`'s token/patch structure is a natural match for the masked-reconstruction
SSL pretext. Given the actual compute budget (§7.2), we are **not** running the full
7-architecture × 2-core matrix in production — `crt` (Core S) and `mae` (Core U) are the
production spine; `linear`/`mlp`/`cnn` stay as fast diagnostics and ablations;
`dnclstm`/`samba` are stretch goals only if the primary pillars land with time/compute to
spare.

---

## 5. Experiment matrix

Primary pillars first (these carry the paper); secondary/exploratory tracks after.

### Primary

| Exp | Paper analog | Question | Hypothesis | Metrics | Depends on |
|-----|--------------|----------|------------|---------|------------|
| **A** | Fig 2 | How fast does a **per-subject, trained-from-scratch** model learn, as a function of that subject's own data budget? | Individual models overfit badly at low budgets (already observed: cc_abs −0.12 @ 4 min) and only become competitive once given substantial per-subject data. | cc_abs, AMAE/RMSE, BHS/AAMI, vs. minutes of subject-specific data | Individual baseline only |
| **B** | Fig 3b | Does **foundation transfer** beat training from scratch at low per-subject data budgets, and does the gap close as budget grows? Which calibration mechanism (§3.2: linear readout / MLP-or-partial-finetune / affine correction) wins, and at what budget? | Foundation transfer dominates at low budgets; the gap narrows as individual data grows (already suggested by AMAE converging by 64 min: 5.3 vs 6.0 mmHg); the shape/calibration decomposition (§3.3) explains *why*. | Same as A, run per calibration mechanism, per core (S/U), **with a matched-capacity random-init control** (§8.1) to isolate "pretraining helped" from "fewer free parameters survived low data" | Core S and/or U (pretrained); matched-capacity control model |
| **C** | Fig 3c–g | Does a core/readout trained on **resting baseline only** generalize to physiologically distinct states (Valsalva, pressor) without retraining? | Foundation transfer generalizes across maneuvers meaningfully better than an individual model trained only on baseline, because the shared core captures maneuver-invariant structure. This is the least-explored, most novel claim we can make. | Same metrics, evaluated OOD (train baseline, test Valsalva/pressor) + within-maneuver and reverse-direction controls | Core S and/or U |
| **G** | — | **Core U vs Core S** head-to-head: does self-supervised (label-free) pretraining match or beat supervised pretraining for downstream BP transfer and OOD generalization? | Open — Core U currently underperforms Core S, but the comparison so far used an unresolved scale/calibration bug (§6.3) and scaffold-only (MLP+signal) settings; needs to be re-run on production settings before concluding anything. | Same as B/C, both cores | Both cores at production settings |

### Secondary / exploratory (kept, demoted — not load-bearing for the paper's core claims)

| Exp | Description | Why demoted |
|-----|-------------|-------------|
| **D** | In-silico perturbation / "digital twin": perturb impedance input, observe predicted-BP response, channel sensitivity (`src/analysis/interpretability.py`). | Needs a physiology-grounded validation criterion to be convincing (e.g., a known-direction prediction: does perturbing a specific channel move predicted BP in the physiologically expected direction), not just raw sensitivity magnitudes. Low-dimensional readout makes rich "digital twin" claims harder to support at Wang-et-al. scale. Keep as supplementary figure, add a physiology sanity check before reporting. |
| **E** | "Functional barcode" — predict subject/maneuver identity from readout weights; latent-space structure (UMAP) by maneuver/HR/BP. | Same low-dimensional-readout caveat as D; genuinely interesting if it lands, but should not be a headline claim. |
| **F** | EIT reconstruction sub-track (§3.4): raw channels vs. Newton image vs. learned reconstruction, on downstream BP/transfer accuracy. | Promoted to a candidate *second pillar* only if real ring geometry is obtainable (§3.4); otherwise stays exploratory/supplementary. |

**Multi-task readout** (BP + maneuver classifier + HR decoder, `src/foundation/multitask.py`)
supports Exp D/E as an interpretability probe, not a primary-metric mechanism.

---

## 6. Metrics

- **cc_abs** (`bp_accuracy` in `src/models/perf_metrics.py`): mean Pearson *r* of predicted
  vs. true SBP and DBP (min/max of the waveform). Primary correlation/shape metric so far;
  **not** a substitute for absolute accuracy — see §3.3.
- **AMAE / ARMSE** (`metrics_waveform`): mean absolute / RMS error in mmHg. This is the
  number that maps onto clinical standards and should be reported alongside cc_abs, always.
- **BHS / AAMI grading** (`metrics_fiducial`): the actual clinical validation standards for
  BP measurement devices — cite these directly in the paper rather than only reporting raw
  mmHg error, since reviewers in a broad-audience journal will want the clinically
  standardized framing.
- **New, proposed (§3.3): shape/bias decomposition.** Per-subject mean signed error
  (bias/calibration term) reported separately from the shape/correlation term, plus
  post-affine-correction accuracy. This is the key new analysis this plan adds beyond what
  was already implemented.
- **Statistical rigor requirement**: every reported curve must be **multi-subject,
  multi-seed, with paired comparisons** (foundation vs. individual evaluated on the *same*
  held-out subjects) and confidence intervals (`src/analysis/budget_curves.py` already
  supports mean/sem aggregation across seeds/subjects — it has just not been run at scale
  yet; see §7.4).

---

## 7. Known issues and near-term fixes (priority-ordered)

These were found during a critical review of the codebase and results as of 2026-07-01, and
should be resolved before further production compute is spent, roughly in this order:

### 7.1 Orphaned holdout set (methodological rigor gap)
A dedicated, never-touched 5-subject holdout (`branch="holdout"`) exists from the original
data engineering but is not referenced anywhere in `src/foundation/`. Every number so far
comes from an in-run `disjoint` re-split of the same ~91-subject pool. **Fix**: wire
`branch="holdout"` into `transfer.py`/`budget_exp.py` as a final-report-only evaluation
path; keep `main`-disjoint for iteration/tuning. Report headline numbers on holdout once,
not repeatedly.

### 7.2 Compute-plan mismatch
The original compute plan assumed 2× H100 NVL 94GB. The real cluster allocation is **1
GPU, 16 CPUs, 250GB RAM**, 10-day SLURM jobs. This is the root cause of several stalled
production jobs (424518, 424519, 424520, 424524) and is why the model/experiment scope in
§4–5 is deliberately trimmed relative to earlier, more expansive versions of this plan.

### 7.3 Core U scale/calibration bug (investigate before spending a production SSL job)
At the 64-min budget, Core U reaches cc_abs 0.307 with **AMAE = 98 mmHg** — a
physiologically impossible absolute error alongside a plausible-looking correlation. This
points to a units/normalization mismatch between the SSL-pretrained core's feature scale
and what the readout expects, not necessarily "the SSL objective is bad." Isolate this
(likely a quick fix) before committing a full production MAE+impedance SSL run.

### 7.4 Missing SSL linear-probe monitoring
`ssl_pretrain.py` only logs mask/forecast loss; it never checks downstream BP relevance
during training. Add a periodic (e.g. every 10 epochs) frozen-linear-probe BP-correlation
check inside the SSL loop, so a bad SSL run is visible in-flight rather than discovered
after a full job completes.

### 7.5 Frozen-core + linear-readout may under-sell the core
Already the reason Exp B is now explicitly designed (§3.2, §5) to compare multiple
calibration mechanisms rather than assume the cheapest one.

### 7.6 Statistics
Every current number is n=1 subject (013), n=1 seed. Not wrong for early debugging, but no
curve should go in a figure until run across a real held-out cohort (§6, last bullet).

---

## 8. Additional design notes from critical review (2026-07-01)

### 8.1 Matched-capacity control for Exp B
To attribute Exp B's low-budget gap to *pretraining* rather than *parameter count*, add a
control arm: a from-scratch model with capacity matched to the readout (not the full core),
randomly initialized, trained on the same low-budget data. If foundation-transfer still
beats this control, that's real evidence pretraining helps, not just that having fewer free
parameters survives low-data regimes better.

### 8.2 Scope discipline
7 architectures × 2 cores × 7 experiments × sub-tracks (EIT recon, multi-task heads,
interpretability) is far more than a single-GPU, 10-day-job cluster allocation can run to
convergence. The model table (§4) and experiment matrix (§5) above are the trimmed,
prioritized scope: `crt`/`mae` production cores, Exp A/B/C/G as the paper's spine, D/E/F as
supplementary. Do not expand scope back out without a specific reason tied to a primary
pillar landing successfully with compute to spare.

---

## 9. Milestones / roadmap

1. ~~Core/readout refactor (`forward_core`/`forward_readout`) + freeze/transfer utility +
   multi-head API + smoke tests.~~ **Done.**
2. ~~Train-budget control + metrics/plot aggregation.~~ **Done.**
3. ~~SSL dual pretext (masked + forecast) + multi-arch core factory (`crt`, `mae`, `cnn`,
   `mlp`).~~ **Done** (code); production-scale SSL pretrain **not yet completed** (§7.3
   blocks this).
4. ~~Supervised cohort core S scaffold (MLP+signal).~~ **Done**; production (`crt`+impedance)
   pretrain **in progress / stalled** (§7.2).
5. **Next**: resolve §7.1–7.4 (holdout wiring, compute-plan correction, Core U scale bug,
   SSL probe monitoring) — cheap, unblocks everything downstream.
6. Production Core S (`crt`+impedance) and Core U (`mae`+impedance) pretrain to convergence,
   exporting best (not last) checkpoint, evaluated once on `branch="holdout"`.
7. Exp A/B (with the matched-capacity control, §8.1, and the 3 calibration mechanisms,
   §3.2) across a real held-out cohort (≥10–20 subjects × ≥3 seeds), paired statistics.
8. Exp C (OOD across maneuvers) — the second pillar.
9. Exp G (Core U vs Core S) once both are at production scale and §7.3 is resolved.
10. Shape/bias decomposition analysis (§3.3) applied retroactively to all of the above.
11. If time/compute remain: Exp D/E (interpretability, with the physiology-grounding
    caveat from §5), Exp F / `dnclstm` (conditional on ring geometry, §3.4).

---

## 10. Empirical findings to date (evidence base)

Preserved from the working run log — the concrete numbers behind the reasoning in §1–3 and
§7. All results below are from `branch="main"` disjoint re-splits (not the dedicated
holdout set — see §7.1), mostly on the **scaffold** MLP+signal configuration unless noted;
treat as directional evidence for debugging/design, not final results.

### 10.1 Reference: `pvi_ml` / `ml-experiments` (non-foundation, separate codebase)

| Model | Split | Test cc_abs | Notes |
|-------|-------|-------------|-------|
| PW15 CRT (depth-1, cosine LR) | `local` (same subjects train+test) | **0.928** | Not a fair generalization baseline — cited only as an existing per-cohort ceiling reference, not a target foundation-transfer should match. |
| `abl-crtsin-bioz-to-waveform` (CRT) | `disjoint` (subject holdout) | ~0.46 peak, 0.21 final | The real motivating gap (§1.1). |
| Longitudinal full-finetune, subject002 | within-subject, d00→d01 | 0.586 @ ep998 | Full fine-tune (not frozen core), from `ml-experiments/_long`; useful upper-bound reference for what "more calibration capacity" can reach. |

### 10.2 Foundation scaffold (MLP + `signal`, 1-channel input) — pipeline-validation only

| Job | Task | Result |
|-----|------|--------|
| 424458 | Core S pretrain, 500 ep, disjoint | Best test cc_abs **0.445 @ epoch 40**; **0.274 @ epoch 499** (0.17 Pearson lost by originally exporting the last epoch instead of best — bug now fixed, `export_core.py` tries best-checkpoint first automatically). |
| 424459 | Core U SSL pretrain, 500 ep | Completed (~35 min); SSL loss → ~0; no BP signal monitored during training (§7.4). |

### 10.3 Transfer to subject013 (frozen core, linear readout only, MLP scaffold)

| Core | Test cc_abs | Notes |
|------|-------------|-------|
| Core S | **0.336** | Flat from epoch 0 — consistent with a linear-only readout having little room to move a frozen core's output (§7.5). |
| Core U | 0.305 | Peak ~epoch 449. |

### 10.4 Exp B budget curve, subject013 (job 424512)

**@ 4 min budget:**

| Method | cc_abs | AMAE (mmHg) |
|--------|--------|-------------|
| foundation_S | **0.336** | 78.6 |
| foundation_U | 0.326 | 98.3 |
| individual | −0.121 | 12.3 |

**@ 64 min budget:**

| Method | cc_abs | AMAE (mmHg) |
|--------|--------|-------------|
| foundation_S | 0.336 | **5.3** |
| foundation_U | 0.307 | 98 ← flagged bug, §7.3 |
| individual | 0.088 | **6.0** |

Read together with §3.3: cc_abs saturates fast and stays roughly flat across the budget
range for foundation_S; AMAE for individual closes most of the gap to foundation_S by 64
min. This is the empirical basis for the shape-vs-calibration hypothesis in §3.3, and for
requiring the matched-capacity control in §8.1 before concluding pretraining (rather than
parameter count) explains the 4-min result.

### 10.5 Production runs — status as of 2026-07-01 (blocked by §7.2/§7.3)

| Job | Component | Status |
|-----|-----------|--------|
| 424523 | Core S, CRT + impedance | In progress; best test cc_abs 0.327 @ epoch 10 (partial) |
| 424518 | Core S, CRT + impedance, lazy HDF5 (no parquet cache) | Abandoned — too slow |
| 424524 | Core U, MAE + impedance | Stopped ~epoch 16 |
| 424519, 424520 | Core U, MAE + impedance | Failed (import/argparse bugs; fixed in repo) |

---

## 11. Operational playbook (how to actually run this)

### 11.1 Environment

```bash
cd /mmfs1/projects/ece_bst/lsanc68/fundational_pvi
uv sync
source .venv/bin/activate
source env/cluster.env    # PVI_DATA_ROOT, PVI_CACHE_ROOT, PVIPROJECT_ROOT
```

### 11.2 Parquet cache (build once per input-mode before production runs)

```bash
sbatch src/launch_build_cache.sh              # signal / default input-mode cache
sbatch src/launch_build_cache_impedance.sh     # impedance cache (needed for crt/mae production)
```

### 11.3 Core pretraining

```bash
# Core S — supervised cohort pretrain, production settings (crt + impedance)
sbatch src/launch_crt_pretrain.sh
# LOGDIR=foundation-pretrain-crt -> artifacts/foundation-pretrain-crt/main/

# Core U — SSL pretrain, production settings (mae + impedance)
sbatch src/launch_mae_ssl.sh
# LOGDIR=foundation-ssl-pretrain-mae -> artifacts/foundation-ssl-pretrain-mae/main/
```

Recommended production defaults (`FoundationConfig` / CLI):

```bash
python -m src.foundation.pretrain \
  --input-mode impedance --output-mode waveform --arch crt \
  --batch-size 256 --max-cache 150 --eval-every 5

python -m src.foundation.ssl_pretrain \
  --input-mode impedance --arch mae --ssl-arch mae \
  --batch-size 256 --max-cache 150
```

### 11.4 Transfer to a held-out subject

```bash
CORE=artifacts/foundation-pretrain-crt/main/foundation_core.pt \
  sbatch src/launch_transfer.sh
# arch auto-loaded from foundation_core_meta.json next to the core file
```

### 11.5 Budget curves (Exp A/B/G)

```bash
python -m src.foundation.budget_exp --subject subject013 \
  --core-s artifacts/foundation-pretrain-crt/main/foundation_core.pt \
  --core-u artifacts/foundation-ssl-pretrain-mae/main/foundation_core_U.pt \
  --arch crt
```

Scale to a full cohort (§7.6, §9 step 7) by looping over held-out subjects and seeds; each
run is small and highly parallelizable across SLURM array jobs.

### 11.6 OOD across maneuvers (Exp C)

Train on baseline-only dataset; evaluate with `evaluate_ood`
(`src/foundation/evaluation.py`) on a separate Valsalva/pressor dataset. No dedicated launch
script yet — needs a small `ood_exp.py` driver analogous to `budget_exp.py` (open item).

### 11.7 Tests

```bash
python -m pytest tests/ -q           # local/CPU; no real data needed (synthetic batches)
sbatch src/launch_test.sh            # on cluster, off the login node
```

### 11.8 Key paths

| Resource | Path |
|----------|------|
| Project root / artifacts | `$PVIPROJECT_ROOT` (`env/cluster.env`) |
| Raw data | `$PVI_DATA_ROOT/main/*_masked.h5` |
| Parquet cache | `$PVI_CACHE_ROOT` (signal), `${PVI_CACHE_ROOT}_impedance` |
| PW15 reference weights (non-foundation) | `/home/lsanc68/artifacts/pw15-crt-bioz-to-waveform-ablation-depth-1-cosine/checkpoints/dataset_lazy_checkpoints_best.pth` |
| Disjoint CRT ablation reference | `ml-experiments/current/artifacts/abl-crtsin-bioz-to-waveform/main/` |

---

## 12. Deviations from Wang et al. (state explicitly when writing this up)

- No noise ceiling (no repeated identical stimuli) → `CC_norm` not reproducible; we use
  `cc_abs` (raw Pearson r) + clinical error (AMAE/BHS/AAMI) instead — see §6.
- Low-dimensional readout (BP: 50-sample waveform / 2 scalars, vs. thousands of neurons):
  this is the central, load-bearing difference discussed in §1.3 and §3.3, and it's why
  Exp D/E are demoted rather than treated as primary claims.
- No perspective/ray-tracing module (no geometric analog in this domain) — `dnclstm`
  (§4, §3.4) is the closest architectural analog to Wang et al.'s actual core (3D-conv
  DenseNet + Conv-LSTM), kept as a stretch goal.
- MSE/MorphologyLoss rather than Poisson NLL (correct for continuous BP vs. spike counts).

## 13. Risks

- Cycle-normalized time means the model sees cardiac *phase*, not absolute time — feed
  HR/timing via `stats` for anything timing-sensitive.
- BP ground-truth quality under maneuvers (motion artifact) — rely on QC masks; report SQI.
- The baseline→pressor distribution shift may exceed what a frozen core can bridge; may
  need partial fine-tune for Exp C (report both frozen and partial-finetune results).
- The learned-EIT-reconstruction pillar (§3.4) depends on obtaining real ring
  geometry/drive-pattern data; resolve this early, it gates how much weight that track can
  carry.
- Compute is 1 GPU / 10-day SLURM jobs, not the originally assumed 2×H100 — scope
  discipline (§8.2) is required to land the primary pillars within this budget.
