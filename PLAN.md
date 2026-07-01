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

We already have direct, paired evidence that the straightforward approach — train a
supervised model end-to-end per architecture, pooled across subjects — **does not
generalize across subjects**. The clearest single fact: in the Nature paper's own
architecture sweep (Wang lab; 20 "population-within" (PW) models, §10.1), the *identical
trained model* `pw15` (CRT, BioZ input, waveform output) scores **AMAE 3.69 mmHg /
r²=0.85 on its own in-distribution test split**, and **AMAE 9.85 mmHg / r²=0.07** when the
*same weights* are evaluated on a true holdout set of unseen subjects. Same architecture,
same capacity, same task — the only thing that changed is whether the test subject was
represented in training, and r² collapses by ~12x. This pattern holds across every
architecture and modality in the sweep (§10.1), and our own separate `ml-experiments`
codebase shows the same shape (CRT: cc_abs 0.928 on `split_mode="local"` vs. ~0.46 peak /
0.21 final on `split_mode="disjoint"`, `abl-crtsin-bioz-to-waveform`). This is the central
empirical fact motivating this project: **naively pooling data across subjects and training
one model does not produce a representation that transfers to new people — and this isn't a
capacity or architecture problem** (see §3.5: the paper's own ablation shows more parameters
didn't fix it either). It's a generalization problem, specifically.

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

- **We are not claiming** that low output dimensionality alone predicts *generalization*.
  It does predict that *in-distribution* accuracy is fast and good — and the data bears
  this out (§10.1: PW/SS in-distribution AMAE 3.7–4.7 mmHg, r² up to 0.85, reached in a few
  hundred epochs). What it does *not* predict, and what the same data shows breaking down
  completely (the pw15 example in §1.1: r² 0.85 → 0.07, same weights, only the test subject
  changed), is cross-subject generalization. In-distribution fit quality and generalization
  are close to decoupled properties in this domain — see §3.3 for why, and what to test.
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

**Do not assume "BP is low-dimensional, so we should get high accuracy fast."** In-distribution,
that part is actually true (§10.1: PW/SS AMAE 3.7–4.7 mmHg, fast). What it does not predict
is generalization, and — separately — it does not predict that a *correlation-shaped*
metric (r²/cc_abs looking good) implies *absolute* accuracy is good. We have **two
independent confirmations** of that second trap: our own Core U at the 64-min budget
reaches `cc_abs = 0.307` (plausible-looking) alongside `AMAE = 98 mmHg` (physiologically
impossible) (§10.5); and in the literature sweep, `ss09`/`ss11` (CNN, subject-specific)
show r²≈0.37–0.57 (looks fine) alongside **AMAE 26–29 mmHg** (also physiologically
impossible) (§10.1). Two different architectures, two different datasets, the same failure
signature: a shape-shaped metric can look reasonable while the absolute number that
actually matters clinically is broken. Treat this as a real, recurring failure mode to
guard against, not a one-off bug — see §6 for the reporting standard this motivates.

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

**Independent supporting evidence for the affine-correction idea specifically**: in the
literature sweep (§10.1), a *per-subject linear regression* (`ss03`, subject-specific,
BioZ→waveform) reaches AMAE **5.0 mmHg** — competitive with the far more complex CRT/CRS
subject-specific models. That's evidence that *within one subject*, the impedance→BP
relationship is already close to affine/well-conditioned — the hard, nonlinear problem is
in the *cross-subject* mapping (representation learning), not the per-subject calibration
step. This is exactly the asymmetry the affine-correction mechanism (§3.2, item 3) is built
to exploit: let the core carry the hard cross-subject nonlinearity, and keep the per-subject
correction as simple as the data suggests it can be.

**How we test it** (a new, explicit analysis axis, not just reporting cc_abs and AMAE
side by side as unrelated numbers): decompose every evaluation into —
- **Shape term**: correlation between predicted and true BP trajectory (extend beyond the
  current min/max-only `cc_abs` to a full-waveform correlation where useful).
- **Bias/calibration term**: mean signed error per subject (`mean(predicted − true)`).
- **Corrected accuracy**: fit a per-subject affine correction (scale + offset, 2
  parameters, using only a handful of calibration points from the target subject) on top
  of the core+readout prediction, then re-report AMAE/BHS/AAMI.
- **Label gap** (adopted from the source paper's own robustness methodology, §10.1): a
  model-independent measure of how far a target subject's/session's BP distribution differs
  from the training cohort's. The paper found holdout-set label gap ≈2.6x the typical
  train-test label gap, and (for LR/MLP) correlation between label gap and AMAE of
  0.44–0.61. Adopt this directly as a diagnostic: compute label gap for every subject in
  our own holdout set (§7.1) *before* trusting it as representative, and pair it with the
  shape/bias decomposition above to separate "this subject transfers poorly because of a
  measurable distribution shift" from "this subject transfers poorly for some other,
  unexplained reason" (architecture, insufficient calibration data, non-affine effects).

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

**Update (2026-07-01)**: a real forward operator likely already exists. The excerpted paper
methods describe a FEM/CEM (complete electrode model) forward solver with a one-step
Gauss-Newton linearized inverse, built in MATLAB (`distmesh` triangulation, first-order
Lagrange FEM, reciprocity-theorem Jacobian) — almost certainly what already produces the
`img` field in the HDF5 data. This changes the blocker from "unknown geometry, needs
acquisition" to "existing MATLAB artifact, needs porting" (export the stiffness
matrix/Jacobian/electrode injection-measurement pattern into `EITForwardOperator.weight`) —
a well-scoped engineering task. That said, the paper explicitly reports **image and BioZ
inputs "yielded similar results"** for BP accuracy (§10.1), so this track's justification
should stay the generalization/interpretability angle (§3.4's original motivation, and Exp
F), not an accuracy claim — it remains secondary priority under §8.2's scope discipline.

### 3.5 Model scale: what "bigger" should mean here, and what it should not

The project's working hypothesis has shifted from validating pipeline scaffolds (small MLP
cores, `signal` input — useful only for testing the core/readout/transfer wiring, see
§4 and §10.3) toward training a substantially larger foundation core, sized well beyond
those scaffolds (they are debugging harnesses — `num_features=512`/6 layers for the MLP
core, `d_model=64`/2 layers for the current `mae` config — not a serious capacity target),
intended to generalize better and overfit less, so it can later be distilled into something
small enough for a wearable. This section is about what "bigger" should concretely mean.

**Against naive scaling of the existing supervised recipe** (a narrower, more precise
claim than "bigger doesn't help," see the correction below). The paper's own ablation study
(Extended Data Table 3) found that compressing PW15 (CRT, 7.5M params) to 2.12M params via
depth reduction *slightly improved* every metric — the authors' own conclusion was the
extra depth was "redundant at this scale." CRT (7.5M) and CRS (~2.5M) land within noise of
each other on every protocol (PW: 3.69 vs 3.80 AMAE; PD: 9.07 vs 9.30). **On this task,
this data volume, trained the same single-task-supervised way, more capacity did not buy
more generalization.**

**Correction — this claim is narrower than it might read.** The paper's sweep covers five
architecture *families* (LR, MLP, CNN, CRT, CRS), essentially two meaningfully different
inductive biases (feedforward vs. conv+recurrence+attention hybrid), all trained the same
way (single-task, BP-supervised). That's real evidence that *this specific recipe*
plateaued — it is not evidence that capacity in general is useless, or that a
fundamentally different architecture (richer attention — relative/rotary positional
encoding, cross-session or cross-subject conditioning, hierarchical multi-scale temporal
modeling over multiple cardiac cycles) or a different training methodology (self-supervised
pretraining at scale, contrastive objectives across subjects) couldn't unlock headroom a
5-architecture, single-methodology sweep never tested. Don't over-read "CRT plateaued" as
"bigger models don't work here" — it specifically rules out one thing (inflating the
existing supervised CRT/CRS recipe), not the broader design space.

**What the evidence does support.** The paper's own diagnosis for *why* PW/PD models don't
generalize is **data breadth**, not capacity: *"the limited generalizability of CRT models
likely arises from insufficient data breadth, which prevents the learned representations
from extending to out-of-distribution subjects"* — and they explicitly contrast their
~225h/91-subject dataset against industry datasets with "hundreds of thousands of
individuals and millions of hours." We cannot get that much more data, but we can get more
**effective** breadth per subject/session by not requiring every training example to carry
a synchronized BP label — which is exactly what Core U (self-supervised, §3.1, and its
tokenized variant, §3.6) is for. **A large core earns its capacity through a data-hungry,
label-free pretraining objective that can draw on more of the available signal (all
sessions, all maneuvers, not just the BP-labeled/synchronized subset) — not through
inflating the existing single-task supervised CRT/CRS recipe.** Concretely: prioritize
scaling Core U's pretraining data footprint, task diversity (masked reconstruction +
forecasting + the discretized pretext in §3.6), and — separately from footprint —
richer architectural components within Core U (attention variants, longer context,
cross-session conditioning) over scaling Core S's parameter count on the existing recipe.

**Inductive bias matters more than raw size** *within the tested families*. The robustness
analysis in §10.1 is telling on this point: CNN (moderate capacity, no
recurrence/attention) is the *least* stable architecture in the entire sweep — SS AMAE
ranging 4.77–56.8 mmHg with no significant correlation to any diagnostic metric (and
directly visible in our own numbers: `ss09`/`ss11` show plausible r² alongside AMAE 26–29
mmHg — the same shape-vs-calibration disconnect as Core U, §3.3, §7.3, independently
confirmed). CRT/CRS (recurrence + attention, matched to the cardiac-cycle-structured
signal) are the stable, generalization-favoring choices, and this holds longitudinally too
(§10.1: CRT degrades 1.21 mmHg over 5 days without recalibration vs. 21.21 mmHg for linear
regression). Given the correction above, treat this as "CRT/CRS-family inductive bias is a
good, evidence-backed starting point," not "the ceiling of what architecture can achieve" —
richer attention/context/conditioning built on top of that same family (not a return to
CNN or feedforward) is the scaling axis with evidence behind it, alongside the
self-supervised data-breadth argument.

**Elevated priority: CRS/`samba`.** `samba` (the CRS analog) should not be treated as a
deprioritized WIP architecture (as an earlier version of this plan had it). In the paper's
own sweep, CRS *wins* the subject-specific table outright (`ss17`: AMAE 4.09, the best
number in any table) and is statistically indistinguishable from CRT everywhere else,
including longitudinal robustness. It's promoted to co-primary alongside `crt` (§4).

**Distillation.** Distilling a large pretrained core into a smaller deployable model is a
sound *deployment* step once the core has actually learned something a smaller model
couldn't — but distillation transfers a teacher's existing knowledge, it does not create
generalization the teacher didn't have. The generalization gain has to come from the
data-breadth/self-supervision/architecture arguments above, not from the distillation step
itself, so Core U's pretraining quality (not the eventual student's size) is the thing to
get right first.

### 3.6 Discretized SSL pretext ("tokenizer") — a HuBERT/WavLM-style extension to Core U

HuBERT and WavLM (self-supervised speech representation learning) train a masked-prediction
objective against **discrete cluster-ID targets** (from k-means over features, iteratively
refined) rather than continuous reconstruction targets; WavLM adds denoising/overlap
augmentation and relative-position attention. This consistently outperforms continuous
masked-reconstruction for downstream representation quality in the speech SSL literature,
because classification over a discrete vocabulary is a more stable objective, and — the
part most relevant here — it forces the model to discard nuisance variation (in speech:
pitch, speaker identity, loudness) that a continuous MSE target is otherwise forced to fit
exactly.

**Why this is a good fit for our specific hypothesis, not just a borrowed trick**: the
"nuisance variation" analogy maps directly onto §3.3's shape-vs-calibration decomposition.
Subject-specific amplitude/baseline (the calibration nuisance) is exactly what a continuous
MSE reconstruction target must fit precisely; a discrete classification target *can* be
built to discard it. If the pretext clusters on normalized/whitened features (removing
per-subject scale and offset before clustering), the resulting vocabulary becomes an
operationalization of "subject-invariant morphological shape" — precisely what the shared
core should learn, and something MSE reconstruction has no explicit pressure to isolate. It
would also plausibly sidestep failure modes like the Core U AMAE=98 bug (§7.3): a
classification loss over discrete bins is scale-invariant by construction, where MSE
against raw units is exactly the kind of objective that produces silent scale/calibration
mismatches.

**Honest risks, not glossed over:**
- Codebook/cluster training is genuinely more fragile than continuous MSE (dead clusters,
  collapse, sensitivity to vocabulary size) — real engineering risk on top of what's
  already running.
- Speech has actual discrete underlying structure (phonemes) for clustering to latch onto;
  it is *not* established that cardiac/impedance morphology has an analogous discrete
  "alphabet" rather than a continuum driven by continuous physiological state (HR,
  contractility, vascular tone). Forcing discretization onto continuous structure can lose
  information rather than help — this is an empirical question to test, not assume.
- Mitigation for the above: we already have a natural discretization axis in the data —
  **cardiac-cycle phase** (`period_length=50`). Cluster jointly over (phase-bin, normalized
  feature) rather than raw features, leaning on structure we already know exists rather
  than hoping k-means discovers it from nothing.

**Update (2026-07-01), de-risking this**: **HuBERT-ECG** (Coppola et al.) already applies
exactly this approach — discrete-target masked-prediction SSL — to 9.1M 12-lead ECGs,
achieving AUROC 0.84–0.99 across 164 cardiac conditions when fine-tuned. ECG is a much
closer analog to our signal (cardiac-cycle-structured biosignal) than generic speech was,
so this is direct evidence the second risk bullet above is likely surmountable in this
signal class, not just a hopeful analogy. Confidence in §3.6 raised accordingly; sequencing
(Core U variant 2, after baseline SSL is validated) is unchanged.

**Recommended sequencing**: this is **Core U, variant 2** — build and validate the baseline
continuous-SSL Core U (`mae`, mask+forecast, §3.1) first, including the linear-probe
monitoring from §7.4, before investing in this. Start with the simplest version (single
round of offline k-means, phase-conditioned, added as a third loss term alongside the
existing mask/forecast MSE terms — not a full replacement, and not the full iterative
HuBERT re-clustering bootstrap) rather than the full pipeline. Evaluate it against the
§3.3 shape/bias decomposition specifically — the falsifiable claim is that it moves the
*shape* term more than continuous SSL does, not just that it moves aggregate cc_abs.

### 3.7 Architecture exploration beyond CRT/CRS (literature review, 2026-07-01)

Prompted by a direct, correct critique of `crt`: it stacks CNN → BiLSTM (via `RRPE`, our
LSTM-based positional encoder) → Transformer — recurrence and self-attention both model
sequential structure, so this is partly redundant, and modern sequence architectures have
largely moved away from this exact pattern. Verified against current (2025–2026)
literature rather than assumed. Ranked by confidence/cost so this doesn't become an
unranked wishlist and undermine §8.2's scope discipline — most items here are *exploratory
backlog*, not commitments; only the first tier is recommended to build soon.

**Tier 1 — build soon (cheap, low-risk, directly evidenced):**
- **Replace `RRPE` with RoPE (rotary position embedding)** in `crt`. Confirmed as the
  current standard for relative positional information in high-performing transformers,
  with proven length-extrapolation properties — a closed-form, non-recurrent alternative
  that directly answers the critique above. Pure module swap; no new training paradigm, no
  new data requirement, minimal implementation risk.
- **Domain-adversarial subject-invariance**: a gradient-reversal layer (GRL) + subject-ID
  classifier as an auxiliary head during core pretraining, explicitly penalizing the core
  for encoding subject-identifiable information. Well-established for exactly this problem
  in the cross-subject EEG/ECG literature (GRL: identity in the forward pass, negated
  gradient in the backward pass — trivial to implement, and slots directly into the
  existing aux-head machinery in `src/foundation/multitask.py`). **This is arguably the
  highest-value item in this section**: unlike an architecture swap, it directly, explicitly
  optimizes for the property §1.1 says we actually need (a subject-invariant core), rather
  than hoping it falls out of scale or architecture choice alone — orthogonal to and
  stackable with any core architecture (`crt`, `samba`, `mae`, or the Tier 2 options below).
  Known risk: adversarial training can destabilize on nonstationary signals (documented in
  the EEG domain-adaptation literature) — needs care (e.g. GRL weight warm-up), not a
  reason to skip it.

**Tier 2 — strong domain fit, moderate effort, explore once the primary Core S/U spine
(§9) is running:**
- **TimesNet-style periodicity-aware encoder**: replace the generic Conv1d/Conv3d frontend
  with an explicit 1D→2D reshape aligned to `period_length=50` (period-index axis vs.
  phase-within-period axis) and inception-style 2D conv blocks that separate intra-period
  (single-heartbeat morphology) from inter-period (beat-to-beat/HRV-like) variation. More
  principled than a generic CNN specifically *because* it's built around structure we
  already know exists and already exploit for masking (`mask01/05/10/15`) — arguably the
  best-motivated non-trivial architecture change found in this review.
- **HiMAE** (hierarchical multi-resolution masked autoencoding for wearables, ICLR 2026):
  produces multi-resolution embeddings rather than collapsing to one temporal scale,
  directly matching our existing multi-scale mask convention. Notably reports
  sub-millisecond on-device inference while beating scale-collapsing foundation models —
  i.e., resolution-aware SSL is architecture-level evidence toward *both* better
  representations and the distillation-for-wearables goal, not distillation as an
  afterthought bolted onto a big model post hoc.

**Tier 3 — exploratory/stretch:**
- **PITN** (arXiv 2408.08488) — **update: code and pretrained weights are public**
  (`github.com/Zest86/ACL-PITN`), moved to §3.8 below as a near-term fine-tuning
  candidate rather than a from-scratch reimplementation.
- **NormWear-style pure-attention encoder** (channel-tokens + shared CLS-pooling token,
  no CNN at all) — an ablation direction for `mae`; lower priority since it tests a
  similar broad hypothesis (pure attention, tokenized) to what `mae` already attempts.
  Pretrained weights exist (§3.8) if a cheap trial is preferred over reimplementation.
- **Pure-Mamba ablation of `samba`** (no sliding-window attention): ECG-specific SSM
  precedent (ECGMamba, S2M2ECG — bidirectional Mamba for multi-beat ECG, reporting faster
  inference at competitive accuracy) suggests this may be worth testing given the 1-GPU
  budget (linear-time scaling vs. attention's quadratic cost).

**Validates existing plan items, no new engineering needed:**
- `microsoft/Samba` (ICLR 2025, "Mamba + MLP + Sliding Window Attention + MLP", stacked at
  the layer level) is confirmed as the real published architecture `PviSamba` already
  mirrors — direct validation that elevating it to co-primary (§3.5, §4) was well-founded,
  not just a naming coincidence.
- **FiLM** (Feature-wise Linear Modulation) is the established name for the affine-subject-
  correction mechanism already proposed in §3.2/§3.3 — cite it properly rather than
  reinventing informally. "Temporal FiLM" suggests a refinement if the simple global-affine
  version proves insufficient: modulate per cardiac-phase rather than with one global
  scale/offset per subject.

### 3.8 Fine-tuning available pretrained checkpoints — a cheap first pass, before any
from-scratch reimplementation

Prompted by the reasonable instinct to just try several of §3.7's candidates rather than
pick one on paper alone. That's expensive if it means reimplementing and pretraining each
architecture from scratch — but several of them have **public code and pretrained
weights**, which changes the cost completely: freeze the pretrained backbone, add a small
trainable input adapter (to accept our channel count/modality) and our BP output head, and
fine-tune only those on our data. This is cheap (hours, not GPU-days), doesn't compete
meaningfully with the production Core S/U compute budget, and lets us test several
architectures' suitability empirically before committing to reimplementing any of them.
**This is the resolution to "try all of them": do this cheap pass broadly, then only
promote to full from-scratch treatment (§3.7 Tier 2/3) whichever ones actually look good
on our data.**

Verified availability (2026-07-01), ranked by how directly usable each is:

- **PITN** — highest priority. Public code *and* pretrained weights
  (`github.com/Zest86/ACL-PITN`). Solves our exact task (cuffless BP) and was evaluated on
  bioimpedance among its three modalities — check specifically whether a
  bioimpedance-trained checkpoint is released separately; if so, this may need only a
  channel-count adapter, not a full modality adapter, making it the cheapest and most
  directly relevant of everything found. Also lets us test its adversarial-augmentation
  and contrastive-BP-similarity components empirically instead of just reading about them.
- **HuBERT-ECG** — public pretrained weights on Hugging Face (multiple model sizes).
  Different modality (12-lead ECG voltage vs. our 64-channel impedance) so this needs a
  real input adapter (swap the frontend, freeze the pretrained transformer blocks) — more
  engineering than PITN, but still far cheaper than pretraining our own §3.6 tokenizer
  from scratch, and this specific trial doubles as a cheap pilot for whether
  discretized-SSL representations transfer into our domain at all before we build one
  ourselves.
- **NormWear** — public pretrained weights (GitHub releases / Hugging Face). Not
  cardiac/BP-specific, but its architecture already handles variable channel counts (mean
  pooling across channels before the shared backbone per its documentation), which is a
  positive sign for adapting it to 64-channel impedance with comparatively little
  engineering.
- **HiMAE** — public checkpoint (`himae_synth.ckpt`), but the "synth" naming is
  unconfirmed — verify whether this is a genuine large-scale pretrained representation or
  a synthetic-data demo/validation checkpoint before relying on it; single-channel-oriented
  (PPG/ECG), so also needs multi-channel handling similar to NormWear's approach.
- **UTransBPNet** — **not pursued for now**: no public code or weights found anywhere
  searched. Its own reported number (Pearson r≈0.61 for SBP, i.e. r²≈0.37) is not clearly
  better than what we can already reproduce from the literature benchmark (§10.1: PD15
  r²=0.32; PW15 r²=0.85 in-distribution), and it's unclear whether its "cross-scenario"
  evaluation (drink/exercise/MIMIC) tests cross-**subject** generalization the way our
  disjoint/holdout protocol does, or cross-**activity** within the same cohort — a
  related but different claim. Adopting it would mean reimplementing a full U-Net+SE+
  cross-attention architecture from the paper description alone with no pretrained
  starting point and no confirmed benchmark advantage — exactly the expensive,
  low-information path §8.2 exists to avoid. Revisit if the full paper (not yet read)
  clarifies the evaluation protocol and the comparison looks more favorable.

**Positioning note.** Foundation models for wearable physiological sensing exist
(NormWear, HiMAE) but aren't BP-specific or evaluated as a population-pretrained
core+readout transfer protocol. BP-specific architectures exist (UTransBPNet, PITN) but
aren't framed or evaluated as true population-pretrained, cross-subject foundation models
with a frozen-core+readout transfer protocol and rigorous held-out-subject evaluation. This
project sits at the intersection of both — worth stating explicitly as a novelty argument,
not just an engineering choice.

Also worth naming for balance: current applied cuffless-BP literature still includes
CNN+BiLSTM+attention designs (e.g. a 2025 PPG-BP paper). The critique that motivated this
section is correct relative to general sequence-modeling SOTA (NLP/speech/generic time
series), which has moved past stacking recurrence and attention — the applied wearable-BP
subfield specifically hasn't fully caught up, which is itself an opportunity, not a sign
the critique is overstated.

**References** (read before implementing; not all were fully accessible during this
review — verify details against the primary source):
- HuBERT-ECG: https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v3.full ,
  code: https://github.com/Edoar-do/HuBERT-ECG
- `microsoft/Samba`: https://arxiv.org/html/2406.07522v1 , code:
  https://github.com/microsoft/Samba
- RoPE / RoFormer: https://www.sciencedirect.com/science/article/abs/pii/S0925231223011864
- TimesNet: https://arxiv.org/abs/2210.02186 (search result summary; verify against paper)
- HiMAE: https://arxiv.org/abs/2510.25785 , code: https://github.com/Simonlee711/HiMAE
- PITN: https://arxiv.org/abs/2408.08488 , code + pretrained weights:
  https://github.com/Zest86/ACL-PITN (§3.8 — highest-priority fine-tuning candidate)
- NormWear: https://arxiv.org/abs/2412.09758 , code + pretrained weights:
  https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear
- HuBERT-ECG pretrained weights: via Hugging Face `AutoModel`, see
  https://github.com/Edoar-do/HuBERT-ECG for loading instructions
- ECGMamba / S2M2ECG: https://arxiv.org/html/2509.03066v1 (S2M2ECG; ECGMamba found via
  search summary, verify independently)
- FiLM: https://arxiv.org/pdf/1709.07871 (original paper); Temporal FiLM:
  https://arxiv.org/pdf/1909.06628
- Gradient reversal / domain-adversarial training: original DANN paper (Ganin & Lempitsky);
  applied to biosignals per search summary, e.g.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12899056/ (neonatal seizure detection,
  representative example of the technique, not a direct precedent for BP)
- UTransBPNet: https://www.nature.com/articles/s41598-025-02963-3 (no public code/weights
  found; §3.8 explains why it's not pursued for now)

---

## 4. Models

All architectures implement `forward_core` / `forward_readout` on `BasePviLearner`
(`src/models/base_model.py`), so any of them can serve as a foundation core and is
transferable via the same mechanism. Selected via `arch` tag through
`src/foundation/model_factory.py` / `src/models/_model_mapper.py`.

| Tag | Class | Best input | Role | Priority |
|-----|-------|------------|------|----------|
| `crt` | `PviCNNTransformer` | impedance (BioZ, 64ch) | **Primary Core S** — conv + LSTM positional encoder + transformer. Best or near-best in every literature protocol (§10.1: PW15 AMAE 3.69, PD15 9.07); the most de-risked choice for the supervised core. | **Primary (co-)** |
| `samba` | `PviSamba` | impedance (BioZ, 64ch) | **Primary, elevated** (was deprioritized as WIP; corrected per §3.5) — the CRS analog *wins* the subject-specific table outright (§10.1: `ss17` AMAE 4.09, best in any table) and matches CRT on PW/PD/longitudinal-robustness. Co-primary alongside `crt`, not a stretch goal. | **Primary (co-)** |
| `mae` | `PviMaskedTransformer` | impedance, image patches | **Primary Core U** — tokenized MAE-style encoder, natural fit for the masked-reconstruction pretext. Per §3.5, this is where "bigger" should be spent — a large, data-hungry, label-free core, not a bigger Core S. | **Primary** |
| `linear` | `PviLinearRegression` | signal, BioZ | Sanity lower bound *and* a useful diagnostic: per-subject (SS) linear models reach AMAE ~5.0 mmHg (§10.1, `ss03`) — evidence that within one subject the impedance→BP relationship is close to affine, supporting the §3.3 calibration hypothesis. | Diagnostic |
| `mlp` | `PviMLP` / `PviCore` | signal, impedance | **Pipeline-validation scaffold only.** Used to test the core/readout/transfer wiring cheaply. Not a production model, not a benchmark — the small-MLP results in §10.3–10.6 describe plumbing tests, not model performance; do not compare them to §10.1 or use them to judge the approach. | Scaffold only — retired from benchmarking |
| `cnn` | `PviCNN` | signal, impedance | Ablation point, and a cautionary one: §10.1 shows CNN is the *least stable* architecture in the entire literature sweep (SS AMAE 4.77–56.8 mmHg, no significant correlation with any diagnostic metric) — moderate capacity without recurrence/attention is not a safe default. | Ablation (do not use as a core) |
| `dnclstm` | `PviDenseNetConvLSTM` | image (40×40 EIT) | Paper-faithful 3D-DenseNet + Conv-LSTM + spatial bilinear readout — the most literal replication of Wang et al.'s actual core. Pairs with §3.4's learned-image track (forward-operator feasibility now upgraded, still secondary given image≈BioZ accuracy per §10.1). | Stretch goal |

**Why `crt`/`samba` (Core S) and `mae` (Core U)**: `crt` and `samba`(CRS) are the two
architectures with directly comparable literature reference points (§10.1) *and* the
strongest cross-sectional + longitudinal robustness — this lets us cleanly attribute
gains/losses to the foundation *paradigm* rather than to an unfamiliar or unstable
architecture (see §3.5 for why CNN is excluded as a core candidate despite superficially
competitive numbers). `mae`'s token/patch structure is the natural match for the
masked-reconstruction SSL pretext, and per §3.5 is where scale should be invested. Given the
actual compute budget (§7.2), we are **not** running the full architecture × 2-core matrix
in production — `crt`/`samba` (Core S candidates) and `mae` (Core U) are the production
spine; `linear` stays as a diagnostic; `cnn` and the small-`mlp` scaffold are explicitly
*not* production candidates (the former for instability, the latter because it was never
meant to be one); `dnclstm` is a stretch goal.

---

## 5. Experiment matrix

Primary pillars first (these carry the paper); secondary/exploratory tracks after.

### Primary

| Exp | Paper analog | Question | Hypothesis | Metrics | Depends on |
|-----|--------------|----------|------------|---------|------------|
| **A** | Fig 2 | How fast does a **per-subject, trained-from-scratch** model learn, as a function of that subject's own data budget? | Individual models overfit badly at low budgets (already observed: cc_abs −0.12 @ 4 min) and only become competitive once given substantial per-subject data. | cc_abs, AMAE/RMSE, BHS/AAMI, vs. minutes of subject-specific data | Individual baseline only |
| **B** | Fig 3b | Does **foundation transfer** beat training from scratch at low per-subject data budgets, and does the gap close as budget grows? Which calibration mechanism (§3.2: linear readout / MLP-or-partial-finetune / affine correction) wins, and at what budget? | Foundation transfer dominates at low budgets; the gap narrows as individual data grows (already suggested by AMAE converging by 64 min: 5.3 vs 6.0 mmHg); the shape/calibration decomposition (§3.3) explains *why*. | Same as A, run per calibration mechanism, per core (S/U), **with a matched-capacity random-init control** (§8.1) to isolate "pretraining helped" from "fewer free parameters survived low data" | Core S and/or U (pretrained); matched-capacity control model |
| **C** | Fig 3c–g | Does a core/readout trained on **resting baseline only** generalize to physiologically distinct states (Valsalva, pressor) without retraining? | Foundation transfer generalizes across maneuvers meaningfully better than an individual model trained only on baseline, because the shared core captures maneuver-invariant structure. This is the least-explored, most novel claim we can make. | Same metrics, evaluated OOD (train baseline, test Valsalva/pressor) + within-maneuver and reverse-direction controls | Core S and/or U |
| **G** | — | **Core U vs Core S** head-to-head: does self-supervised (label-free) pretraining match or beat supervised pretraining for downstream BP transfer and OOD generalization? | Open — Core U currently underperforms Core S, but the comparison so far used an unresolved scale/calibration bug (§7.3) and scaffold-only (MLP+signal) settings; needs to be re-run on production settings before concluding anything. | Same as B/C, both cores | Both cores at production settings |

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
4. ~~Supervised cohort core S scaffold (MLP+signal).~~ **Done, and retired** — per §3.5/§4,
   this scaffold validated the pipeline only; it is not carried forward as a production
   candidate or a benchmark (§10.3–10.6 relabeled accordingly).
5. **Next**: resolve §7.1–7.4 (holdout wiring, compute-plan correction, Core U scale bug,
   SSL probe monitoring) — cheap, unblocks everything downstream.
6. **Tier-1 architecture updates (§3.7), before/alongside production pretraining**:
   replace `RRPE` with RoPE in `crt`; add the gradient-reversal subject-adversarial
   auxiliary head (available to any core). Both are cheap, low-risk, and should land before
   the production pretrain runs so the primary cores already reflect them, rather than
   being bolted on after the fact.
7. **Cheap external-checkpoint fine-tuning pass (§3.8)**, run in parallel with steps 6/8,
   not competing for the same GPU-week-scale budget: adapter-based fine-tune of PITN
   (highest priority — public weights, same task, possibly same modality), HuBERT-ECG, and
   NormWear on our data; verify HiMAE's checkpoint provenance before relying on it. Only
   promote any of these to full from-scratch treatment (§3.7 Tier 2/3) if the cheap trial
   actually looks good — this is how "try all of them" gets done without violating §8.2's
   scope discipline.
8. Production Core S candidates (`crt` **and** `samba`, both BioZ/impedance — §3.5, §4
   elevates `samba` to co-primary) and Core U (`mae`, impedance) pretrain to convergence,
   exporting best (not last) checkpoint, evaluated once on `branch="holdout"`. Prioritize
   scaling Core U's pretraining data footprint/task diversity over Core S's parameter count
   (§3.5 — the evidence argues against naive supervised-model scaling).
9. Exp A/B (with the matched-capacity control, §8.1, the 3 calibration mechanisms §3.2, and
   the shape/bias/label-gap diagnostics §3.3) across a real held-out cohort (≥10–20
   subjects × ≥3 seeds), paired statistics.
10. Exp C (OOD across maneuvers) — the second pillar.
11. Exp G (Core U vs Core S) once both are at production scale and §7.3 is resolved.
12. Shape/bias/label-gap decomposition (§3.3) applied retroactively to all of the above.
13. **Core U, variant 2**: discretized/tokenized SSL pretext (§3.6) — only once the
    baseline continuous-SSL Core U is validated (step 8 + §7.4's linear-probe monitoring
    shows it's learning something useful). Evaluated specifically against whether it moves
    the shape term of §3.3's decomposition more than continuous SSL does.
14. **Tier-2/3 architecture exploration (§3.7)**, if the Tier-1-updated primary spine
    (step 6/8) or the cheap fine-tuning pass (step 7) shows promise: TimesNet-style
    periodicity-aware encoder, HiMAE-style multi-resolution SSL, and any promoted
    checkpoint-fine-tune candidates given full from-scratch treatment. Not before the
    primary pillars land; see §8.2 scope discipline.
15. If time/compute remain: Exp D/E (interpretability, with the physiology-grounding
    caveat from §5), Exp F / `dnclstm` (EIT forward-operator porting now more feasible per
    §3.4's update, still secondary since image≈BioZ accuracy per §10.1).
16. Once a core generalizes well (Exp B/C/G land): consider distillation into a smaller
    deployable model (§3.5) — a deployment step, not a substitute for getting
    generalization right first. HiMAE (§3.7) suggests resolution-aware SSL may partly
    solve this by architecture rather than requiring a separate distillation pass.

---

## 10. Empirical findings to date (evidence base)

§10.1 is the **primary literature benchmark** — the source paper's own controlled
architecture sweep, the strongest evidence we have. §10.2–10.6 are our own run history;
per explicit direction, the small-MLP/`signal`-scaffold numbers in there are **pipeline
validation only** (they tested that the core/readout/transfer wiring works end-to-end),
**not benchmarks** — do not use them to judge the approach or compare them to §10.1.

### 10.1 Primary literature benchmark: Wang-lab architecture sweep (this project's source paper)

Five architectures (linear regression `LR`, `MLP`, `CNN`, hybrid conv+transformer `CRT`,
hybrid conv+Samba `CRS`) × two inputs (`image`, `BioZ`) × two outputs (`waveform`,
`fiducials`) × three partitioning protocols, N=91 subjects, ≈225h of data:

- **SS (subject-specific)**: separate model per subject, metrics aggregated across all 91.
- **PW (population-within)**: pooled training, tested on held-out *sequences* — but from
  subjects also present in training (≈ our `split_mode="within"`/"local").
- **PW→holdout**: the *same trained PW models*, evaluated on entirely unseen subjects — the
  paper's actual generalization test.
- **PD (population-disjoint)**: trained *and* tested with subjects held out from the start
  (≈ our `split_mode="disjoint"`).

**PW (in-distribution) — selected rows, full waveform output:**

| ID | Arch | Input | AMAE | ARMSE | r²(agg) | r²(weighted) |
|----|------|-------|------|-------|---------|---------------|
| pw15 | CRT | BioZ | **3.69** | 4.00 | **0.85** | 0.61 |
| pw19 | CRS | BioZ | 3.80 | 4.11 | 0.84 | 0.57 |
| pw13 | CRT | image | 3.95 | 4.27 | 0.83 | 0.57 |
| pw17 | CRS | image | 4.17 | 4.56 | 0.81 | 0.53 |
| pw05 | MLP | image | 4.62 | 5.08 | 0.75 | 0.45 |
| pw09 | CNN | image | 6.45 | 7.13 | 0.68 | 0.35 |
| pw01 | LR | image | 11.23 | 11.73 | 0.11 | 0.05 |

**PW→holdout (same trained models, unseen subjects) — same IDs:**

| ID | Arch | Input | AMAE | ARMSE | r²(agg) | r²(weighted) | Δ vs. PW in-distribution |
|----|------|-------|------|-------|---------|---------------|--------------------------|
| pw15 | CRT | BioZ | **9.85** | 10.81 | **0.07** | 0.18 | AMAE +6.16, r² 0.85→0.07 (§1.1) |
| pw19 | CRS | BioZ | 10.92 | 11.88 | 0.03 | 0.18 | |
| pw13 | CRT | image | 11.04 | 11.88 | 0.04 | 0.10 | |
| pw17 | CRS | image | 9.95 | 10.90 | 0.03 | 0.09 | |
| pw01 | LR | image | 13.94 | 14.95 | 0.02 | 0.05 | |

Every architecture and modality collapses on holdout; this is not a CRT-specific weakness.

**PD (trained subject-disjoint from the start) — selected rows:**

| ID | Arch | Input | AMAE | ARMSE | r²(agg) | r²(weighted) | Epochs |
|----|------|-------|------|-------|---------|---------------|--------|
| pd15 | CRT | BioZ | **9.07** | 9.76 | **0.32** | 0.19 | 65 |
| pd19 | CRS | BioZ | 9.30 | 10.02 | 0.10 | 0.13 | 418 |
| pd13 | CRT | image | 9.88 | 10.61 | 0.19 | 0.16 | 500 |
| pd09 | CNN | image | 8.67 | 9.42 | 0.28 | 0.14 | 52 (early-stopped) |
| pd01 | LR | image | 14.23 | 14.89 | 0.05 | 0.06 | 500 |

Training *for* subject-disjointness from the start (PD) partially closes the gap vs.
PW→holdout for the same architecture (pd15 AMAE 9.07 vs. pw15-on-holdout 9.85), but is
still far below PW/SS in-distribution — a ceiling around AMAE≈9 mmHg / r²≈0.3 for
population-disjoint protocols at this data volume, regardless of architecture (§3.5).

**SS (subject-specific) — selected rows:**

| ID | Arch | Input | AMAE | ARMSE | r²(agg) | r²(weighted) |
|----|------|-------|------|-------|---------|---------------|
| ss17 | CRS | image | **4.09** | 4.47 | 0.81 | **0.48** | ← best in any table |
| ss19 | CRS | BioZ | 4.14 | 4.53 | 0.80 | 0.50 |
| ss03 | LR | BioZ | 5.00 | 5.27 | 0.71 | 0.36 | ← §3.3: per-subject linear already competitive |
| ss15 | CRT | BioZ | 5.12 | 5.47 | 0.76 | 0.63 |
| ss13 | CRT | image | 6.02 | 6.33 | 0.71 | 0.65 |
| ss09 | CNN | image | **28.85** | 29.27 | 0.37 | 0.56 | ← §3.3/§3.5: correlation looks fine, AMAE is broken |
| ss11 | CNN | BioZ | 26.39 | 26.74 | 0.57 | 0.58 | ← same disconnect |

**Ablation (Extended Data Table 3)**: compressing PW15 (CRT, 7.5M params) to 2.12M params
*slightly improved* all metrics — extra depth was "redundant at this scale" (§3.5). CRT
(7.5M) and CRS (~2.5M) land within noise of each other on every protocol above.

**Robustness / longitudinal recalibration** (5 subjects, pilot study, progressive
fine-tuning over 5 days): feedforward models degrade severely without recalibration —
`SS03` (LR) ARMSE 12.3→33.51 mmHg (+21.21) over 5 days; hybrid transformer/Samba models are
far more stable — `SS15` (CRT) 12.21→13.42 mmHg (+1.21) over the same period. Even after
fine-tuning on the previous day's data, `SS03`/`SS07` (LR/MLP) fail to match `SS15`/`SS17`'s
*un-recalibrated* day-3 performance. This is the direct evidence behind promoting `samba`
to co-primary and excluding `cnn` as a core candidate (§3.5, §4).

**Label gap** (the paper's own distribution-shift diagnostic, model-independent): PW
holdout-set label gap ≈2.6x the typical PW train-test label gap; correlates with AMAE for
LR/MLP (r 0.44–0.61). Adopted directly into our own methodology (§3.3).

*(Full sweep: 20 PW + 20 PD + 20 SS model configs, all architecture/input/output
combinations — this table shows the decision-relevant subset; the complete tables are
preserved in the project's shared notes if a full reproduction is needed.)*

### 10.2 Reference: `pvi_ml` / `ml-experiments` (our separate, non-foundation codebase)

| Model | Split | Test cc_abs | Notes |
|-------|-------|-------------|-------|
| PW15 CRT (depth-1, cosine LR) | `local` (same subjects train+test) | **0.928** | Not a fair generalization baseline — cited only as an existing per-cohort ceiling reference, not a target foundation-transfer should match. |
| `abl-crtsin-bioz-to-waveform` (CRT) | `disjoint` (subject holdout) | ~0.46 peak, 0.21 final | The real motivating gap (§1.1). |
| Longitudinal full-finetune, subject002 | within-subject, d00→d01 | 0.586 @ ep998 | Full fine-tune (not frozen core), from `ml-experiments/_long`; useful upper-bound reference for what "more calibration capacity" can reach. |

### 10.3 Our foundation scaffold (MLP + `signal`, 1-channel input) — pipeline-validation only, not a benchmark

| Job | Task | Result |
|-----|------|--------|
| 424458 | Core S pretrain, 500 ep, disjoint | Best test cc_abs **0.445 @ epoch 40**; **0.274 @ epoch 499** (0.17 Pearson lost by originally exporting the last epoch instead of best — bug now fixed, `export_core.py` tries best-checkpoint first automatically). |
| 424459 | Core U SSL pretrain, 500 ep | Completed (~35 min); SSL loss → ~0; no BP signal monitored during training (§7.4). |

### 10.4 Our scaffold transfer to subject013 (frozen core, linear readout only, MLP scaffold) — pipeline-validation only

| Core | Test cc_abs | Notes |
|------|-------------|-------|
| Core S | **0.336** | Flat from epoch 0 — consistent with a linear-only readout having little room to move a frozen core's output (§7.5). |
| Core U | 0.305 | Peak ~epoch 449. |

### 10.5 Our scaffold Exp B budget curve, subject013 (job 424512) — pipeline-validation only

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
parameter count) explains the 4-min result. Despite being scaffold-only, this pattern
(cc_abs saturating while AMAE tracks budget) is consistent with §10.1's decoupling of
in-distribution shape-fitting from generalization/calibration — worth re-testing at
production scale, not just discarding because the underlying models were scaffolds.

### 10.6 Our production runs — status as of 2026-07-01 (blocked by §7.2/§7.3)

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
