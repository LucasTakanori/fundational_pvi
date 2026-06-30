# Plan: Foundation model & digital twin of EIT-ring hemodynamics (adapting Wang et al. 2025)

## Context

Adapt the methodology of Wang et al., *Foundation model of neural activity predicts response to new stimulus types*, Nature 2025 (s41586-025-08829-y) — the MICrONS mouse-visual-cortex foundation model — to our wearable electrical-impedance-tomography (EIT) ring data, reusing the existing pvi-ml infrastructure (`/home/srl/pvi-ml`; data `/mnt/d/datasets/masked/main`).

The paper is not cardiovascular; "replication" = transferring its mechanism — a shared foundation core trained across many subjects, frozen/adapted to new subjects with a small readout, that is (a) far more data-efficient than per-subject models and (b) generalizes out-of-distribution to new stimulus domains. Mapping:

| Wang et al. (mouse V1) | This work (EIT ring) |
| --- | --- |
| Mouse (subject) | Human subject (subject001…, 91 available) |
| Natural-video stimulus (input) | PVI ring signal (32 resistance + 32 reactance channels, HP/LP; or 40×40 EIT image) |
| Neural activity (measured response = target) | Continuous arterial BP (mmHg, independently measured) |
| Behavioural covariates → modulation | Per-cycle stats (duration, tMax) injected pre-readout (already in repo) |
| Perspective module (ray tracing) | Dropped — no geometric analog |
| Core (3D-conv + Conv-LSTM), shared, frozen | Shared foundation core (conv/transformer/SSM body) |
| Readout (per-neuron, spatial position) | Per-subject BP readout head (+ auxiliary heads) |
| Stimulus domains (videos → Gabor/dots/noise) | Maneuvers: baseline → valsalva / pressor |
| Foundation cohort (8 mice) → new mice (4) | Cohort subjects → held-out subjects |
| CC_norm (noise-ceiling normalized) | CC_abs (Pearson r) + MAE/RMSE + BHS/AAMI (see Deviations) |
| Poisson NLL loss | MSE / MorphologyLoss (continuous BP) |

**Goals (Nature-defensible contributions):** (1) first cross-subject foundation model / digital twin of wearable bioimpedance hemodynamics; (2) SSL pretraining → few-shot BP calibration for new users (foundation ≫ individual at low data) — the central unsolved problem in cuffless BP; (3) OOD across maneuvers (train resting → predict valsalva/pressor); (4) interpretability + digital twin (latent organization, in-silico perturbation, learned EIT representation); (5) methodological comparisons (SSL vs supervised cohort core; raw channels vs Newton vs learned reconstruction).

## Decisions (locked with user)

- SSL pretext = BOTH masked reconstruction and causal forecasting (multi-objective).
- Run BOTH cores and compare: SSL core (U) vs supervised cohort core (S).
- Build a learned EIT reconstruction as its own sub-track (beyond raw-channels + Newton baselines).
- Multi-head readout: BP (primary) + auxiliary decoders (maneuver/state, HR).

## Verified data facts (inspected subject001_{baseline,valsalva,pressor})

- Ground truth present in all 3 maneuvers. `data/bp/signal` = continuous BP, mmHg. baseline SBP≈109/DBP≈59; valsalva swings (DBP→35, SBP→144); pressor sustained (SBP 130±14). OOD is measurable.
- Cardiac-cycle-normalized time: each period = exactly `period_length`=50 frames (≈40–50 Hz; cycle ≈1.0–1.2 s); sessions ≈9 min; `num_periods` ≈440–870/session. Absolute timing/HR lives in stats.
- Ring native output = `data/{pviHP,pviLP}/{resistance,reactance}` each (32,T); signal (1,T); img (40,40,T) = 1-step-Newton EIT reconstruction, fixed circular FOV (228/1600 px NaN, constant over time).
- Masks (`masks/mask{01,05,10,15}`, 1-based inclusive period indices) = clean QC sequences; SQI ≈80%.
- Availability: 91 baseline, 74 valsalva, 51 pressor (216 files); ~51 subjects have all three; ≈30 h impedance total — ample for SSL.

---

## Approach

### Foundation pretraining — run BOTH, compare (U vs S)

- **(U) SSL core (primary):** pretrain on all 216 sessions with a multi-objective pretext = masked reconstruction (MAE/BERT: hide channel×time patches) + causal forecasting (predict next window). Combined loss `L = λ_m·maskedMSE + λ_f·forecastMSE`. Then attach readouts (frozen linear-probe or finetune).
- **(S) Supervised cohort core (paper-faithful):** train core+readout end-to-end to predict BP across the cohort with `split_mode='disjoint'`, freeze core, transfer.
- Both are first-class deliverables; head-to-head on the same transfer benchmark (Exp G).

### Input representation & EIT reconstruction sub-track (committed)

- **(a) Raw 64 channels** (resistance+reactance, HP[+LP]) — true sensor data, avoids the lossy linearized image. Primary input.
- **(b) Newton image** — existing 40×40 1-step-Newton img — baseline.
- **(c) Learned EIT reconstruction** (new `models/eit_recon.py`) — channels→conductivity-image decoder, trained supervised (vs a finer iterative recon) and/or physics-regularized via a differentiable EIT forward operator (data-consistency on boundary voltages), and/or jointly with the SSL/BP objective (recon that is maximally predictive of dynamics). Yields a richer input and an interpretability artifact (what the ring "sees" spatially). Requires ring geometry/forward model (see Remaining inputs).

### Architecture candidates (try a few, pick best on transfer)

1. **PviCNNTransformer** (existing) — conv + LSTM positional encoder + transformer ≈ paper's conv+recurrent core.
2. **Masked spatiotemporal Transformer** (MAE-style, new) — tokenize 64 channels (or image patches) × time; strongest representation + interpretable attention.
3. **PviSamba/S4-Mamba** (existing WIP) — long-context SSM for high-rate raw sequences.
4. **Paper-faithful core** (new) — 3D-conv DenseNet + Conv-LSTM, replicating Wang et al.'s actual core: feedforward DenseNet of causal (time-shifted) 3D spatiotemporal convolutions (3 blocks, dense connections, GeLU/ELU, spatial pooling) → Conv-LSTM recurrent cells (2D spatial-conv gates; hidden maps concatenated → core output `H_t ∈ R^{C×H×W}`); optional CvT-LSTM attention variant. Pairs with the 40×40 EIT image input (direct analog of their video frames) and instantiates the paper's full module stack minus perspective:
   - **Modulation:** small LSTM over stats (duration/tMax → HR/timing) → modulation maps concatenated into the Conv-LSTM input (faithful modulation analog).
   - **Readout:** per-output bilinear interpolation at a learned spatial position on the core feature map → BP (mirrors their per-neuron spatial readout; learned positions are interpretable = "where on the ring BP is read from").

   The most literal "same architecture as them" candidate and the primary vehicle for the digital-twin/interpretability analyses (Exp D/E).

### Readout heads (multi-task, on the shared core)

- **Primary:** BP — waveform (50-dim) and fiducials (SBP/DBP).
- **Auxiliary:** (i) maneuver/state classifier (baseline/valsalva/pressor) — probes hemodynamic-state encoding; (ii) HR / cycle-timing decoder from stats. Light heads, trained jointly or as linear probes; power Exp D/E. Mirrors the paper's multi-head readout/ensemble.

---

## Code changes (reuse existing infra; well-contained)

1. **Core/Readout split + freeze-and-transfer + multi-head API (central new piece)**
   - Split already exists physically: `PviCNN`→`conv_layers`(core)+`fc*`(readout); `PviCNNTransformer`→conv+projection+rrpe+transformer(core)+mlp(readout) (`models/cnn_models.py`, `models/attn_models.py`).
   - Refactor `BasePviLearner` (`models/base_model.py`): expose `self.core`/`self.readout` (+ optional aux heads) and `forward_core()`/`forward_readout()`; keep `forward()` = composition (backward-compatible; assert numerical equality in tests).
   - Transfer utility: load checkpoint (`TrainingCheckpoint` saves full `model.state_dict()`, `models/tracking.py`), copy `core.*` via `load_state_dict(strict=False)`, `model.core.requires_grad_(False)`, optimizer over `filter(lambda p: p.requires_grad, model.parameters())`.

2. **SSL pretext + dataset (new; both objectives)**
   - `PviSSLDataset` over `PviConfiguredDataset.__getitem__`/`h5io.slice_sequences`: per sample emit a masked view (random channel×time patches; reuse constant circular-FOV mask for images) and a past→future split. Two lightweight pretext heads (recon + forecast), discarded after pretraining; core kept. Combined masked-MSE + forecast-MSE. Monitor recon/forecast error + periodic linear-probe BP correlation.

   **2b. Learned EIT reconstruction (new sub-track)** — `models/eit_recon.py` (channels→image; optional differentiable forward operator for data-consistency; standalone or joint with §2).

   **2c. Multi-task readout heads (new)** — BP (waveform, fiducials) + maneuver + HR heads via the §1 API.

3. **Training-data-budget control (gap; for Fig 2/3 curves)** — `set_train_budget(minutes|n_seq, seed)`: subsample `train_mask` (→ minutes via stats/duration), rebuild with existing `_get_subsets_from_split` + `set_dataloaders` (`pipeline/_data_preparation.py`).

4. **OOD-across-maneuver eval (cleanest: separate datasets, zero leakage)** — train on baseline-only dataset; evaluate with `ModelEvaluator` (`models/trainer_v3.py`) on a separate valsalva/pressor dataset; + within/reverse controls.

5. **Metrics/plots (reuse)** — primary CC_abs=`bp_accuracy` (`models/perf_metrics.py`); secondary MAE/RMSE (`metrics_waveform`) + BHS tolerances/concordance (`metrics_fiducial`); add aggregation/plot script in `analysis/` for foundation-vs-individual curves.

6. **Experiment drivers** — adapt `scripts/train_population_v2.py` (cohort/SSL pretrain via `PviLazyDataset`+`PviBatchSampler`) and `scripts/train_subjects_v2.py` (individual + transfer readout).

---

## Experiment matrix (maps to paper figures)

- **Exp A — Fig 2 (data-efficiency, individual).** Per held-out subject, train end-to-end at budgets {≈4,8,16,…} min; CC_abs vs minutes; average over subjects/seeds.
- **Exp B — Fig 3b (foundation transfer).** Cohort cores (U & S); per held-out subject fit only readout at each budget; foundation vs individual curves (few-shot BP calibration).
- **Exp C — Fig 3c–g (OOD across maneuvers).** Core trained on baseline; readout fit on baseline; test on valsalva & pressor + controls.
- **Exp D — Fig 4 analog (in-silico digital twin).** Perturb impedance input / roll out forecasts; measure BP response & channel/feature sensitivity → physiological interpretability.
- **Exp E — Fig 5 analog (functional barcode + latent structure).** Predict subject/maneuver from readout weights (logistic regression); UMAP of core latents by maneuver/HR/BP (≈ Extended Data Fig 2); aux-decoder accuracy as a state probe.
- **Exp F — EIT reconstruction.** Input variants (a) raw vs (b) Newton vs (c) learned on BP/transfer accuracy; recon maps; physics data-consistency error.
- **Exp G — U vs S.** SSL core vs supervised cohort core across Exp B/C (data-efficiency & OOD deltas).
- **(Optional) Ablations:** drop modulation(stats)/diff/recurrence; architecture comparison.

---

## Compute plan (2× H100 NVL 94 GB, ≤10 days/job)

Phased: prototype on 64-channel input (SSL pretrain in hours) → scale to images if warranted. DDP across both GPUs for cohort/SSL pretraining; per-subject transfer/individual runs are small and parallelizable. 10-day budget is ample for the full matrix + multiple architectures.

## Milestones

1. Core/readout refactor (`forward_core`/`forward_readout`) + freeze/transfer utility + multi-head API + 1-subject smoke test.
2. Train-budget control + metrics/plot aggregation.
3. SSL dataset + dual pretext (masked + forecast) + cohort SSL pretrain → core U (architectures: candidates 1–4, incl. the paper-faithful 3D-conv DenseNet + Conv-LSTM core on the image input).
4. Supervised cohort core S baseline (disjoint).
5. Exp A/B (data-efficiency, foundation vs individual) + Exp G (U vs S).
6. Exp C (OOD maneuvers).
7. Aux decoders + Exp D/E (digital twin, interpretability, functional barcode).
8. Learned EIT reconstruction + Exp F; scale best config to images; finalize figures.

---

## Deviations from the paper (state when defending)

- No noise ceiling (no repeated identical stimuli) → CC_norm not reproducible; use CC_abs + clinical error (biggest metric deviation).
- Low-dimensional readout (BP = 50 samples/2 scalars vs thousands of neurons): system-level transfer maps perfectly; per-neuron richness does not (Fig 4/5 are adapted analogs, supported by aux decoders + latent probing).
- No perspective module (the one paper module with no analog); we do include a paper-faithful 3D-conv DenseNet + Conv-LSTM core + modulation LSTM + spatial bilinear readout (candidate 4); MSE not Poisson (correct for continuous BP).

## Risks

- Cycle-normalization → model sees cardiac phase, not absolute time (good for morphology; feed HR/timing via stats).
- BP ground-truth quality under maneuvers (motion artifact) → rely on QC masks; report SQI.
- baseline→pressor shift may exceed a frozen core → may need partial finetune (report both).
- Learned reconstruction depends on EIT geometry/forward model availability.

## Verification

- Smoke: build `PviCompositeDataset` for one subject (img + channels), check shapes, run 1 epoch of `TrainingWorkflow`; assert `forward_core∘forward_readout == original forward` numerically.
- Transfer correctness: assert core params frozen and unchanged after readout training; only readout updates.
- No leakage: assert disjoint cohort vs held-out; OOD eval shares no periods with train.
- End-to-end: reproduce one Exp-B curve on a small cohort; sanity-check foundation ≥ individual at low budget; report CC_abs + MAE/BHS.

## Remaining inputs to confirm early in implementation

- EIT geometry/forward model (electrode positions, drive pattern) for §2b physics regularization; else fall back to a data-driven decoder supervised against a finer iterative Newton recon.
- Dedicated pvi-ml Python env (torch+h5py+ot+sklearn, CUDA) on the H100 server (sibling speech-tongue-cancer/.venv used only for offline inspection).
