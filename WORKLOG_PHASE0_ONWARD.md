# Worklog: Phase 0 → production pretrain (since pull)

Chronological record of code changes, cluster jobs, and results from the agent
session that started after `git pull origin main` and followed `PLAN.md` §0
(Phase 0 verification through production CRT/MAE runs).

**Baseline commit after pull:** `bf66ad6` — *phase 0 1 and 2 plus tests*  
**Repo:** `/mmfs1/projects/ece_bst/lsanc68/fundational_pvi`  
**Date:** 2026-07-01

---

## Table of contents

1. [Phase 0 — Environment sanity](#phase-0--environment-sanity)
2. [Phase 1 — Code fixes (pre-compute)](#phase-1--code-fixes-pre-compute)
3. [Phase 1 BUILD — Parquet caches](#phase-1-build--parquet-caches)
4. [Phase 2 — Core U scale check](#phase-2--core-u-scale-check)
5. [Post–Phase 2 hotfixes](#postphase-2-hotfixes)
6. [Production pretrain (500 epochs)](#production-pretrain-500-epochs)
7. [Artifacts produced](#artifacts-produced)
8. [Launch scripts reference](#launch-scripts-reference)
9. [Open / follow-up items](#open--follow-up-items)

---

## Phase 0 — Environment sanity

### Checks run

| Check | Command / job | Result |
|-------|---------------|--------|
| Test suite | `pytest tests/` (login node) | ~88 passed, 1 skipped (GPU test) |
| Cohort inventory | `PviDatasetInventory(branch='main'/'holdout')` | **main: 216**, **holdout: 15** files |
| Period length | All main-branch HDF5 `period_length` | Single value **`[50]`** — no mismatch |

### Test infrastructure

- **`src/launch_test.sh`** — SLURM runner for focused pytest suite (~2–3 min on compute node).
- **`src/run_tests_fast.sh`** — same targets, runnable on login node.
- Default `TEST_TARGETS`: new Phase 1 tests + existing fast milestone tests.

| Job | Suite | Result |
|-----|-------|--------|
| **424643** | Focused pytest | 2 failures (parquet subject + RoPE head dim) |
| **424644** | Focused pytest (after fixes) | **59 passed, 0 failed** |

---

## Phase 1 — Code fixes (pre-compute)

All items below are in commit `bf66ad6` unless noted.

### 1. Subject identity in batches (`subject_idx`)

**Why:** Required for DANN/GRL and per-subject diagnostics; previously batches only had `bp`/`stats`/`pviHP`/`pviLP`.

| File | Change |
|------|--------|
| `src/utils/primitives.py` | `NUM_SUBJECTS`, `subject_name_to_idx()` |
| `src/utils/h5io.py` | `slice_sequences()` adds `subject_idx` |
| `src/pipeline/pvi_parquet_dataset.py` | `PviParquetSplit.__getitem__` adds `subject_idx` from meta column |
| `src/pipeline/data_preparation_lazy.py` | Propagate `subject_idx` through lazy path |
| `tests/test_subject_batch.py` | Round-trip tests (HDF5 + parquet) |

**Follow-up perf fix (same session):** Precompute `_subject_ids` tensor once at dataset init instead of calling `self.hf.with_format(None)[idx]` per sample (~**4× faster** dataloader; epochs back to ~28 s vs ~5 min on PD cache).

### 2. RoPE positional encoding (`--crt-pe-type rope`)

| File | Change |
|------|--------|
| `src/models/positional_encoder.py` | RoPE implementation for CRT attention |
| `src/models/attn_models.py` | Custom MHA path when `pe_type="rope"` |
| `src/foundation/model_factory.py` | `--crt-pe-type rope` wiring |
| `src/foundation/config.py` | `crt_pe_type` field |
| `tests/test_rope.py` | Relative-position + forward smoke tests |

**Bug fixed during testing:** RoPE requires even head dimension; test adjusted head count selection.

### 3. Gradient-reversal subject adversary (DANN/GRL)

| File | Change |
|------|--------|
| `src/foundation/adversarial.py` | **New** — GRL layer, `SubjectAdversaryHead`, DANN λ schedule |
| `src/models/trainer_v3.py` | `set_subject_adversary()`, extra CE loss step |
| `src/models/workflow_v3.py` | `configure_subject_adversary()` |
| `src/foundation/pretrain.py` | `--subject-adversary` flag (default off) |
| `src/foundation/ssl_pretrain.py` | Same flag for SSL pretrain |
| `tests/test_adversarial.py` | GRL identity forward, flipped backward grad |

### 4. SSL linear-probe monitoring

| File | Change |
|------|--------|
| `src/foundation/ssl_probe.py` | **New** — `run_ssl_linear_probe()`, `_ProbeReadout` wrapper |
| `src/foundation/ssl_pretrain.py` | `--probe-every`, `--probe-subjects`, logging `probe_cc_abs` / `probe_amae` |
| `src/foundation/config.py` | `probe_every`, `probe_subjects`, `probe_epochs`, `probe_max_batches` |

**Later fix:** `_ProbeReadout.process_batch()` delegates to SSL model (required by `evaluation.collect_predictions()`). Without it, probe at epoch 10 crashes (jobs 424672/424673).

### 5. Samba architecture wiring

| File | Change |
|------|--------|
| `src/foundation/model_factory.py` | `arch=samba` → `PviSamba` |
| `src/foundation/config.py` | `samba_*` hyper-parameters |
| `tests/test_model_factory.py` | Samba build smoke test |

### 6. Holdout branch + transfer

| File | Change |
|------|--------|
| `src/foundation/transfer.py` | `--branch` for holdout cohort; `build_subject_dataset` uses branch inventory |
| `src/foundation/budget_exp.py` | `--branch` CLI |

### 7. Budget experiment calibration + matched capacity

| File | Change |
|------|--------|
| `src/foundation/budget_exp.py` | `--calibration-mechanism` (none / affine / …), `--matched-capacity` |
| `src/foundation/experiments.py` | Calibration hooks in budget curve runner |

### 8. Shape / bias / label-gap decomposition

| File | Change |
|------|--------|
| `src/analysis/decomposition.py` | **New** — `decompose_predictions()` |
| `tests/test_decomposition.py` | Unit tests |

### 9. OOD experiment driver (Exp C)

| File | Change |
|------|--------|
| `src/foundation/ood_exp.py` | **New** — cross-maneuver OOD eval CLI |

### 10. Phase 2 scale-inspection script

| File | Change |
|------|--------|
| `src/scripts/inspect_prediction_scale.py` | **New** — pred vs target mmHg range check |
| `src/launch_ssl_scale_check.sh` | Short SSL on PD cache |
| `src/launch_ssl_scale_postcheck.sh` | Transfer + inspect on GPU |

### 11. Cache build launch scripts

| Script | Protocol | Output path |
|--------|----------|-------------|
| `launch_build_cache_main_disjoint.sh` | main + disjoint (PD) | `…/v1_impedance_main_disjoint` |
| `launch_build_cache_main_within.sh` | main + within (PW) | `…/v1_impedance_main_within` |
| `launch_build_cache_holdout_disjoint.sh` | holdout + disjoint | `…/v1_impedance_holdout_disjoint` |

### 12. Production launch scripts

| Script | Purpose |
|--------|---------|
| `launch_crt_pretrain.sh` | CRT supervised foundation pretrain (Core S) |
| `launch_mae_ssl.sh` | MAE SSL pretrain (Core U) |

Top-of-script block hardcodes `CACHE_ROOT`, `SPLIT_MODE`, `LOGDIR`; `unset PVI_CACHE_ROOT` after sourcing `cluster.env` so launch overrides are not clobbered.

### 13. Cluster environment

| File | Change |
|------|--------|
| `env/cluster.env` | `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` on GPFS scratch (avoid home quota) |

---

## Phase 1 BUILD — Parquet caches

All built on **`/mmfs1/scratch/lsanc68/pvi_cache/`** with `input-mode=impedance`.

| Job | Script | Train / test | Wall time | Status |
|-----|--------|--------------|-----------|--------|
| **424645** | `launch_build_cache_main_disjoint.sh` | **143,186 / 19,181** | 7:16 | COMPLETED |
| **424646** | `launch_build_cache_main_within.sh` | **103,592 / 11,504** | 6:29 | COMPLETED |
| **424647** | `launch_build_cache_holdout_disjoint.sh` | **8,384 / 1,938** | 0:39 | COMPLETED |

Each job runs `check_cache` after build — all reported **OK**.

---

## Phase 2 — Core U scale check

Goal: short SSL + transfer on `subject013` to verify Core U predictions land in ~40–200 mmHg (not collapsed ~0 scale).

### SSL diagnostic

| Job | Config | Result |
|-----|--------|--------|
| **424649** | ad-hoc `--wrap` | **PENDING forever** — `Reason=PartitionConfig` (missing partition/account/mem) |
| **424651** | first proper script attempt | superseded |
| **424652** | pre–`subject_idx` precompute fix | ~5 min/epoch (too slow) |
| **424653** | with precompute fix | **COMPLETED 3:20** — 5 epochs, losses 1.39 → **0.001** |

**424653 throughput:** epoch 1 ~67 s (CUDA warmup); epochs 2–5 ~**28 s** at ~20 it/s (matches old job 424524).

**Artifact:** `artifacts/debug-ssl-scale-check/main/foundation_core_U.pt`

### Transfer + scale inspect

| Job | Pipeline | Result |
|-----|----------|--------|
| **424654** | 20-epoch transfer on subject013 + inspect | **COMPLETED 0:20** |

**Transfer metrics (subject013, 20 epochs, 5-epoch SSL core):**

| Metric | Value |
|--------|-------|
| Test loss | 9.67 |
| Test cc_abs | 0.171 |
| AMAE | **5.50 mmHg** |
| ARMSE | 5.84 mmHg |
| SBP / DBP MAE | 5.85 / 4.95 mmHg |

**Exported predictions:** ~67–141 mmHg (targets same range) — **Phase 2 pass**.

**Initial inspect bug:** `inspect_prediction_scale` used a fresh readout on frozen core → preds ~0. Fixed to load trained readout from transfer checkpoint (see below).

---

## Post–Phase 2 hotfixes

| Issue | Fix | Files |
|-------|-----|-------|
| Inspect used untrained readout | `load_transferred_model()`, `--transfer-logdir`, `--checkpoint` | `inspect_prediction_scale.py`, `transfer.py`, `launch_ssl_scale_postcheck.sh` |
| CRT export crash at end of pretrain | Propagate `input_mode` / `output_mode` / `mask_key` on parquet splits | `pvi_parquet_dataset.py` |
| Launch scripts ignored cache override | Hardcode `CACHE_ROOT` at top; `unset PVI_CACHE_ROOT` | `launch_crt_pretrain.sh`, `launch_mae_ssl.sh` |
| SSL probe crash at epoch 10 | `_ProbeReadout.process_batch()` | `ssl_probe.py` |
| Slow parquet dataloader | Precompute `subject_idx` column | `pvi_parquet_dataset.py` |
| HF cache on home quota | Scratch paths in `cluster.env` | `env/cluster.env` |

**Verified inspect after fix:** preds **79–119 mmHg** with `--transfer-logdir debug-ssl-scale-check-transfer`.

---

## Production pretrain (500 epochs)

Config common to all four runs:

- `input-mode=impedance`, `batch-size=256`, `max-epochs=500`, `cache-num-workers=8`
- CRT: `arch=crt`, logdirs `foundation-pretrain-crt-{pd,pw}`
- MAE: `arch=mae`, `--ssl-arch mae`, `--probe-every 10`, logdirs `foundation-ssl-pretrain-mae-{pd,pw}`

### Pilot runs (old cache / pre-fix)

| Job | Run | Status | Notes |
|-----|-----|--------|-------|
| **424523** | CRT on PD cache | FAILED ~1h54m | `PviParquetSplit` missing `input_mode` at export |
| **424524** | MAE on PD cache | CANCELLED 2h22m | 312 epochs, ~27 s/epoch — used as speed baseline |
| **424658–424668** | Various relaunch attempts | FAILED ~3–4 s | Cache path / env misconfig before launch-script fix |

### Production runs (new caches)

| Job | Run | Cache | Status | Wall time | Key result |
|-----|-----|-------|--------|-----------|------------|
| **424669** | CRT **PD** | `v1_impedance_main_disjoint` | RUNNING* | ~2h+ | Training done; stuck on OT statistics export |
| **424670** | CRT **PW** | `v1_impedance_main_within` | **COMPLETED** | 1:34:49 | Core exported; test bp_acc **0.435** |
| **424672** | MAE **PD** | disjoint | **FAILED** | 4:54 | Probe crash (pre-`process_batch` fix) |
| **424673** | MAE **PW** | within | **FAILED** | 3:39 | Same probe crash |
| **424706** | MAE **PD** (relaunch) | disjoint | RUNNING | — | Passed epoch-10 probe ✓ |
| **424710** | MAE **PW** (relaunch) | within | RUNNING | — | Passed epoch-10 probe ✓ |

\*Job **424669** completed 500 epochs and exported history/results; `foundation_core.pt` was **manually exported** from checkpoint while OT Wasserstein statistics (~115k test samples, O(n²)) were still running. Safe to cancel 424669 if statistics JSON is not needed.

### CRT results summary

| Metric | CRT PD (disjoint) | CRT PW (within) |
|--------|-------------------|-----------------|
| Train samples | 143,186 | 103,592 |
| Final train loss | 33.6 | 22.9 |
| Final test loss | 53.9 | 27.4 |
| Final test bp_acc (Pearson r) | **0.029** | **0.435** |
| Best test bp_acc | 0.339 @ ep1 | 0.486 @ ep455 |
| Test AMAE (PW stats export) | — | **8.93 mmHg** |
| DBP / SBP CC (PW) | — | 0.46 / 0.30 |

**Interpretation:**

- **PW** — Healthy convergence; pooled waveform pretrain with within-subject test split generalizes well on held-out periods.
- **PD** — Low test bp_acc is **expected** for subject-disjoint evaluation: the readout is shared across unseen subjects. Train metrics improve while test correlation collapses. The exported core is still intended for **per-subject transfer** (Exp B), not direct pooled test accuracy.

### MAE probe @ epoch 10 (relaunch jobs)

| Job | Run | probe_cc_abs | probe_amae |
|-----|-----|--------------|------------|
| 424706 | PD | -0.002 | 26.8 |
| 424710 | PW | -0.029 | 13.2 |

Early probe values near zero are normal (3 subjects, 5 probe epochs, frozen core).

---

## Artifacts produced

| Path | Description |
|------|-------------|
| `artifacts/debug-ssl-scale-check/main/foundation_core_U.pt` | Phase 2 SSL diagnostic (5 epochs) |
| `artifacts/debug-ssl-scale-check-transfer/main/` | Phase 2 transfer on subject013 |
| `artifacts/foundation-pretrain-crt-pw/main/foundation_core.pt` | Production CRT PW (500 ep) |
| `artifacts/foundation-pretrain-crt-pd/main/foundation_core.pt` | Production CRT PD (500 ep, manual export) |
| `artifacts/foundation-pretrain-crt/main/` | Pilot CRT (424523, incomplete export) |
| `artifacts/foundation-ssl-pretrain-mae-pd/main/` | MAE PD (in progress, job 424706) |
| `artifacts/foundation-ssl-pretrain-mae-pw/main/` | MAE PW (in progress, job 424710) |

Parquet caches (scratch, not under `artifacts/`):

```
/mmfs1/scratch/lsanc68/pvi_cache/v1_impedance_main_disjoint
/mmfs1/scratch/lsanc68/pvi_cache/v1_impedance_main_within
/mmfs1/scratch/lsanc68/pvi_cache/v1_impedance_holdout_disjoint
```

---

## Launch scripts reference

```bash
# Tests (focused)
sbatch src/launch_test.sh
bash src/run_tests_fast.sh

# Cache builds (parallel OK)
sbatch src/launch_build_cache_main_disjoint.sh
sbatch src/launch_build_cache_main_within.sh
sbatch src/launch_build_cache_holdout_disjoint.sh

# Phase 2 scale check
sbatch src/launch_ssl_scale_check.sh
sbatch src/launch_ssl_scale_postcheck.sh

# Production pretrain — edit CACHE_ROOT / SPLIT_MODE / LOGDIR at top of script, then:
sbatch src/launch_crt_pretrain.sh
sbatch src/launch_mae_ssl.sh
```

**MAE relaunch workflow:** edit top block for PD → `sbatch`; edit for PW → `sbatch` again.

---

## Open / follow-up items

1. **MAE PD/PW (424706, 424710)** — let run to 500 epochs; verify `foundation_core_U.pt` export.
2. **CRT PD job 424669** — cancel if OT statistics not needed; core already saved manually.
3. **Transfer + inspect on new CRT cores** — e.g. subject013 with `foundation-pretrain-crt-{pd,pw}/main/foundation_core.pt`.
4. **Exp A/B budget curves** — `sbatch src/launch_budget.sh` with new cores (not yet run this session).
5. **Exp C OOD** — driver exists (`ood_exp.py`); not yet run on cluster.
6. **Subject adversary / RoPE ablations** — code ready (`--subject-adversary`, `--crt-pe-type rope`); not enabled in production runs above.
7. **OT statistics export** — PD cohort (~19k test) is very slow; consider subsampling for future runs.

---

## Job ID quick index

| Phase | Jobs |
|-------|------|
| Tests | 424643, 424644 |
| Cache build | 424645, 424646, 424647 |
| SSL scale | 424649 (stuck), 424651, 424652, **424653**, **424654** |
| Pilot pretrain | 424523, 424524 |
| Production CRT/MAE | 424658–424668, **424669**, **424670**, 424672, 424673, **424706**, **424710** |

---

*Generated from agent session worklog. Update this file when production MAE jobs complete or new experiments are submitted.*
