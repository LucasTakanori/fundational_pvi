# Experiment runbook

Operational plan for running the foundation-model experiment matrix on the
`ece_bst` cluster. Research context and paper mapping live in [`PLAN.md`](PLAN.md).

Paths are set in [`env/cluster.env`](env/cluster.env):

```bash
export PVIPROJECT_ROOT="/mmfs1/projects/ece_bst/lsanc68/fundational_pvi"
export PVI_DATA_ROOT="/home/lsanc68/ece_bst_link/common/data/pvi_data"
```

Data: 216 masked HDF5 sessions under `$PVI_DATA_ROOT/main/` (~91 subjects;
~51 with baseline + valsalva + pressor).

---

## What `src/launch.sh` runs (default)

Unless you override `JOB_SCRIPT`, `sbatch src/launch.sh` submits:

| Setting | Value |
|--------|--------|
| Resources | 1 GPU, 16 CPUs, 250 GB RAM, 1 node, **10 days** |
| Partition / account | `ece_bst` |
| Data | All sessions under `$PVI_DATA_ROOT/main/` |
| Command | `python -m src.foundation.pretrain --input-mode signal --output-mode waveform --max-epochs 500` |

This is **supervised cohort pretraining (core S)**: pooled population training
with a shared readout, predicting BP waveforms from the 1-D PVI signal, up to
500 epochs with early stopping.

**Outputs:** `$PVIPROJECT_ROOT/artifacts/foundation-pretrain/main/` (checkpoints,
logs, and `foundation_core.pt` for later transfer).

**Start here:**

```bash
cd /mmfs1/projects/ece_bst/lsanc68/fundational_pvi
uv sync                    # once, if .venv/ not created yet
sbatch src/launch.sh
```

Override the command for other jobs:

```bash
JOB_SCRIPT="python -m src.foundation.ssl_pretrain --input-mode signal --max-epochs 500" sbatch src/launch.sh
```

---

## Run order

Later experiments depend on artifacts from earlier phases.

### Phase 0 — Prerequisites (one-time)

| # | What | Purpose | Entry point |
|---|------|---------|-------------|
| P0a | **Core S** (supervised cohort) | Shared backbone trained on all subjects with BP labels | `python -m src.foundation.pretrain` ← *default `launch.sh`* |
| P0b | **Core U** (SSL cohort) | Same backbone via masked recon + forecasting (no BP labels) | `python -m src.foundation.ssl_pretrain` |
| P0c | **Architecture sweep** (optional) | Pick best core among `cnn`, `crt`, `mae`, `samba`, `dnclstm` | Same pretrain scripts + model / `--input-mode` config |

**Input representation tracks** (can branch across experiments):

| Track | Description | Flag / module |
|-------|-------------|---------------|
| **(a)** Raw 64 channels | `resistance` + `reactance`, HP/LP — primary | default signal modes |
| **(b)** Newton 40×40 image | Existing 1-step-Newton recon | `--input-mode image` |
| **(c)** Learned EIT recon | Channels → conductivity image | `src/models/eit_recon.py` (Exp F) |

---

### Phase 1 — Main experiment matrix

Maps to [`PLAN.md`](PLAN.md) §98–107.

| Exp | Paper analog | Question | How to run | Depends on |
|-----|--------------|----------|------------|------------|
| **A** | Fig 2 | How fast does a **per-subject** model learn? | Per held-out subject: end-to-end train at budgets {≈4, 8, 16, …} min; plot CC_abs vs minutes | `run_budget_curve` (`src/foundation/experiments.py`); individual baseline |
| **B** | Fig 3b | Does **foundation transfer** beat individual at low data? | Per held-out subject: freeze core, fit readout only at each budget; foundation vs individual curves | Core S and/or U (P0) |
| **C** | Fig 3c–g | **OOD across maneuvers** | Train on **baseline only**; test on valsalva & pressor (+ within/reverse controls) | `evaluate_ood` (`src/foundation/evaluation.py`) |
| **D** | Fig 4 | **Digital twin** — in-silico perturbation | Perturb impedance input, roll out forecasts, channel sensitivity | Trained core + `src/analysis/interpretability.py` |
| **E** | Fig 5 | **Functional barcode + latent structure** | Predict subject/maneuver from readout weights; UMAP of core latents; aux-decoder probes | `src/foundation/multitask.py` + interpretability |
| **F** | — | **EIT reconstruction** | Compare raw vs Newton vs learned recon on BP/transfer accuracy | `src/models/eit_recon.py` |
| **G** | — | **U vs S head-to-head** | Same as B/C but compare SSL core vs supervised core | Both cores (P0a + P0b) |

**Optional ablations:** drop stats modulation, `diff`, recurrence; compare architectures.

**Primary metrics:** CC_abs (`bp_accuracy`), MAE/RMSE (`metrics_waveform`), BHS/AAMI (`metrics_fiducial`).

---

### Phase 2 — Sub-tracks

| Track | Description | Tied to |
|-------|-------------|---------|
| Multi-task readout | BP (waveform + SBP/DBP) + maneuver classifier + HR decoder | Exp D, E |
| Learned EIT recon | Channels → image; optional physics data-consistency | Exp F |
| Budget curves | `set_train_budget` + `src/analysis/budget_curves.py` | Exp A, B, G |

---

## Suggested `sbatch` commands

```bash
# 1. Core S — supervised cohort pretrain (START HERE)
sbatch src/launch.sh

# 2. Core U — SSL pretrain
sbatch src/launch_ssl.sh

# 3. Transfer to one held-out subject (repeat per subject)
JOB_SCRIPT="python -m src.foundation.transfer --subject subject013 \
  --core artifacts/foundation-pretrain/main/foundation_core.pt" \
  sbatch src/launch.sh

# 4. Exp A / B / G — budget sweeps (per subject; driver: experiments.run_budget_curve)
# 5. Exp C — OOD eval (baseline train → valsalva/pressor test via evaluate_ood)
# 6. Exp D / E — interpretability.py after models exist
# 7. Exp F — eit_recon training + input-mode comparison
```

---

## Milestones vs code

| Milestone | Repo status | Experiments unlocked |
|-----------|-------------|----------------------|
| 1. Core/readout refactor + transfer | `src/foundation/`, smoke tests | P0, transfer |
| 2. Train-budget + metrics/plots | `set_train_budget`, `budget_curves.py` | A, B |
| 3. SSL dual pretext + core U | `ssl_pretrain.py`, `ssl.py` | G (U) |
| 4. Supervised cohort core S | `pretrain.py` | G (S), B |
| 5. Exp A/B + G | `experiments.run_budget_curve` | Main figures |
| 6. Exp C (OOD maneuvers) | `evaluate_ood` | Maneuver generalization |
| 7. Aux decoders + Exp D/E | `multitask.py`, `interpretability.py` | Digital twin / barcode |
| 8. Learned EIT + Exp F | `eit_recon.py` | Reconstruction sub-track |

---

## Scale and job sizing

- **Exp A/B/G** multiply quickly: subjects × budgets × seeds × {individual, foundation-S, foundation-U}.
- Default wall time is **10 days** (`#SBATCH --time=10-00:00:00`), matching the PLAN.md compute budget and enough for long cohort pretrains at ~2 h/epoch.
- Per-subject transfer and individual runs are small and highly parallelizable.

---

## Quick reference — module entry points

```bash
source .venv/bin/activate
source env/cluster.env

# Smoke test (no real data)
python -m pytest tests/test_smoke.py -q

# Core S (supervised)
python -m src.foundation.pretrain --input-mode signal --output-mode waveform --max-epochs 500

# Core U (SSL)
python -m src.foundation.ssl_pretrain --input-mode signal --max-epochs 500

# Transfer
python -m src.foundation.transfer --subject subject013 \
  --core artifacts/foundation-pretrain/main/foundation_core.pt
```
