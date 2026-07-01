# PVI Foundation — run log

Operational history for **fundational_pvi** cluster work.  
Paths: `env/cluster.env` (`PVI_DATA_ROOT`, `PVI_CACHE_ROOT`, `PVIPROJECT_ROOT`).

---

## Phase 0 — Environment & data pipeline

| Item | Status | Notes |
|------|--------|-------|
| uv project + `.venv` | Done | `pyproject.toml`, `uv sync` |
| Parquet cache builder | Done | `src/pipeline/pvi_cache.py`, `build_pvi_cache.py` |
| `PviParquetCohort` fast training | Done | Wired into pretrain/ssl when `PVI_CACHE_ROOT` valid |
| Cache location | Done | `PVI_CACHE_ROOT=/mmfs1/scratch/lsanc68/pvi_cache/v1` (GPFS scratch) |
| Disjoint-split mask bug | Fixed | Copy mask lists in `data_preparation_lazy.py` (no aliasing) |

---

## Phase 0 — Scaffold pretrain (MLP + `signal`)

These runs validated **pipeline wiring only** (not production models).  
`cc_abs` = mean Pearson r(SBP, DBP); ceiling ~0.35–0.45 on scaffold.

| Job | Task | Result | Artifacts |
|-----|------|--------|-----------|
| 424458 | Core S pretrain (MLP, signal, 500 ep) | Trained; export failed initially | `artifacts/foundation-pretrain/main/` |
| — | Manual core export | Done | `foundation_core.pt` (epoch 490, acc ~0.27) |
| 424459 | Core U SSL (MLP, signal) | **COMPLETED** ~35 min | `artifacts/foundation-ssl-pretrain/main/foundation_core_U.pt` |

**Best scaffold Core S metric:** test Pearson **0.45 @ epoch 40** (best checkpoint not exported until later fix).

---

## Transfer & experiment jobs (subject013)

### Bugs fixed before transfer worked

1. **`KeyError: 'cluster_size'`** — double `get_dataloaders()` stripped `stratified`; fixed with `.get('stratified', False)` in `data_preparation_lazy.py`.
2. **`pin_memory` + CUDA tensors** — `trainer_v3.py` skips `dataset.to(cuda)` for `PviLazyDataset`.
3. **Artifact overwrite** — same logdir for Core S/U; added `--core-tag` → `foundation-transfer-{subject}-{tag}`.

### Transfer runs (full readout, 500 epochs, MLP cores)

| Job | Core | Test Pearson (final) | Time |
|-----|------|----------------------|------|
| 424509 | Core S (`foundation_core.pt`) | **0.336** | ~2 min |
| 424510 | Core U (`foundation_core_U.pt`) | 0.305 | ~2 min |

On-disk artifacts = **Core U only** (overwrite). Core S metrics in `foundation-transfer_424509.out`.

### Exp B — budget curve (subject013)

| Job | Description | Result |
|-----|-------------|--------|
| 424512 | individual + foundation_S + foundation_U @ {4,8,16,32,64} min | **COMPLETED** ~20 s |

Output: `artifacts/exp-b-subject013/main/budget_records.json`

**Peek @ 4 min budget (`cc_abs`):** foundation_S **0.336** > foundation_U 0.326 > individual **-0.12**  
(Negative Pearson = anti-correlated SBP/DBP ordering at tiny data; see `bp_accuracy` in `perf_metrics.py`.)

---

## Phase 1 — Production model wiring (code, Jun 2025)

Stopped scaling the MLP scaffold. Implemented real architectures per `MODELS.md`:

| Component | Change |
|-----------|--------|
| `src/foundation/arch.py` | Arch tag normalization (`crt`, `mae`, aliases) |
| `src/foundation/model_factory.py` | `build_foundation_model()`, `build_ssl_model()` |
| `src/foundation/foundation_model.py` | Multi-arch: **mlp**, **crt**, **mae**, **cnn** + `SubjectReadout` heads |
| `src/foundation/ssl.py` | SSL encoders: **mlp** (scaffold) or **mae** (production) |
| `pretrain.py` | Uses factory; saves `foundation_core_meta.json` (arch, input_mode, shapes) |
| `ssl_pretrain.py` | Uses factory; saves `foundation_core_U_meta.json` |
| `transfer.py` | Factory + auto-load arch from `*_meta.json`; `--core-tag` |
| `budget_exp.py` | Factory; default `--arch crt` |
| `tests/test_model_factory.py` | crt/mae forward + SSL→foundation core load |

### New launch scripts

| Script | Purpose |
|--------|---------|
| `src/launch_transfer.sh` | Per-subject transfer (unsets cache) |
| `src/launch_budget.sh` | Exp B budget sweeps |
| `src/launch_crt_pretrain.sh` | **Production Core S**: `--arch crt --input-mode impedance` |
| `src/launch_mae_ssl.sh` | **Production Core U**: `--arch mae --input-mode impedance` |
| `src/launch_test.sh` | Pytest via SLURM (not on login node) |

### Production jobs to run next

```bash
# 1. Rebuild cache for impedance (first CRT/MAE run, or set new PVI_CACHE_ROOT)
sbatch src/launch_crt_pretrain.sh
# LOGDIR=foundation-pretrain-crt  → artifacts/foundation-pretrain-crt/main/

sbatch src/launch_mae_ssl.sh
# LOGDIR=foundation-ssl-pretrain-mae → artifacts/foundation-ssl-pretrain-mae/main/

# 2. After cores exist — transfer with auto arch from meta
CORE=artifacts/foundation-pretrain-crt/main/foundation_core.pt \
  sbatch src/launch_transfer.sh

CORE=artifacts/foundation-ssl-pretrain-mae/main/foundation_core_U.pt \
  sbatch src/launch_transfer.sh

# 3. Tests (SLURM)
sbatch src/launch_test.sh
```

---

## Key commands reference

```bash
source env/cluster.env && source .venv/bin/activate

# Scaffold (legacy)
python -m src.foundation.pretrain --input-mode signal --arch mlp
python -m src.foundation.ssl_pretrain --input-mode signal --arch mlp

# Production
python -m src.foundation.pretrain --input-mode impedance --arch crt --logdir foundation-pretrain-crt
python -m src.foundation.ssl_pretrain --input-mode impedance --arch mae --ssl-arch mae

# Transfer
python -m src.foundation.transfer --subject subject013 \
  --core artifacts/foundation-pretrain-crt/main/foundation_core.pt
```

---

## Open / not yet done

- [ ] **CRT + MAE pretrain on cluster** (impedance, full cohort)
- [ ] **Transfer** with production cores (tagged logdirs)
- [ ] **Exp B** on production cores + more subjects
- [ ] **Exp C OOD** — needs `ood_exp.py` + launch script
- [ ] Re-export **best** Core S scaffold checkpoint (epoch 40) if still needed for comparison
- [ ] Optional: `dnclstm` + `image` track (Exp F)

---

## File map (artifacts)

```
artifacts/
  foundation-pretrain/main/           # scaffold Core S + MLP
  foundation-ssl-pretrain/main/       # scaffold Core U + MLP
  foundation-transfer-subject013/     # last transfer write (Core U overwrote S)
  exp-b-subject013/main/              # budget curve JSON/CSV
  foundation-pretrain-crt/main/       # (pending) production Core S
  foundation-ssl-pretrain-mae/main/   # (pending) production Core U
```

---

*Last updated: 2025-07-01 — production arch wiring landed; cluster CRT/MAE jobs not yet submitted.*
