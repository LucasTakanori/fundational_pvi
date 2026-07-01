# Models per task

Which neural architecture to use for each experiment phase. The **foundation
entry points** (`pretrain.py`, `ssl_pretrain.py`, `transfer.py`) currently
instantiate `PviFoundationModel` with a shared **`PviCore` MLP** — a deliberate
**scaffold** to validate the core/readout + transfer wiring before scaling up.

Production runs should move to the larger `src/models/*` architectures listed
below (all implement `forward_core` / `forward_readout` on `BasePviLearner`).

See also [`PLAN.md`](PLAN.md) §56–66 and [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Architecture catalog

| Tag | Class | Best input | Role |
|-----|-------|------------|------|
| `linear` | `PviLinearRegression` | signal | Sanity / lower bound |
| `mlp` | `PviMLP` / `PviFoundationModel`+`PviCore` | signal, impedance | Scaffold + fast debug |
| `cnn` | `PviCNN` | signal, impedance | Local temporal conv baseline |
| `crt` | `PviCNNTransformer` | impedance, signal | **Primary supervised core (S)** — conv + LSTM PE + transformer |
| `mae` | `PviMaskedTransformer` | impedance, image patches | **Primary SSL core (U)** — MAE-style tokens |
| `dnclstm` | `PviDenseNetConvLSTM` | **image** (40×40 EIT) | Paper-faithful 3D DenseNet + Conv-LSTM + spatial readout |
| `samba` | `PviSamba` (WIP) | long raw sequences | SSM / Mamba for high-rate impedance |

Legacy scripts resolve tags via `src/models/_model_mapper.py` (`ml_session_mapper`).

---

## Model × task matrix

| Phase / experiment | Model | Input mode | Notes |
|--------------------|-------|------------|-------|
| **P0 — Core S** (supervised cohort) | `crt` or `cnn` | `impedance` | Current launch uses scaffold `mlp`+`signal`; switch before paper runs |
| **P0 — Core U** (SSL cohort) | `mae` | `impedance` | Current launch uses `PviSSLModel`+`PviCore`; same scaffold |
| **Transfer** (Exp B) | Same core as P0 + `SubjectReadout` | match pretrain | `foundation.transfer` loads `foundation_core.pt` or `foundation_core_U.pt` |
| **Exp A** (individual baseline) | `cnn` or `crt` | `impedance` | `train_subjects_v2.py` + `_model_mapper` |
| **Exp C** (OOD maneuvers) | Frozen core from P0 + readout | `impedance` | Train baseline only; test valsalva/pressor |
| **Exp D/E** (digital twin / barcode) | Best core from arch sweep | `impedance` or `image` | + `multitask.py` aux heads |
| **Exp F** (EIT recon) | `dnclstm` or `crt` on learned image | `image` | `eit_recon.py` sub-track |
| **Exp G** (U vs S) | Compare `mae` vs `crt` cores | `impedance` | Same transfer benchmark |

---

## Why the current jobs use a small MLP

1. **Scaffold first** — `PviFoundationModel` + `PviCore` was added to prove pooled
   pretrain → freeze → transfer without refactoring every architecture at once.
2. **`--input-mode signal`** — 1-D aggregated signal (~50 time steps) is the
   smallest input; even a large MLP cannot saturate a GPU.
3. **Defaults were conservative** — `batch_size=32`, `max_cache=50`, cache
   cleared every epoch, full test eval every epoch (now improved; see below).

After the scaffold converges, **re-run P0 with `impedance` + `crt`/`mae`** for
the Nature-target experiments.

---

## Recommended production defaults (next runs)

```bash
# Core S — supervised cohort (example; wiring TBD for non-mlp arch)
python -m src.foundation.pretrain \
  --input-mode impedance --output-mode waveform \
  --batch-size 256 --max-cache 150 --eval-every 5

# Core U — SSL (example)
python -m src.foundation.ssl_pretrain \
  --input-mode impedance \
  --batch-size 256 --max-cache 150
```

`dnclstm` + `--input-mode image` when running the paper-faithful image core
(Exp F and candidate 4 in PLAN.md).

---

## Throughput settings (shared)

Configured in `FoundationConfig` / CLI:

| Setting | Default | Purpose |
|---------|---------|---------|
| `batch_size` | 512 | Larger steps per epoch (signal MLP is still tiny) |
| `max_cache` | 150 | Keep HDF5 files warm in 250 GB RAM |
| `clear_cache_every_epoch` | false | Stop re-reading disk every epoch |
| `eval_every` | 1 | Full test eval every epoch (early stopping uses test accuracy) |
| `use_amp` | true | Mixed precision on CUDA |
| `num_workers` | 0 | Safe with persistent HDF5 handles; raise only after fork testing |
