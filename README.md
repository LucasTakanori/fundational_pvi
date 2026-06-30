# fundational_pvi

A **foundation model for PVI → blood-pressure prediction**, built by reusing the
data-loading and training infrastructure from the `pvi-ml` project and layering a
shared-core + per-subject-readout foundation model on top.

The approach follows the "core + readout" recipe (cf. the Nature paper
*Foundation model of neural activity predicts response to new stimulus types*,
`s41586-025-08829-y.pdf`): a single backbone (the **core**) is pretrained across
the pooled population so it learns a general representation, and each subject gets
a lightweight **readout** head. Transfer to a new subject = freeze the pretrained
core and fit a fresh readout on that subject's data.

## Layout

```
src/                     # the reusable PVI package (importable as `src`)
├── utils/               # primitives (enums, paths), HDF5 I/O, helpers
├── pipeline/            # dataset discovery, extraction, eager/lazy datasets
├── models/              # BasePviLearner + CNN/MLP/Transformer/S4 architectures,
│                        #   training workflow, trainer, loss, early stopping
├── scripts/             # original train/inference example scripts
├── analysis/            # post-training reporting
└── foundation/          # NEW: core + per-subject readout foundation model
    ├── core.py          #   PviCore            (shared backbone)
    ├── readout.py       #   SubjectReadout     (per-subject head)
    ├── foundation_model.py  # PviFoundationModel (BasePviLearner subclass)
    ├── pretrain.py      #   pool-population pretraining entry point
    ├── transfer.py      #   freeze-core + fit-readout transfer entry point
    └── config.py        #   FoundationConfig hyper-parameters
tests/test_smoke.py      # import + synthetic training-step smoke tests
legacy_matlab/           # original MATLAB analysis scripts (not used by Python)
```

## Install

```bash
pip install -r requirements.txt
```

A CPU build of torch is enough for the smoke test; install a CUDA build for real
training. `geomloss` and `mambapy` are optional (only `quality_evaluator.py` and
the WIP `s4_models.py` need them — their imports are guarded).

## Running

All code is the package `src`, so **run scripts as modules from the repo root**:

```bash
# Smoke test (no data needed)
python -m pytest tests/test_smoke.py -q

# Supervised cohort pretraining of the shared core (core S)
python -m src.foundation.pretrain --input-mode signal --output-mode waveform --max-epochs 50

# SSL pretraining of the shared core (core U): masked reconstruction + forecasting
python -m src.foundation.ssl_pretrain --input-mode signal --max-epochs 50

# Transfer the pretrained core to a held-out subject
python -m src.foundation.transfer --subject subject013 \
    --core artifacts/foundation-pretrain/main/foundation_core.pt
```

## Data

Datasets are discovered under `$PVIPROJECT_ROOT/datasets/main/` (defaulting to
`./data/datasets/main/` when the env var is unset), as `{subject}_{session}_masked.h5`
HDF5 files. **The recordings are not included in this repo** — only an empty
placeholder `src/utils/test.h5`. Point `PVIPROJECT_ROOT` (or `--ds-root`) at your
data before pretraining:

```bash
export PVIPROJECT_ROOT=/path/to/PviProject
```

Training artifacts (checkpoints, results, the saved core) are written under
`$PVIPROJECT_ROOT/artifacts/<logdir>/main/` and are git-ignored.

## Data-efficiency & analysis

- `BasePviLearner` exposes a core/readout split (`forward_core`/`forward_readout`),
  `freeze_core()`, and transfer helpers (`transfer_core`, `load_core_from_state_dict`)
  plus auxiliary heads (`add_aux_head`) — the basis for foundation pretrain → readout transfer.
- `PviConfiguredDataset.set_train_budget(n_seq=… | minutes=…, seed=…)` restricts the
  training set to a data budget (test set untouched) for the Exp A/B data-efficiency curves.
- `src/analysis/budget_curves.py` aggregates per-run metrics (`cc_abs` via `bp_accuracy`,
  `amae`/`armse` via `metrics_waveform`) into mean±sem foundation-vs-individual curves and plots them.
- Two interchangeable cores (same `PviCore`, transfer via `load_core_state_dict`):
  **S** = supervised cohort core (`src/foundation/pretrain.py`); **U** = SSL core
  (`src/foundation/ssl_pretrain.py`, masked reconstruction + forecasting in `src/foundation/ssl.py`).
- Multi-task heads (BP + maneuver/HR) in `src/foundation/multitask.py`; learned EIT
  reconstruction in `src/models/eit_recon.py`.
- `src/foundation/evaluation.py` (`evaluate`/`evaluate_ood`) + `experiments.py`
  (`run_budget_curve`) drive the Exp A/B/C/G matrix into records for `budget_curves`.

## Notes

- The foundation model is a **scaffold**: the core/readout architecture, the
  pretrain → transfer wiring, and the integration with the existing
  `TrainingWorkflow` are in place, but core depth, loss, and readout form should
  be tuned against the paper and real data.
- The dataset samples do not carry a subject id, so the model trains one *active*
  readout per run (pooled pretraining via the shared readout; per-subject readout
  during transfer) rather than routing mixed-subject batches per sample.
