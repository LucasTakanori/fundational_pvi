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

With [uv](https://docs.astral.sh/uv/) (recommended on the cluster):

```bash
cd /mmfs1/projects/ece_bst/lsanc68/fundational_pvi
uv sync          # creates .venv/ and installs deps (CUDA 12.1 torch via pyproject.toml)
source .venv/bin/activate
```

Or with pip:

```bash
pip install -r requirements.txt
```

A CPU build of torch is enough for the smoke test; the `pyproject.toml` pins a
CUDA 12.1 PyTorch wheel for GPU training. Change `[tool.uv]` if your node's
driver needs a different CUDA build.

## Cluster jobs

Paths for this project live in `env/cluster.env`:

```bash
export PVIPROJECT_ROOT=/mmfs1/projects/ece_bst/lsanc68/fundational_pvi   # artifacts
export PVI_DATA_ROOT=/home/lsanc68/ece_bst_link/common/data/pvi_data      # …/main/*.h5
```

Submit a GPU job (1 GPU, 16 CPUs, 250 GB RAM, 10 days):

```bash
sbatch src/launch.sh       # core S (supervised cohort pretrain)
sbatch src/launch_ssl.sh   # core U (SSL pretrain)
```

Override the command:

```bash
JOB_SCRIPT="python -m src.foundation.ssl_pretrain --max-epochs 50" sbatch src/launch.sh
```

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

Datasets are discovered under `$PVI_DATA_ROOT/main/` (see `env/cluster.env`),
`$PVIPROJECT_ROOT/datasets/main/`, or `./data/datasets/main/` when unset, as
`{subject}_{session}_masked.h5` HDF5 files. **The recordings are not included in
this repo** — only an empty placeholder `src/utils/test.h5`. Set `PVI_DATA_ROOT`
(or `PVIPROJECT_ROOT` / `--ds-root`) before pretraining:

```bash
source env/cluster.env
# or:
export PVI_DATA_ROOT=/home/lsanc68/ece_bst_link/common/data/pvi_data
export PVIPROJECT_ROOT=/mmfs1/projects/ece_bst/lsanc68/fundational_pvi
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
- `src/analysis/interpretability.py` (Exp D/E): in-silico input perturbation, gradient
  saliency, latent extraction (for UMAP), and the readout functional-barcode probe.

See [`MODELS.md`](MODELS.md) for which architecture to use per experiment phase.

`linear`, `mlp`, `cnn` (PviCNN), `crt` (PviCNNTransformer), `samba` (PviSamba, WIP),
`mae` (PviMaskedTransformer — tokenized MAE-style encoder, candidate 2), and
`dnclstm` (PviDenseNetConvLSTM — paper-faithful 3D-DenseNet + Conv-LSTM + spatial
bilinear readout, candidate 4).

## Notes

- The foundation model is a **scaffold**: the core/readout architecture, the
  pretrain → transfer wiring, and the integration with the existing
  `TrainingWorkflow` are in place, but core depth, loss, and readout form should
  be tuned against the paper and real data.
- The dataset samples do not carry a subject id, so the model trains one *active*
  readout per run (pooled pretraining via the shared readout; per-subject readout
  during transfer) rather than routing mixed-subject batches per sample.
