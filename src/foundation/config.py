"""Hyper-parameters for the PVI foundation model and its training entry points."""

from dataclasses import dataclass


@dataclass
class FoundationConfig:
    # --- data ---
    input_mode: str = "signal"      # InputMode value: img|impedance|signal|resistance|reactance
    output_mode: str = "waveform"   # OutputMode value: sbp|dbp|fiducials|waveform
    mask_key: str = "mask05"        # SequenceMask value: mask01|mask05|mask10|mask15
    test_size: float = 0.1
    split_seed: int = 42            # disjoint subject shuffle (lazy + parquet cache)
    branch: str = "main"            # TrainingBranch: main|holdout|longitudinal (PLAN.md §7.1)
    split_mode: str = "disjoint"    # SplitMode: disjoint (PD) | within (PW) | global

    # --- parquet cache (set PVI_CACHE_ROOT or pass --cache-root) ---
    # NOTE: a cache is built for exactly one (branch, split_mode) pair (PLAN.md §3.8/§9) --
    # build separate cache directories per pair and point --cache-root at the right one.
    cache_root: str | None = None     # overrides env when set
    cache_num_workers: int = 8        # DataLoader workers when training from cache

    # --- architecture ---
    # `mlp` = scaffold; production: `crt` (Core S), `mae` (Core U). See PLAN.md §4.
    arch: str = "mlp"
    ssl_arch: str | None = None       # SSL encoder; default mae when arch is production
    diff: int = 2
    use_stats: bool = True
    num_features: int = 512
    num_hidden_layers: int = 6
    readout_hidden: int = 0
    # crt (PviCNNTransformer)
    crt_projection_dim: int = 100
    crt_transformer_dim: int = 64
    crt_cnn_depth: int = 2
    crt_mlp_depth: int = 3
    crt_pe_type: str = "rrpe"
    # mae (PviMaskedTransformer / MAE SSL encoder)
    mae_d_model: int = 64
    mae_num_layers: int = 2
    mae_mlp_depth: int = 2
    # cnn
    cnn_num_layers: int = 2
    cnn_factor: int = 2

    # --- optimisation ---
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 1e-2
    mse_weight: float = 0.2         # base weight for MorphologyLoss
    min_epochs: int = 1
    max_epochs: int = 500

    # --- data loading / throughput (216-file lazy cohort) ---
    max_cache: int = 150            # RAM-cached source HDF5 files (250 GB node budget)
    cluster_size: int = 16          # files per PviBatchSampler chunk (must stay small)
    stratified: bool = True         # False for SSL cohort pretrain
    persistent_h5: bool = True      # keep HDF5 handles open (False for SSL)
    num_workers: int = 0            # keep 0 with persistent HDF5 handles unless fork-safety tested
    pin_memory: bool = True
    prefetch_factor: int = 2        # only used when num_workers > 0
    persistent_workers: bool = False
    clear_cache_every_epoch: bool = False
    eval_every: int = 1              # full test-set eval every epoch (needed for early stopping)
    use_amp: bool = True            # torch.autocast on CUDA

    # --- SSL pretext (masked reconstruction + causal forecasting) ---
    mask_ratio: float = 0.5         # fraction of channel x time entries hidden
    horizon: int = 10               # forecast window length (time steps)
    lambda_mask: float = 1.0        # weight on masked-reconstruction loss
    lambda_forecast: float = 1.0    # weight on forecasting loss
