"""Hyper-parameters for the PVI foundation model and its training entry points."""

from dataclasses import dataclass


@dataclass
class FoundationConfig:
    # --- data ---
    input_mode: str = "signal"      # InputMode value: img|impedance|signal|resistance|reactance
    output_mode: str = "waveform"   # OutputMode value: sbp|dbp|fiducials|waveform
    mask_key: str = "mask05"        # SequenceMask value: mask01|mask05|mask10|mask15
    test_size: float = 0.1

    # --- architecture ---
    diff: int = 2                   # differential channels added by BasePviLearner (0|1|2)
    use_stats: bool = True
    num_features: int = 200         # width of the shared core's latent representation
    num_hidden_layers: int = 4      # hidden layers in the core trunk
    readout_hidden: int = 0         # >0 adds a hidden layer to each readout; 0 = linear readout

    # --- optimisation ---
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-2
    mse_weight: float = 0.2         # base weight for MorphologyLoss
    min_epochs: int = 1
    max_epochs: int = 500

    # --- SSL pretext (masked reconstruction + causal forecasting) ---
    mask_ratio: float = 0.5         # fraction of channel x time entries hidden
    horizon: int = 10               # forecast window length (time steps)
    lambda_mask: float = 1.0        # weight on masked-reconstruction loss
    lambda_forecast: float = 1.0    # weight on forecasting loss
