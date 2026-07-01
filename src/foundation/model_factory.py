"""Build foundation / SSL models from ``FoundationConfig`` and dataset shapes."""

from __future__ import annotations

from src.foundation.config import FoundationConfig
from src.foundation.arch import normalize_arch
from src.foundation.foundation_model import PviFoundationModel
from src.foundation.ssl import PviSSLModel


def foundation_kwargs(cfg: FoundationConfig) -> dict:
    """Keyword args for ``PviFoundationModel`` from config."""
    arch = normalize_arch(cfg.arch)
    kw = {
        "arch": arch,
        "num_features": cfg.num_features,
        "num_hidden_layers": cfg.num_hidden_layers,
        "readout_hidden": cfg.readout_hidden,
        "diff": cfg.diff,
        "use_stats": cfg.use_stats,
        "verbose": False,
    }
    if arch == "crt":
        kw.update({
            "projection_dim": cfg.crt_projection_dim,
            "transformer_dim": cfg.crt_transformer_dim,
            "cnn_depth": cfg.crt_cnn_depth,
            "mlp_depth": cfg.crt_mlp_depth,
            "pe_type": cfg.crt_pe_type,
        })
    elif arch == "mae":
        kw.update({
            "d_model": cfg.mae_d_model,
            "num_layers": cfg.mae_num_layers,
            "mlp_depth_mae": cfg.mae_mlp_depth,
        })
    elif arch == "samba":
        kw.update({
            "samba_projection_dim": cfg.samba_projection_dim,
            "samba_mamba_layers": cfg.samba_mamba_layers,
            "samba_samba_layers": cfg.samba_samba_layers,
            "samba_cnn_depth": cfg.samba_cnn_depth,
            "samba_mlp_depth": cfg.samba_mlp_depth,
            "samba_pe_type": cfg.samba_pe_type,
        })
    elif arch == "cnn":
        kw.update({
            "num_conv_layers": cfg.cnn_num_layers,
            "factor": cfg.cnn_factor,
        })
    return kw


def ssl_kwargs(cfg: FoundationConfig) -> dict:
    arch = normalize_arch(cfg.ssl_arch or cfg.arch)
    kw = {
        "arch": arch,
        "num_features": cfg.num_features,
        "num_hidden_layers": cfg.num_hidden_layers,
        "mask_ratio": cfg.mask_ratio,
        "horizon": cfg.horizon,
        "lambda_mask": cfg.lambda_mask,
        "lambda_forecast": cfg.lambda_forecast,
        "diff": cfg.diff,
        "use_stats": cfg.use_stats,
        "verbose": False,
    }
    if arch == "mae":
        kw.update({
            "d_model": cfg.mae_d_model,
            "num_layers": cfg.mae_num_layers,
        })
    return kw


def build_foundation_model(data_shapes: dict,
                           cfg: FoundationConfig,
                           subjects: list[str] | None = None,
                           verbose: bool = False) -> PviFoundationModel:
    kw = foundation_kwargs(cfg)
    kw["verbose"] = verbose
    return PviFoundationModel(data_shapes, subjects=subjects, **kw)


def build_ssl_model(data_shapes: dict,
                    cfg: FoundationConfig,
                    verbose: bool = False) -> PviSSLModel:
    kw = ssl_kwargs(cfg)
    kw["verbose"] = verbose
    return PviSSLModel(data_shapes, **kw)


def default_supervised_config() -> FoundationConfig:
    """Production Core S defaults (``crt`` + ``impedance``)."""
    return FoundationConfig(
        input_mode="impedance",
        output_mode="waveform",
        arch="crt",
        batch_size=256,
        max_cache=150,
        eval_every=5,
        stratified=True,
    )


def default_ssl_config() -> FoundationConfig:
    """Production Core U defaults (``mae`` + ``impedance``)."""
    return FoundationConfig(
        input_mode="impedance",
        output_mode="waveform",
        arch="mae",
        ssl_arch="mae",
        batch_size=256,
        max_cache=150,
        stratified=False,
        persistent_h5=False,
    )
