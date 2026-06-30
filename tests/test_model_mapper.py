"""Tests for _model_mapper after the core/readout rename + candidate-4 registration."""

import json

from src.utils.primitives import InputMode, OutputMode
from src.models._model_mapper import ml_session_mapper, get_samba_kwargs
from src.models.attn_models import PviCNNTransformer
from src.models.cnn_models import PviCNN
from src.models.densenet_convlstm import PviDenseNetConvLSTM

SHAPES_1D = {"input": (1, 50), "output": (8,), "stats": (1,)}


def test_session_mapper_known_models():
    model, im, om = ml_session_mapper("ps17-cnn-img-to-waveform")
    assert model is PviCNN and im == InputMode.IMAGE and om == OutputMode.WAVEFORM

    model, _, _ = ml_session_mapper("ps18-dnclstm-img-to-waveform")
    assert model is PviDenseNetConvLSTM


def test_get_samba_kwargs_reads_namespaced_modules(tmp_path):
    # Build a model, serialise its module map the way training checkpoints do.
    model = PviCNNTransformer(SHAPES_1D, cnn_depth=2, mlp_depth=2, pe_type="rrpe")
    config = {"model": model.get_params_shallow()}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, default=str))

    kw = get_samba_kwargs(path)
    assert kw["cnn_depth"] == 2          # core.conv_layers.0/.1
    assert kw["mlp_depth"] == 2          # two Linear layers under readout.*
    assert kw["pe_type"] == "rrpe"
