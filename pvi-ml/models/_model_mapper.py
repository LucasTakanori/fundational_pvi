import json
from pathlib import Path

from src.utils.primitives import InputMode, OutputMode
from src.models.attn_models import PviCNNTransformer
from src.models.cnn_models import PviCNN
from src.models.mlp_models import PviLinearRegression, PviMLP
from src.models.s4_models import PviSamba


def ml_session_mapper(identifier: str):
    model_lookup = {'linear': PviLinearRegression,
                    'mlp': PviMLP,
                    'cnn': PviCNN,
                    'crt': PviCNNTransformer,
                    'samba': PviSamba}

    keywords = identifier.split(sep='-')

    model = model_lookup[keywords[1]]
    input_mode = InputMode(keywords[2])
    output_mode = OutputMode(keywords[4])

    return (model, input_mode, output_mode)


def get_samba_kwargs(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)

    modules = config['model']['modules']

    # cnn_depth: count top-level conv_layers.N entries (exactly one dot after conv_layers)
    cnn_depth = sum(1 for k in modules if k.startswith('conv_layers.') and k.count('.') == 1)

    # pe_type: infer from the rrpe module string
    rrpe_str = modules.get('rrpe', '')
    if 'LearnablePositional' in rrpe_str:
        pe_type = 'learnable'
    elif 'Sinusoidal' in rrpe_str:
        pe_type = 'sinusoidal'
    elif 'Identity' in rrpe_str:
        pe_type = 'none'
    else:
        pe_type = 'rrpe'

    # mlp_depth: count Linear layers under mlp.*
    mlp_linears = sum(1 for k, v in modules.items()
                      if k.startswith('mlp.') and isinstance(v, str) and v.startswith('Linear'))
    if mlp_linears <= 1:
        mlp_depth = 1
    elif mlp_linears == 2:
        mlp_depth = 2
    else:
        mlp_depth = 3

    return {'cnn_depth': cnn_depth, 'pe_type': pe_type, 'mlp_depth': mlp_depth}
