"""`PviFoundationModel` - a shared core + per-subject readout heads.

Supports multiple core architectures (``mlp``, ``crt``, ``mae``, ``cnn``) via the
``arch`` argument.  Each architecture reuses the corresponding ``BasePviLearner``
encoder body; per-subject ``SubjectReadout`` heads handle BP prediction during
pooled pretrain and transfer.
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)

from src.models.base_model import BasePviLearner
from src.models.cnn_models import PviCNN
from src.models.attn_models import PviCNNTransformer
from src.models.mae_transformer import PviMaskedTransformer
from src.models.s4_models import PviSamba
from src.foundation.core import PviCore
from src.foundation.readout import SubjectReadout
from src.foundation.arch import normalize_arch

SHARED_READOUT = "shared"


class PviFoundationModel(BasePviLearner):
    SUPPORTED_ARCHES = frozenset({"mlp", "crt", "mae", "cnn", "samba"})

    def __init__(self,
                 data_shapes: dict[str, tuple[int, ...]],
                 arch: str = "mlp",
                 num_features: int = 200,
                 num_hidden_layers: int = 4,
                 readout_hidden: int = 0,
                 subjects: list[str] | None = None,
                 diff: int = 2,
                 use_stats: bool = True,
                 verbose: bool = True,
                 # crt
                 projection_dim: int = 100,
                 transformer_dim: int = 64,
                 cnn_depth: int = 2,
                 mlp_depth: int = 3,
                 pe_type: str = "rrpe",
                 # samba
                 samba_projection_dim: int = 100,
                 samba_mamba_layers: int = 2,
                 samba_samba_layers: int = 2,
                 samba_cnn_depth: int = 1,
                 samba_mlp_depth: int = 1,
                 samba_pe_type: str = "rrpe",
                 # mae
                 d_model: int = 64,
                 num_layers: int = 2,
                 mlp_depth_mae: int | None = None,
                 # cnn
                 num_conv_layers: int = 2,
                 factor: int = 2,
                 ) -> None:

        self.arch = normalize_arch(arch)
        if self.arch not in self.SUPPORTED_ARCHES:
            raise ValueError(
                f"Unsupported arch '{arch}'. Choose from {sorted(self.SUPPORTED_ARCHES)}."
            )

        super().__init__(data_shapes=data_shapes, diff=diff,
                         use_stats=use_stats, verbose=verbose)

        self.num_features = int(num_features)
        self.num_hidden_layers = int(num_hidden_layers)
        self.readout_hidden = int(readout_hidden)

        self._encoder = None
        self._stats_in_readout = self.arch != "mlp"

        self._crt_kwargs = dict(
            projection_dim=projection_dim,
            transformer_dim=transformer_dim,
            cnn_depth=cnn_depth,
            mlp_depth=mlp_depth,
            pe_type=pe_type,
        )
        self._samba_kwargs = dict(
            projection_dim=samba_projection_dim,
            mamba_layers=samba_mamba_layers,
            samba_layers=samba_samba_layers,
            cnn_depth=samba_cnn_depth,
            mlp_depth=samba_mlp_depth,
            pe_type=samba_pe_type,
        )
        self._mae_kwargs = dict(
            d_model=d_model,
            num_layers=num_layers,
            mlp_depth=mlp_depth_mae if mlp_depth_mae is not None else 2,
        )
        self._cnn_kwargs = dict(num_conv_layers=num_conv_layers, factor=factor)

        self._make_layers()

        self.readouts = nn.ModuleDict()
        self.add_readout(SHARED_READOUT)
        for sid in (subjects or []):
            self.add_readout(sid)
        self._active = SHARED_READOUT

    def _data_shapes_dict(self) -> dict:
        stats_shape = (int(self.stats_size),) if self.stats_size else (0,)
        return {
            "input": self.input_shape,
            "output": (self.output_size,),
            "stats": stats_shape,
        }

    # ------------------------------------------------------------------ build
    def _flatten_size_mlp(self) -> int:
        if self.input_ndims == 2:
            n = self.num_channels * self.sequence_length
        elif self.input_ndims == 4:
            n = self.num_channels * (40 ** 2) * self.sequence_length
        else:
            n = 0
        return int(n + self.stats_size)

    def _readout_input_dim(self) -> int:
        if self.arch == "mlp":
            return self.num_features
        return int(self.feature_size + self.stats_size)

    def _freeze_encoder_readout(self) -> None:
        if self._encoder is not None and getattr(self._encoder, "readout", None) is not None:
            for param in self._encoder.readout.parameters():
                param.requires_grad_(False)

    def _make_layers(self) -> None:
        if self.arch == "mlp":
            self.core = PviCore(self._flatten_size_mlp(),
                                num_features=self.num_features,
                                num_hidden_layers=self.num_hidden_layers)
            self.feature_size = self.num_features
            return

        shapes = self._data_shapes_dict()
        if self.arch == "crt":
            self._encoder = PviCNNTransformer(shapes, **self._crt_kwargs)
        elif self.arch == "samba":
            self._encoder = PviSamba(shapes, **self._samba_kwargs)
        elif self.arch == "mae":
            self._encoder = PviMaskedTransformer(
                shapes, diff=self._diff, use_stats=self._use_stats,
                verbose=False, **self._mae_kwargs,
            )
        elif self.arch == "cnn":
            self._encoder = PviCNN(
                shapes, diff=self._diff, use_stats=self._use_stats,
                **self._cnn_kwargs,
            )
        else:
            raise RuntimeError(self.arch)

        self.core = self._encoder.core
        self.feature_size = self._encoder.feature_size
        self._freeze_encoder_readout()

    def _flatten_input(self,
                       input_sequences: dict[str, torch.Tensor],
                       input_stats: torch.Tensor) -> torch.Tensor:
        s = self._process_sequence(input_sequences)
        s = s.flatten(start_dim=1)
        if input_stats.numel():
            s = torch.hstack([s, input_stats.flatten(start_dim=1)])
        return s

    # -------------------------------------------------------------- readouts
    def add_readout(self, name: str) -> SubjectReadout:
        name = str(name)
        if name not in self.readouts:
            self.readouts[name] = SubjectReadout(
                self._readout_input_dim(),
                self.output_size,
                hidden=self.readout_hidden,
            )
        return self.readouts[name]

    def set_active(self, name: str) -> "PviFoundationModel":
        name = str(name)
        if name not in self.readouts:
            raise KeyError(f"No readout '{name}'. Call add_readout('{name}') first. "
                           f"Available: {list(self.readouts)}")
        self._active = name
        self.readout = self.readouts[name]
        return self

    @property
    def active(self) -> str:
        return self._active

    @property
    def subjects(self) -> list[str]:
        return [k for k in self.readouts if k != SHARED_READOUT]

    # ---------------------------------------------------------- core/readout
    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        if self.arch == "mlp":
            return self.core(self._flatten_input(input_sequences, input_stats))
        return self._encoder.forward_core(input_sequences, input_stats)

    encode = forward_core

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        if self._stats_in_readout and input_stats.numel():
            features = torch.hstack([features, input_stats.flatten(start_dim=1)])
        return self.readouts[self._active](features)

    def forward_for(self,
                    input_sequences: dict[str, torch.Tensor],
                    input_stats: torch.Tensor,
                    subject: str) -> torch.Tensor:
        prev = self._active
        try:
            return self.set_active(subject).forward(input_sequences, input_stats)
        finally:
            self.set_active(prev)
