"""`PviFoundationModel` - a shared core + per-subject readout heads.

This subclasses `BasePviLearner` and implements the `forward_core` /
`forward_readout` contract: `forward_core` is the shared encoder, and
`forward_readout` routes through the currently *active* per-subject readout.
`forward` (inherited) composes the two, so the model plugs straight into the
existing training stack (`trainer_v3.py` / `workflow_v3.py`).

    * Pretraining (pooled population): keep the SHARED_READOUT active.
    * Transfer (new subject): `freeze_core()` (inherited), `add_readout(subject)`,
      `set_active(subject)`, then train only the readout.

The dataset samples do not carry a subject id, so per-sample routing within a
mixed batch is intentionally not attempted; the paper's core+readout transfer
protocol (one active readout per training run) is what this models.
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)

from src.models.base_model import BasePviLearner
from src.foundation.core import PviCore
from src.foundation.readout import SubjectReadout

SHARED_READOUT = "shared"


class PviFoundationModel(BasePviLearner):
    def __init__(self,
                 data_shapes: dict[str, tuple[int, ...]],
                 num_features: int = 200,
                 num_hidden_layers: int = 4,
                 readout_hidden: int = 0,
                 subjects: list[str] | None = None,
                 diff: int = 2,
                 use_stats: bool = True,
                 verbose: bool = True,
                 ) -> None:

        super().__init__(data_shapes=data_shapes, diff=diff,
                         use_stats=use_stats, verbose=verbose)

        self.num_features = int(num_features)
        self.num_hidden_layers = int(num_hidden_layers)
        self.readout_hidden = int(readout_hidden)

        self._make_layers()

        # One shared readout (used for pooled pretraining) plus any requested
        # per-subject readouts. `self.readout` (the BasePviLearner primary head)
        # is kept pointing at the active readout for API compatibility.
        self.readouts = nn.ModuleDict()
        self.add_readout(SHARED_READOUT)
        for sid in (subjects or []):
            self.add_readout(sid)
        self._active = SHARED_READOUT

    # ------------------------------------------------------------------ build
    def _flatten_size(self) -> int:
        if self.input_ndims == 2:        # (C, T)
            n = self.num_channels * self.sequence_length
        elif self.input_ndims == 4:      # (C, H, W, T); image side is 40x40
            n = self.num_channels * (40 ** 2) * self.sequence_length
        else:
            n = 0
        return int(n + self.stats_size)

    def _make_layers(self) -> None:
        self.core = PviCore(self._flatten_size(),
                            num_features=self.num_features,
                            num_hidden_layers=self.num_hidden_layers)
        self.feature_size = self.num_features

    def _flatten_input(self,
                       input_sequences: dict[str, torch.Tensor],
                       input_stats: torch.Tensor) -> torch.Tensor:
        s = self._process_sequence(input_sequences)   # (B, C, ...) with diffs concatenated
        s = s.flatten(start_dim=1)
        if input_stats.numel():
            s = torch.hstack([s, input_stats.flatten(start_dim=1)])
        return s

    # -------------------------------------------------------------- readouts
    def add_readout(self, name: str) -> SubjectReadout:
        name = str(name)
        if name not in self.readouts:
            self.readouts[name] = SubjectReadout(self.num_features,
                                                 self.output_size,
                                                 hidden=self.readout_hidden)
        return self.readouts[name]

    def set_active(self, name: str) -> "PviFoundationModel":
        name = str(name)
        if name not in self.readouts:
            raise KeyError(f"No readout '{name}'. Call add_readout('{name}') first. "
                           f"Available: {list(self.readouts)}")
        self._active = name
        self.readout = self.readouts[name]   # keep BasePviLearner.readout in sync
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
        """Shared-core features (subject-agnostic)."""
        return self.core(self._flatten_input(input_sequences, input_stats))

    # convenience alias
    encode = forward_core

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        return self.readouts[self._active](features)

    def forward_for(self,
                    input_sequences: dict[str, torch.Tensor],
                    input_stats: torch.Tensor,
                    subject: str) -> torch.Tensor:
        """Forward through a specific subject's readout without changing state."""
        prev = self._active
        try:
            return self.set_active(subject).forward(input_sequences, input_stats)
        finally:
            self.set_active(prev)
