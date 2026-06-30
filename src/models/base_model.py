from src.packages import *

class BasePviLearner(nn.Module, ABC):
    """Base learner with an explicit core / readout split.

    Every concrete model builds two sub-modules in ``_make_layers``:
      * ``self.core``    - the shared backbone (frozen/transferred across subjects)
      * ``self.readout`` - the (per-subject) head producing the BP target
    and implements ``forward_core`` / ``forward_readout``. ``forward`` is the
    composition of the two, so existing call-sites (``model(seqs, stats)``) are
    unchanged. Optional ``aux_heads`` share the core representation for
    multi-task probes (maneuver/state, HR, ...).
    """

    def __init__(self,
                 data_shapes: dict[str,tuple[int,...]],
                 diff: int=2,
                 use_stats: bool=True,
                 verbose: bool=True,
                 ) -> None:

        super().__init__()
        self._alias = type(self).__name__

        assert diff in [0, 1, 2], f"Invalid differential level {diff}. Must be an integer in [0, 1, 2]"
        self._diff = diff
        self.nan_values = 0.0

        self._use_stats = use_stats
        self._read_data_shapes(data_shapes)
        self._verbose = verbose

        # Populated by subclasses in _make_layers():
        self.core: nn.Module = None       # shared backbone
        self.readout: nn.Module = None     # primary (per-subject) head
        self.feature_size: int = None      # dimensionality of forward_core output

        # Optional auxiliary heads operating on the shared core representation.
        self.aux_heads = nn.ModuleDict()

    def process_batch(self,
                      batch: dict[str, torch.Tensor],
                      ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        targets = batch['bp'] # Shape: (batch_size, output_size)

        input_sequences = {'pviLP': batch['pviLP'],
                           'pviHP': batch['pviHP']} # Shape: ((B, C, H, W, T), (B, C, H, W, T))

        if self._use_stats:
            input_stats = batch['stats']
        else:
            input_stats = torch.FloatTensor([])

        return input_sequences, input_stats, targets

    def _read_data_shapes(self,
                          data_shapes: dict[str,tuple[int,...]]) -> None:
        shapes = data_shapes['input']
        ndims = len(shapes)

        self.input_ndims = ndims
        self.input_shape = shapes
        self.sequence_length = int(shapes[-1])
        self.output_size = int(data_shapes['output'][0])

        self.num_channels = int(data_shapes['input'][0]) if not ndims == 1 else 1
        self.num_channels = int(self.num_channels * (self._diff + 1))

        if self._use_stats:
            self.stats_size = torch.tensor(data_shapes['stats']).prod().item()
        else:
            self.stats_size = 0
    def print_info(self) -> None:
        shapes = self.input_shape
        ndims = self.input_ndims
        print("="*15 + f" [{type(self).__name__}] " + "="*15)
        if ndims == 2:
            self._alias = f"{type(self).__name__} (1D)"
            print(f"\t Data shape: (C, T) = {shapes}.")
            print(f"\t Data modeled as {shapes[0]}-channel sequences of length {shapes[-1]}.")

        elif ndims == 4:
            self._alias = f"{type(self).__name__} (3D)"
            print(f"\t Data shape: (C, H, W, T) = {shapes}.")
            print(f"\t Data modeled as image sequences of dimension ({shapes[1]}, {shapes[2]}) and length {shapes[-1]}.")

        else:
            raise NotImplementedError(f"Input shape ({shapes}) not supported")

        print(f"\t Network variant: '{self._alias}'")
        print("="*15 + f" [{type(self).__name__}] " + "="*15)

    def _process_sequence(self,
                          sequences: dict[str,torch.Tensor]) -> torch.Tensor:
        # Compute diff and concatenate
        # Input shape: (batch_size, num_channels, sequence_length)
        # Output shape: (batch_size, 3*num_channels, sequence_length)

        # Shape: (batch_size, num_channels, sequence_length)
        xLP = torch.nan_to_num(sequences['pviLP'], nan=self.nan_values)
        xHP = torch.nan_to_num(sequences['pviHP'], nan=self.nan_values)

        dxLP = self._compute_diff(xLP)
        ddxLP = self._compute_diff(dxLP)

        if self._diff == 0:
            return xHP
        elif self._diff == 1:
            return torch.cat((xHP, dxLP), dim=1) # num_channels * 2
        else:
            return torch.cat((xHP, dxLP, ddxLP), dim=1) # num_channels * 3

    def _compute_diff(self, x: torch.Tensor) -> torch.Tensor:
        # Compute centered differences
        dx = (x[...,2:] - x[...,:-2]) / 2 # Shape: (B,C,H,W,T-2)
        dx = nn.functional.pad(dx,(1,1),mode='constant') # Shape: (B,C,H,W,T)
        return dx

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # z-score normalization
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-12
        x_norm = (x - x_mean) / x_std
        return x_norm

    # ------------------------------------------------------------- core/readout
    @abstractmethod
    def _make_layers(self) -> None:
        pass

    @abstractmethod
    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        """Shared backbone: inputs -> (B, feature_size) latent representation."""
        pass

    @abstractmethod
    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        """Primary head: core features -> (B, output_size) prediction."""
        pass

    def forward(self,
                input_sequences: dict[str, torch.Tensor],
                input_stats: torch.Tensor,
                head: str = None) -> torch.Tensor:
        # forward == forward_readout o forward_core  (backward-compatible).
        features = self.forward_core(input_sequences, input_stats)
        if head is None:
            return self.forward_readout(features, input_stats)
        if head not in self.aux_heads:
            raise KeyError(f"No aux head '{head}'. Available: {list(self.aux_heads)}")
        return self.aux_heads[head](features)

    # ----------------------------------------------------------------- aux heads
    @staticmethod
    def _default_head(in_features: int, out_features: int, hidden: int = 0) -> nn.Module:
        if hidden and hidden > 0:
            return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_features))
        return nn.Linear(in_features, out_features)

    def add_aux_head(self, name: str, module: nn.Module = None,
                     out_features: int = None, hidden: int = 0) -> nn.Module:
        """Attach an auxiliary head on top of the shared core representation."""
        if module is None:
            if out_features is None:
                raise ValueError("Provide either `module` or `out_features` for the aux head.")
            if self.feature_size is None:
                raise RuntimeError("feature_size is unset; build the model before adding aux heads.")
            module = self._default_head(self.feature_size, out_features, hidden)
        self.aux_heads[str(name)] = module
        return module

    # --------------------------------------------------------- freeze / transfer
    def freeze_core(self) -> "BasePviLearner":
        if self.core is None:
            raise RuntimeError("self.core is not set; nothing to freeze.")
        for p in self.core.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze_core(self) -> "BasePviLearner":
        for p in self.core.parameters():
            p.requires_grad_(True)
        return self

    def core_parameters(self):
        return self.core.parameters() if self.core is not None else iter(())

    def readout_parameters(self):
        return self.readout.parameters() if self.readout is not None else iter(())

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def core_state_dict(self) -> dict:
        return self.core.state_dict()

    def load_core_state_dict(self, state: dict, freeze: bool = True) -> "BasePviLearner":
        self.core.load_state_dict(state)
        if freeze:
            self.freeze_core()
        return self

    # --------------------------------------------------------------- properties
    @property
    def num_params(self) -> int:
        params_trainable = [p.numel() for p in self.parameters() if p.requires_grad]
        return sum(params_trainable)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_params_shallow(self) -> dict:

        params_available = sum([p.numel() for p in self.parameters()])
        params_trainable = sum([p.numel() for p in self.parameters() if p.requires_grad])

        dict_out = {'name': self._alias,
                    'total_params': params_available,
                    'trainable_params': params_trainable,
                    'device': self.device}

        all_modules = {}
        for name, module in self.named_modules():
            if name:
                if len(list(module.children())) == 0:  # leaf module
                    all_modules[name] = str(module)
                else:  # container
                    all_modules[name] = ''

        dict_out['modules'] = all_modules

        return dict_out


# --------------------------------------------------------------------- transfer
def transfer_core(src: BasePviLearner, dst: BasePviLearner, freeze: bool = True) -> BasePviLearner:
    """Copy the shared core weights from `src` into `dst` (and optionally freeze)."""
    dst.load_core_state_dict(src.core_state_dict(), freeze=freeze)
    return dst


def load_core_from_state_dict(model: BasePviLearner, full_state_dict: dict,
                              freeze: bool = True, prefix: str = "core.") -> BasePviLearner:
    """Load just the `core.*` entries from a full model state_dict (e.g. a checkpoint)."""
    core_state = {k[len(prefix):]: v for k, v in full_state_dict.items() if k.startswith(prefix)}
    if not core_state:
        raise KeyError(f"No '{prefix}*' keys found in the provided state_dict.")
    model.core.load_state_dict(core_state)
    if freeze:
        model.freeze_core()
    return model


class PviTestModel(BasePviLearner):
    '''Dummy model to test basic functionalities'''
    def __init__(self,
                 data_shapes: dict[str,tuple[int,...]],
                 diff: int=0,
                 use_stats: bool=True,
                 ) -> None:

        super().__init__(data_shapes, diff, use_stats)
        self._make_layers()
        self._alias = type(self).__name__

    def _make_layers(self) -> None:
        self.core = nn.Identity()
        self.readout = nn.Identity()
        self.feature_size = 0

    def forward_core(self, input_sequences=None, input_stats=None) -> torch.Tensor:
        return self.core(self._process_sequence(input_sequences).flatten(start_dim=1))

    def forward_readout(self, features, input_stats=None) -> torch.Tensor:
        return self.readout(features)

if __name__ == "__main__":
    pass
