from src.packages import *

class BasePviLearner(nn.Module, ABC):
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

        # if self._verbose:
        #     self.print_info()

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

    @abstractmethod
    def _make_layers(self) -> None:
        pass

    @abstractmethod
    def forward(self,
                input_sequences: dict[str, torch.Tensor],
                input_stats: torch.Tensor
                ) -> torch.Tensor:
        # Must be implemented by subclasses
        pass

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

class PviTestModel(BasePviLearner):
    '''Dummy model to test basic functionalities'''
    def __init__(self,
                 data_shapes: dict[str,tuple[int,...]],
                 diff: int=0,
                 use_stats: bool=True,
                 ) -> None:

        super().__init__(data_shapes, diff, use_stats)

        self._alias = type(self).__name__

        print(f"{self._alias} (WARNING): The class '{type(self).__name__}' is meant for testing and does not contain any method!!!")

    def _make_layers(self) -> None:
        pass

    def forward(self, input_sequences=None, input_stats=None) -> torch.Tensor:
        pass

if __name__ == "__main__":
    pass