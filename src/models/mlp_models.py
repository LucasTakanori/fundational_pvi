from torch import nn

from src.packages import *

from src.models.base_model import BasePviLearner


class PviLinearRegression(BasePviLearner):
    def __init__(self,
                 data_shapes: dict[str,tuple[int,...]],
                 diff: int = 2,
                 use_stats: bool=True,
                 ) -> None:

        super().__init__(data_shapes=data_shapes, diff=diff, use_stats=use_stats)

        self._make_layers()

    def _make_layers(self) -> None:
        if self.input_ndims == 2:
            flatten_size = self.num_channels*self.sequence_length
        elif self.input_ndims == 4:
            flatten_size = self.num_channels*(40**2)*self.sequence_length
        else:
            flatten_size = 0

        # forward_core emits the flattened input (without stats); stats are
        # injected pre-readout, so the readout Linear consumes flatten + stats.
        self.feature_size = flatten_size
        self.core = nn.Identity()
        self.readout = nn.Linear(flatten_size + self.stats_size, self.output_size)

    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        s = self._process_sequence(input_sequences)
        s = s.flatten(start_dim=1)
        return self.core(s)

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        if input_stats.numel():
            features = torch.hstack([features, input_stats.flatten(start_dim=1)])
        return self.readout(features)


class PviMLP(BasePviLearner):
    def __init__(self,
                 data_shapes: dict[str, tuple[int, ...]],
                 diff: int = 2,
                 use_stats: bool = True,
                 num_hidden_layers: int = 5,
                 num_features: int = 100,
                 ) -> None:

        super().__init__(data_shapes=data_shapes, diff=diff, use_stats=use_stats)

        self.num_hidden_layers = num_hidden_layers
        self.num_features = num_features
        self._make_layers()

    def _make_layers(self) -> None:
        if self.input_ndims == 2:
            flatten_size = self.num_channels * self.sequence_length
        elif self.input_ndims == 4:
            flatten_size = self.num_channels * (40 ** 2) * self.sequence_length
        else:
            flatten_size = 0

        flatten_size += self.stats_size

        # Core = input projection + hidden trunk (the shared representation);
        # stats are concatenated before the trunk (as in the original model).
        trunk = [nn.Sequential(nn.Linear(flatten_size, self.num_features), nn.ReLU())]
        for _ in range(self.num_hidden_layers):
            trunk.append(nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU()))
        self.core = nn.Sequential(*trunk)

        self.feature_size = self.num_features
        self.readout = nn.Linear(self.num_features, self.output_size)

    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        s = self._process_sequence(input_sequences)
        s = s.flatten(start_dim=1)
        if input_stats.numel():
            s = torch.hstack([s, input_stats.flatten(start_dim=1)])
        return self.core(s)

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        return self.readout(features)


if __name__ == "__main__":
    pass
