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

        flatten_size += self.stats_size

        self.fc = nn.Linear(flatten_size, self.output_size)

    def forward(self,
                input_sequences: dict[str, torch.Tensor],
                input_stats: torch.Tensor
                ) -> torch.Tensor:
        s = self._process_sequence(input_sequences) # shape: (B, C, H, W, T)
        s = s.flatten(start_dim=1)  # Flatten all dimensions except batch dimension

        if input_stats.numel():
            f = input_stats.flatten(start_dim=1)
            s = torch.hstack([s,f])

        outputs = self.fc(s)

        return outputs

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

        self.fc_in = nn.Sequential(
                nn.Linear(flatten_size, self.num_features),
                nn.ReLU()
                    )

        self.fc_layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            self.fc_layers.append(
                    nn.Sequential(
                            nn.Linear(self.num_features, self.num_features),
                            nn.ReLU()
                    ))

        self.fc_out = nn.Linear(self.num_features, self.output_size)

    def forward(self,
                input_sequences: dict[str, torch.Tensor],
                input_stats: torch.Tensor
                ) -> torch.Tensor:
        s = self._process_sequence(input_sequences)  # shape: (B, C, H, W, T)
        s = s.flatten(start_dim=1)  # Flatten all dimensions except batch dimension


        if input_stats.numel():
            f = input_stats.flatten(start_dim=1)
            s = torch.hstack([s, f])

        s = self.fc_in(s)

        for fc_layer in self.fc_layers:
            s = fc_layer(s)

        outputs = self.fc_out(s)

        return outputs

if __name__ == "__main__":
    pass