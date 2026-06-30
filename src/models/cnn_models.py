from src.packages import *

from src.models.base_model import BasePviLearner

class PviCNN(BasePviLearner):
    def __init__(self,
                 data_shapes: dict[str, tuple[int,...]],
                 diff: int=2,
                 use_stats: bool=True,
                 num_conv_layers: int = 2,
                 factor: int = 2,
                 ) -> None:

        super().__init__(data_shapes=data_shapes, diff=diff, use_stats=use_stats)

        self.num_conv_layers = num_conv_layers
        self.factor = factor
        self._make_layers()

    def _make_layers(self) -> None:

        in_multipliers = [2**i for i in range(self.num_conv_layers)]
        out_multipliers = [2**(i+1) for i in range(self.num_conv_layers)]
        tdim_kernel = [(7 - 2*i) for i in range(self.num_conv_layers)]
        tdim_padding = [(i-1)//2 for i in tdim_kernel]

        self.conv_layers = nn.ModuleList()

        if self.input_ndims == 2: #1D
            channel_size = self.num_channels

            for i in range(self.num_conv_layers):
                self.conv_layers.append(
                    nn.Sequential(
                    nn.Conv1d(self.num_channels*in_multipliers[i],
                              self.num_channels*out_multipliers[i],
                              kernel_size=tdim_kernel[i],
                              padding=tdim_padding[i]),
                    nn.BatchNorm1d(self.num_channels*out_multipliers[i]),
                    nn.ReLU()
                    ))

        elif self.input_ndims == 4:  #3D
            initial_img_size = self.input_shape[-2]
            final_img_size = initial_img_size//(2**(self.num_conv_layers))
            channel_size = self.num_channels * (final_img_size**2)

            for i in range(self.num_conv_layers):
                self.conv_layers.append(
                    nn.Sequential(
                    nn.Conv3d(self.num_channels*in_multipliers[i],
                              self.num_channels*out_multipliers[i],
                              kernel_size=(3, 3, tdim_kernel[i]),
                              padding=(1, 1, tdim_padding[i])),
                    nn.BatchNorm3d(self.num_channels*out_multipliers[i]),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=(2, 2, 1)) # Time dimension is not pooled
                    ))
        else:
            # error case, already handled in the base class
            channel_size = None
            pass

        flatten_size = channel_size * out_multipliers[-1] * self.sequence_length
        flatten_size += self.stats_size

        self.fc1 = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
            )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
            )

        self.fc_out = nn.Linear(256, self.output_size)

    def forward(self,
                input_sequences: dict[str, torch.Tensor],
                input_stats: torch.Tensor) -> torch.Tensor:
        s = self._process_sequence(input_sequences)

        for conv in self.conv_layers:
            s = conv(s)

        s = s.flatten(start_dim=1)  # Flatten all dimensions except batch dimension

        if input_stats.numel():
            f = input_stats.flatten(start_dim=1)
            s = torch.hstack([s,f])

        # Apply fully connected layers
        s = self.fc1(s)
        s = self.fc2(s)
        outputs = self.fc_out(s)

        return outputs # Output shape: (batch_size, output_size)

if __name__ == "__main__":
    pass