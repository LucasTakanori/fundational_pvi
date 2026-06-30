"""The shared *core* backbone of the foundation model.

`PviCore` is the subject-agnostic feature extractor: it maps a flattened PVI
feature vector to a fixed-size latent representation that every per-subject
readout consumes. It mirrors the MLP trunk in
`src/models/mlp_models.py::PviMLP` (everything up to, but not including, the
final output layer), kept as a standalone module so it can be pretrained once
and reused/frozen across subjects.
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)


class PviCore(nn.Module):
    def __init__(self,
                 flatten_size: int,
                 num_features: int = 200,
                 num_hidden_layers: int = 4,
                 ) -> None:
        super().__init__()
        self.flatten_size = int(flatten_size)
        self.num_features = int(num_features)

        self.fc_in = nn.Sequential(
            nn.Linear(self.flatten_size, self.num_features),
            nn.ReLU(),
        )
        self.hidden = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
            )
            for _ in range(int(num_hidden_layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, flatten_size) -> (B, num_features)
        x = self.fc_in(x)
        for layer in self.hidden:
            x = layer(x)
        return x
