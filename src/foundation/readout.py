"""Per-subject *readout* head of the foundation model.

A `SubjectReadout` maps the shared core's latent features to a subject's
blood-pressure target (systolic / diastolic / fiducials / waveform, depending on
the configured `OutputMode`). One readout is created per subject; during transfer
the core is frozen and only a fresh readout is trained.
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)


class SubjectReadout(nn.Module):
    def __init__(self,
                 num_features: int,
                 output_size: int,
                 hidden: int = 0,
                 ) -> None:
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(num_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_size),
            )
        else:
            self.net = nn.Linear(num_features, output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, num_features) -> (B, output_size)
        return self.net(features)
