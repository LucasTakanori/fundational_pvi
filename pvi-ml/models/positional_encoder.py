import torch
import torch.nn as nn


class ResidualRecurrentPositionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        recurrent_type: str = "LSTM",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if recurrent_type.lower() == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )

        elif recurrent_type.lower() == "gru":
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )

        else:  # vanilla rnn
            self.encoder = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # map to input_size so we can use residual
        self.fc2 = nn.Linear(4 * hidden_size, input_size)

    def forward(self, x):
        # res = x # will this be overridden?
        res = torch.clone(x)  # how does this work?

        x = self.encoder(x)[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = res + x

        return x


class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class LearnablePositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
