import torch
import torch.nn as nn


class LSTMSeqToVec(nn.Module):
    def __init__(
        self,
        horizon: int,
        timesteps: int,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # External dropout (recommended even if num_layers=1)
        self.dropout = nn.Dropout(dropout)

        # Fully connected output
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        """
        x shape: (batch_size, timesteps, num_features)
        """

        out, _ = self.lstm(x)

        # Take last timestep output
        out = out[:, -1, :]  # (batch_size, hidden_size)

        out = self.dropout(out)

        out = self.fc(out)

        return out