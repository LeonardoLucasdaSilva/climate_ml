import torch
import torch.nn as nn


class LSTMSeqToVec(nn.Module):
    def __init__(self, horizon, timesteps, num_features):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=64,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=48,
            batch_first=True
        )

        self.lstm3 = nn.LSTM(
            input_size=48,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Linear(32, horizon)

    def forward(self, x):
        # x: (batch, timesteps, num_features)

        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)

        # Take last time step
        out = out[:, -1, :]  # (batch, 32)

        out = self.fc(out)

        return out