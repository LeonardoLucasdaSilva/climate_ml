import torch
import torch.nn as nn


class LSTMSeqToVec(nn.Module):
    """Sequence-to-one LSTM. Always predicts a single value.

    The forecast horizon (how many steps ahead) is determined by the training
    data, not by the model architecture. Use direct forecasting: train one
    model per desired horizon value.
    """

    def __init__(
        self,
        timesteps: int,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        multitask: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.multitask = multitask

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

        # Single-value regression head
        self.fc = nn.Linear(hidden_size, 1)

        # Optional classification head (rain / no-rain)
        if multitask:
            self.fc_cls = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, timesteps, num_features)

        Returns
        -------
        If multitask is False:
            torch.Tensor of shape (batch_size, 1)
        If multitask is True:
            tuple (y_reg, y_cls_logits), both shape (batch_size, 1)
        """

        out, _ = self.lstm(x)

        # Take last timestep output
        out = out[:, -1, :]  # (batch_size, hidden_size)

        out = self.dropout(out)

        y_reg = self.fc(out)

        if not self.multitask:
            return y_reg

        y_cls_logits = self.fc_cls(out)
        return y_reg, y_cls_logits

