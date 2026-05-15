import torch
import torch.nn as nn

def weighted_mse_loss(
    y_pred,
    y_true,
    extreme_quantile=0.9,
    extreme_weight=10.0,
    is_log=False,
    eps=1e-6
):
    """
    Weighted MSE loss that emphasizes extreme rainfall, compatible with both
    log-transformed and raw targets.

    Args:
        y_pred: torch.Tensor, model predictions
        y_true: torch.Tensor, true values
        extreme_quantile: float, quantile to define "extreme rainfall" (default 0.9)
        extreme_weight: float, weight multiplier for extreme rainfall (default 5.0)
        is_log: bool, whether y_true (and y_pred) are log-transformed (default True)
        eps: small number to prevent log(0)
    """
    # Compute threshold in original scale
    if is_log:
        y_true_orig = torch.expm1(y_true)
        y_pred_orig = torch.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    threshold = torch.quantile(y_true_orig, extreme_quantile)

    # Create weights: extreme rainfall gets high weight
    weights = torch.ones_like(y_true)
    weights[y_true_orig > threshold] = extreme_weight

    # Weighted MSE (compute on original scale if is_log)
    if is_log:
        # compute loss in log space (optional: can use pred vs true in log)
        loss = torch.mean(weights * (y_pred - y_true) ** 2)
    else:
        loss = torch.mean(weights * (y_pred - y_true) ** 2)

    return loss

class QuantileLoss(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.maximum(
            self.q * errors,
            (self.q - 1) * errors
        )
        return torch.mean(loss)

class MultitaskLoss(nn.Module):
    """
    Combines a regression loss (precipitation amount) with a binary
    classification loss (rain / no-rain) to improve spike detection.

    Args:
        regression_loss: nn.Module  — e.g. QuantileLoss or nn.MSELoss
        cls_weight:      float      — weight of the classification loss
        rain_threshold:  float      — mm threshold to define "rain"
        is_log:          bool       — whether targets are log-transformed
    """

    def __init__(
        self,
        regression_loss: nn.Module,
        cls_weight: float = 0.3,
        rain_threshold: float = 0.1,
        is_log: bool = False,
    ):
        super().__init__()
        self.regression_loss = regression_loss
        self.cls_weight = cls_weight
        self.rain_threshold = rain_threshold
        self.is_log = is_log
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        y_pred_reg: torch.Tensor,   # (B, horizon)
        y_pred_cls: torch.Tensor,   # (B, horizon) — raw logits
        y_true: torch.Tensor,       # (B, horizon)
    ):
        reg_loss = self.regression_loss(y_pred_reg, y_true)

        y_orig = torch.expm1(y_true) if self.is_log else y_true
        rain_label = (y_orig > self.rain_threshold).float()

        cls_loss = self.bce(y_pred_cls, rain_label)

        total = reg_loss + self.cls_weight * cls_loss
        return total, reg_loss.detach(), cls_loss.detach()