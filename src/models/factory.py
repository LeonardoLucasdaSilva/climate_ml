import torch
from src.models.losses import weighted_mse_loss
from src.models.losses import QuantileLoss, MultitaskLoss


def _is_multitask(config: dict) -> bool:
    return bool(config.get("training", {}).get("multitask", False))


def build_model_and_loss(model_builder, config, num_features):

    model = model_builder(config, num_features)

    loss_name = config["training"].get("loss", "mse")
    use_log = config.get("preprocessing", {}).get("use_log", False)

    # ------------------------
    # Base regression loss
    # ------------------------
    if loss_name == "mse":
        base_criterion = torch.nn.MSELoss()

    elif loss_name == "huber":
        base_criterion = torch.nn.SmoothL1Loss()

    elif loss_name == "mae":
        base_criterion = torch.nn.L1Loss()

    elif loss_name == "weighted_mse":
        base_criterion = lambda yp, yt: weighted_mse_loss(
            yp, yt, extreme_weight=5.0, is_log=use_log
        )

    elif loss_name.startswith("quantile"):
        # example: "quantile_0.9"
        q = float(loss_name.split("_")[1])
        base_criterion = QuantileLoss(q=q)

    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    # ------------------------
    # Optional multitask wrapper
    # ------------------------
    if _is_multitask(config):
        mt_cfg = config.get("training", {})
        cls_weight = mt_cfg.get("multitask_cls_weight", 0.3)
        rain_threshold = mt_cfg.get("rain_threshold", 0.1)

        criterion = MultitaskLoss(
            regression_loss=base_criterion,
            cls_weight=cls_weight,
            rain_threshold=rain_threshold,
            is_log=use_log,
        )
    else:
        criterion = base_criterion

    return model, criterion