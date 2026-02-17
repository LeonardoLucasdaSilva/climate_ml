import torch
from src.models.pytorch.losses import weighted_mse_loss


def build_model_and_loss(model_builder, config, num_features):

    model = model_builder(config, num_features)

    loss_name = config["training"].get("loss", "mse")
    use_log = config.get("preprocessing", {}).get("use_log", False)

    if loss_name == "mse":
        criterion = torch.nn.MSELoss()

    elif loss_name == "huber":
        criterion = torch.nn.SmoothL1Loss()

    elif loss_name == "mae":
        criterion = torch.nn.L1Loss()

    elif loss_name == "weighted_mse":
        criterion = lambda yp, yt: weighted_mse_loss(
            yp, yt, extreme_weight=5.0, is_log=use_log
        )

    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return model, criterion