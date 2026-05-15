from src.models.architectures import LSTMSeqToVec


def lstm_builder(config, num_features):
    multitask = config.get("training", {}).get("multitask", False)

    return LSTMSeqToVec(
        timesteps=config["data"]["timesteps"],
        num_features=num_features,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        multitask=multitask,
    )