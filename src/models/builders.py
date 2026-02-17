from src.models.pytorch.architectures import LSTMSeqToVec

def lstm_builder(config, num_features):
    return LSTMSeqToVec(
        horizon=config["data"]["horizon"],
        timesteps=config["data"]["timesteps"],
        num_features=num_features,
    )