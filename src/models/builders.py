from src.models.pytorch.architectures import LSTMSeqToVec

def lstm_builder(config):
    return LSTMSeqToVec(
        horizon=config["data"]["horizon"],
        timesteps=config["data"]["timesteps"],
        num_features=1,
    )