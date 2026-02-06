import tensorflow as tf

def build_lstm_seq_to_vec(horizon, timesteps, num_features, activation = 'tanh', kernel_initializer='glorot_uniform'):
    """
    Builds an LSTM sequence-to-vector model.

    Parameters
    ----------
    horizon : int
        Number of future steps to predict.
    timesteps : int
        Number of past timesteps used as input.
    num_features : int
        Number of features per timestep.
    activation : str
        Activation function for LSTM layers.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM sequence-to-vector model.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation=activation, return_sequences=True, input_shape=(timesteps, num_features)),
        tf.keras.layers.LSTM(48, activation=activation, return_sequences=True),
        tf.keras.layers.LSTM(32, activation=activation, return_sequences=False),
        tf.keras.layers.Dense(horizon, kernel_initializer=kernel_initializer)])
    return model

