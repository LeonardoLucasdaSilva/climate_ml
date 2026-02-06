import numpy as np
from sklearn import preprocessing

def create_sliding_windows(
    series,
    window_size,
    horizon=1,
    multi_step=False
):
    """
    Transform a time series into sliding windows for forecasting.
    """

    if multi_step:
        print("Multi-step not implemented yet")
        return None
    else:

        X,y = [],[]

        m = len(series) - (window_size + horizon)

        for i in range(m+1):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size:i+window_size+horizon])

        return np.array(X), np.array(y)


def prepare_data_seq_to_one(X, y, num_features, scaler_x = None, scaler_y = None):
    """
    Prepare data for sequence-to-one prediction.
    This preparation includes a global scaling to all the sequences/windows

    Parameters
    ----------
    X : ndarray, shape (samples, timesteps, features)
    y : ndarray, shape (samples,) or (samples, 1)
    num_features : int
    scaler_x, scaler_y : MinMaxScaler or None

    Returns
    -------
    X : ndarray, shape (samples, timesteps, features)
    y : ndarray, shape (samples, 1, features)
    scaler_x, scaler_y : fitted scalers
    """

    # Set default values to the scalers
    # Obs: If the default values are set in the params of the function, Python can use the same scaler
    # for multiple instances of the function

    if scaler_x is None:
        scaler_x = preprocessing.MinMaxScaler((0, 1))
    if scaler_y is None:
        scaler_y = preprocessing.MinMaxScaler((0, 1))

    num_sequences = X.shape[0]
    timesteps = X.shape[1]

    # Flatten and scale
    X = scaler_x.fit_transform(X.flatten().reshape(-1, 1))
    y = scaler_y.fit_transform(y.flatten().reshape(-1, 1))

    # Reshape to 3D
    X = X.reshape((num_sequences, timesteps, num_features))
    y = y.reshape((num_sequences, 1, num_features))

    return X, y, scaler_x, scaler_y


def prepare_data_seq_to_seq(X, y, num_features, scaler_x = None, scaler_y = None):
    """
    Prepares data for sequence-to-sequence prediction.
    Scales X and y separately and reshapes into 3D tensors.
    """

    # Set default values to the scalers
    # Obs: If the default values are set in the params of the function, Python can use the same scaler
    # for multiple instances of the function

    if scaler_x is None:
        scaler_x = preprocessing.MinMaxScaler((0, 1))
    if scaler_y is None:
        scaler_y = preprocessing.MinMaxScaler((0, 1))

    num_sequences = X.shape[0]
    timesteps = X.shape[1]
    output_timesteps = y.shape[1]

    # Flatten and scale
    X = scaler_x.fit_transform(X.flatten().reshape(-1, 1))
    y = scaler_y.fit_transform(y.flatten().reshape(-1, 1))

    # Reshape to 3D
    X = X.reshape((num_sequences, timesteps, num_features))
    y = y.reshape((num_sequences, output_timesteps, num_features))

    return X, y, scaler_x, scaler_y



