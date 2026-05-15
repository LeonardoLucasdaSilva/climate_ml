import numpy as np
from sklearn import preprocessing

def create_sliding_windows(
    X,
    y,
    window_size,
    horizon=1,
):
    """
    Transform multivariate time series into sliding windows for direct forecasting.

    Each window of `window_size` past timesteps is paired with a single target
    value located `horizon` steps after the window end (direct multi-step):
      - horizon=1 → predict y[t+1]   (next day)
      - horizon=2 → predict y[t+2]   (2 days ahead)
      - horizon=3 → predict y[t+3]   (3 days ahead)

    Train one model per horizon value to obtain multi-step forecasts.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,)
        Target variable.
    window_size : int
        Number of past timesteps used as input.
    horizon : int
        How many steps ahead to predict (≥ 1).

    Returns
    -------
    X_windows : ndarray of shape (n_windows, window_size, n_features)
    y_windows : ndarray of shape (n_windows, 1)
    """
    X_windows, y_windows = [], []

    m = len(X) - (window_size + horizon)

    for i in range(m + 1):
        X_windows.append(X[i : i + window_size])
        y_windows.append([y[i + window_size + horizon - 1]])

    return np.array(X_windows), np.array(y_windows)


def prepare_data_seq_to_one(X, y, num_features, scaler_x=None, scaler_y=None):
    """
    Prepare multivariate sequence-to-one data with proper feature-wise scaling.

    Parameters
    ----------
    X : ndarray, shape (samples, timesteps, features)
    y : ndarray, shape (samples,) or (samples, 1)
    num_features : int
    scaler_x, scaler_y : MinMaxScaler or None

    Returns
    -------
    X_scaled : ndarray, shape (samples, timesteps, features)
    y_scaled : ndarray, shape (samples, 1)
    scaler_x, scaler_y : fitted scalers
    """

    if scaler_x is None:
        scaler_x = preprocessing.MinMaxScaler((0, 1))

    if scaler_y is None:
        scaler_y = preprocessing.MinMaxScaler((0, 1))

    samples, timesteps, features = X.shape

    # Reshape to 2D preserving feature dimension
    X_2d = X.reshape(-1, features)

    # Scale each feature independently
    X_scaled_2d = scaler_x.fit_transform(X_2d)

    # Reshape back to 3D
    X_scaled = X_scaled_2d.reshape(samples, timesteps, features)

    # Scale target separately — works for both (n, 1) and (n, horizon)
    y_shape = y.shape
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y_shape)

    return X_scaled, y_scaled, scaler_x, scaler_y


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



