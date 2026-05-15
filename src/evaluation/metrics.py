import numpy as np

def mape(y_true, y_pred):
    """
    Function to calculate mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Mean absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return np.mean(np.abs(y_true - y_pred))

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

def mae_per_step(y_true, y_pred):
    """MAE for each horizon step. Inputs shape: (n_samples, horizon)."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), axis=0).tolist()


def smape_per_step(y_true, y_pred):
    """SMAPE for each horizon step. Inputs shape: (n_samples, horizon)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return (np.mean(np.abs(y_true - y_pred) / denominator, axis=0) * 100).tolist()


def pearson_correlation(y_true, y_pred):
    """
    Pearson Correlation Coefficient

    Measures the linear correlation between predicted and true values.
    Captures whether the model follows the same trend/shape as the target.

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].
        1 = perfect positive correlation, 0 = no correlation, -1 = inverse
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    numerator = np.sum((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denominator = np.sqrt(
        np.sum((y_true - y_true.mean()) ** 2) * np.sum((y_pred - y_pred.mean()) ** 2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def nse(y_true, y_pred):
    """
    Nash-Sutcliffe Efficiency (NSE)

    Measures how well the model performs compared to simply using the mean
    of the observations. Very common in hydrology and time-series forecasting.

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        NSE score. 1 = perfect, 0 = model is as good as mean baseline,
        negative = model is worse than mean baseline.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_true.mean()) ** 2)

    if denominator == 0:
        return 0.0

    return 1 - (numerator / denominator)


def kge(y_true, y_pred):
    """
    Kling-Gupta Efficiency (KGE)

    Decomposes model performance into three components:
      - r  : Pearson correlation        (timing/shape)
      - alpha: std ratio                (variability)
      - beta : mean ratio               (bias)

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        KGE score. 1 = perfect. Values > -0.41 outperform the mean baseline.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    r = pearson_correlation(y_true, y_pred)
    alpha = y_pred.std() / y_true.std() if y_true.std() != 0 else 0.0
    beta = y_pred.mean() / y_true.mean() if y_true.mean() != 0 else 0.0

    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def dtw_distance(y_true, y_pred):
    """
    Dynamic Time Warping (DTW) Distance

    Measures similarity between two sequences allowing elastic shifts in time.
    Unlike MAE, it tolerates phase offsets in the series.

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        DTW distance. 0 = identical sequences. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n, m = len(y_true), len(y_pred)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(y_true[i - 1] - y_pred[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],     # insertion
                dtw_matrix[i, j - 1],     # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    return dtw_matrix[n, m]