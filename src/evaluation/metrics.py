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

