import numpy as np

def mape(y_true, y_pred):
    """
    Function to calculate mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def abs_error(y_true, y_pred):
    """
    Function to calculate the absolute error for each instant
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred)/y_true)*100


