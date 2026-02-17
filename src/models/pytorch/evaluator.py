import numpy as np
from src.models.pytorch.predict import predict_timeseries_model


def evaluate_model(
    model,
    X,
    y,
    scaler_y=None,
    use_log=False,
):
    y_pred = predict_timeseries_model(model, X)

    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_true = scaler_y.inverse_transform(y.reshape(-1, 1))
    else:
        y_true = y.copy()

    if use_log:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)

    y_pred = np.clip(y_pred, 0.0, None).squeeze()
    y_true = y_true.squeeze()

    return y_true, y_pred