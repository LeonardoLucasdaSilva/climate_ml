import numpy as np
from src.models.predict import predict_timeseries_model


def evaluate_model(
    model,
    X,
    y,
    scaler_y=None,
    use_log=False,
    debug=False,  # NEW: Add debug flag
):
    y_pred = predict_timeseries_model(model, X)

    # DEBUG: Log predictions before transforms
    if debug:
        print(f"[EVAL DEBUG] Raw y_pred shape: {y_pred.shape}")
        print(f"[EVAL DEBUG] y (targets) shape: {y.shape}")
        print(f"[EVAL DEBUG] y_pred[0:5] (raw): {y_pred[0:5]}")
        print(f"[EVAL DEBUG] y[0:5] (raw targets): {y[0:5]}")

    if scaler_y is not None:
        pred_shape = y_pred.shape
        true_shape = y.shape
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(pred_shape)
        y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).reshape(true_shape)
    else:
        y_true = y.copy()

    if use_log:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)

    y_pred = np.clip(y_pred, 0.0, None)

    # DEBUG: Log after transforms
    if debug:
        yt = y_true.flatten()
        yp = y_pred.flatten()
        print(f"[EVAL DEBUG] After inverse transform:")
        print(f"[EVAL DEBUG] y_true[0:5]: {yt[0:5]}")
        print(f"[EVAL DEBUG] y_pred[0:5]: {yp[0:5]}")
        print(f"[EVAL DEBUG] Correlation check: shift=-1 vs shift=0")
        corr_shift_minus1 = np.corrcoef(yt[1:20], yp[0:19])[0, 1] if len(yt) > 20 else np.nan
        corr_shift_0 = np.corrcoef(yt[0:19], yp[0:19])[0, 1] if len(yt) > 19 else np.nan
        print(f"[EVAL DEBUG] corr at shift=-1: {corr_shift_minus1:.4f}")
        print(f"[EVAL DEBUG] corr at shift=0: {corr_shift_0:.4f}")

    return y_true, y_pred