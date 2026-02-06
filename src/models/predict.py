def predict_next_window(
    model,
    window,
    scaler,
    window_size,
    num_features
):

    """
    Predict the next time step value using a trained time-series model
    and a single sliding window.

    This function performs inference. It takes the most recent window
    of time-series data, reshapes it into the format expected by the model
    (batch_size=1), runs the model prediction, and converts the output
    back to the original scale.

    Parameters
    ----------
    model : tf.keras.Model
        A trained time-series model (e.g., LSTM).
    window : np.ndarray
        Sliding window containing the most recent time steps.
        Shape: (window_size, num_features).
    scaler : sklearn scaler
        Scaler fitted on the training target, used to inverse-transform
        the prediction.
    window_size : int
        Number of time steps in the input window.
    num_features : int
        Number of features per time step.

    Returns
    -------
    float
        Predicted value for the next time step in original units.
    """

    x_predict = window.reshape(1, window_size, num_features)
    predict                      = model.predict(x_predict)
    predict                      = scaler.inverse_transform(predict[0,0].reshape(-1, 1))

    return predict[0,0]