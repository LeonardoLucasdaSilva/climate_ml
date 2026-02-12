import tensorflow as tf

def train_timeseries_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    patience=20,
    epochs=250,
    verbose=0
):
    """
    Compile and train a Keras model using explicit validation data
    and early stopping.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled Keras model.
    X_train, y_train : array-like
        Training data.
    X_val, y_val : array-like
        Validation data.
    patience : int
        Number of epochs with no improvement before early stopping.
    epochs : int
        Maximum number of training epochs.
    verbose : int
        Verbosity level for model.fit.

    Returns
    -------
    history : tf.keras.callbacks.History
        Training history.
    """

    # If validation loss does not decrease for patience epochs, the training early stops

    early_stopping           = tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',
        patience             = patience,
        mode                 = 'min',
        restore_best_weights = True
    )

    # How learning works
    model.compile(
        loss                 = 'mean_absolute_error',
        # adam = adaptative gradient method
        optimizer            = 'adam',
        metrics              = ['mean_squared_error']
    )
    history                  = model.fit(
        X_train,
        y_train,
        epochs               = epochs,
        # Here is good to separate the data manually instead, to get more control of the validation data.
        validation_data     = (X_val, y_val),
        callbacks            = [early_stopping],
        verbose              = verbose
    )
    return history