import tensorflow as tf

def train_timeseries_model(model, X, y, patience=20):
    """
    Compile and train a Keras model using early stopping based on validation loss.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled Keras model architecture.
    X : array-like
        Input training data.
    y : array-like
        Target values.
    patience : int, optional
        Number of epochs with no improvement after which training stops.

    Returns
    -------
    history : tf.keras.callbacks.History
        Training history containing loss and metric values per epoch.
    """

    # If validation loss does not decrease for patience epochs, the training early stops

    early_stopping           = tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',
        patience             = patience,
        mode                 = 'min'
    )

    # How learning works
    model.compile(
        loss                 = 'mean_absolute_error',
        # adam = adaptative gradient method
        optimizer            = 'adam',
        metrics              = ['mean_squared_error']
    )
    history                  = model.fit(
        X,
        y,
        epochs               = 250,
        # Here is good to separate the data manually instead, to get more control of the validation data.
        validation_split     = 0.2,
        callbacks            = [early_stopping],
        verbose              = 0
    )
    return history