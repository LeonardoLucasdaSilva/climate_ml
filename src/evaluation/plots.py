import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by model.fit(). Must contain
        'loss' and 'val_loss' inside history.history.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(history.history["loss"], label="train_loss")
    ax.plot(history.history["val_loss"], label="val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    return fig


def plot_real_vs_predicted_scatter(y_true, y_pred):
    """
    Plot a scatter comparison between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predicted values.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Real values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Real vs Predicted")
    ax.grid(True)
    ax.set_aspect("equal")

    return fig


def plot_real_vs_predicted_scatter_no_outliers(
    y_true, y_pred, inf_limit=1, sup_limit=99
):
    """
    Plot predicted vs true values while excluding extreme outliers.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predicted values.
    inf_limit : float, default=1
        Lower percentile limit for filtering.
    sup_limit : float, default=99
        Upper percentile limit for filtering.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    """
    y_true_f = np.asarray(y_true).flatten()
    y_pred_f = np.asarray(y_pred).flatten()

    low, high = np.percentile(y_true_f, [inf_limit, sup_limit])
    mask = (y_true_f >= low) & (y_true_f <= high)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true_f[mask], y_pred_f[mask], s=5, alpha=0.5)
    ax.plot([low, high], [low, high], linestyle="--")

    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title(f"Prediction vs True ({inf_limit}–{sup_limit} percent)")
    ax.grid(True)
    ax.set_aspect("equal")

    return fig

def plot_training_history_torch(history):
    """
    Plots training and validation loss curves.

    Parameters
    ----------
    history : dict
        Dictionary containing:
        - "train_loss": list of floats
        - "val_loss": list of floats

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for saving or further customization.
    """

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig