import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates


def _build_dates_from_metadata(metadata: dict, title: str, length: int):
    """
    Extract correct date range (Validation or Test) from metadata
    and build a DatetimeIndex matching the series length.
    """

    if metadata is None:
        return None

    # Decide which range to use based on title
    if "Validation" in title and "Validation" in metadata:
        range_str = metadata["Validation"]
    elif "Test" in title and "Test" in metadata:
        range_str = metadata["Test"]
    else:
        return None  # fallback to index

    # Extract start date (before arrow)
    # Example format:
    # "2020-01-01 → 2020-12-31 (365 samples)"
    start_date = range_str.split("→")[0].strip()

    return pd.date_range(
        start=pd.to_datetime(start_date),
        periods=length,
        freq="D",
    )

def plot_training_history(history, identifier: str = None):
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by model.fit(). Must contain
        'loss' and 'val_loss' inside history.history.
    identifier : str, optional
        String to identify the plot, appended to the title.

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

    if identifier:
        ax.set_title(f"Training History - {identifier}")

    return fig


def plot_real_vs_predicted_scatter(y_true, y_pred, identifier: str = None):
    """
    Plot a scatter comparison between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predicted values.
    identifier : str, optional
        String to identify the plot, appended to the title.

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
    title = "Real vs Predicted"
    if identifier:
        title += f" - {identifier}"
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal")

    return fig


def plot_real_vs_predicted_scatter_no_outliers(y_true, y_pred, inf_limit=1, sup_limit=99, identifier: str = None):
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
    identifier : str, optional
        String to identify the plot, appended to the title.

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
    title = f"Prediction vs True ({inf_limit}–{sup_limit} percent)"
    if identifier:
        title += f" - {identifier}"
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal")

    return fig


def plot_training_history_torch(history, identifier: str = None):
    """
    Plots training and validation loss curves.

    Parameters
    ----------
    history : dict
        Dictionary containing:
        - "train_loss": list of floats
        - "val_loss": list of floats
    identifier : str, optional
        String to identify the plot, appended to the title.

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
    title = "Training and Validation Loss"
    if identifier:
        title += f" - {identifier}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def save_table_as_image(df, save_path, dpi=300, identifier: str = None):
    """
    Saves a pandas DataFrame as an image using matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    save_path : Path or str
        Where to save the image
    dpi : int
        Image resolution
    identifier : str, optional
        String to add to the title of the table (optional)
    """
    fig, ax = plt.subplots(figsize=(df.shape[1] * 1.5, df.shape[0] * 0.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    if identifier:
        ax.set_title(identifier)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_real_vs_predicted_timeseries(
    y_main: np.ndarray,
    y_pred: np.ndarray,
    y_ref: np.ndarray | None = None,
    title: str = "Prediction vs Reference",
    metadata: dict | None = None,
    main_label: str = "Main",
    ref_label: str = "Reference",
):
    """Plot prediction, main reference series and optional secondary reference.

    Parameters
    ----------
    y_main : np.ndarray
        Main ground-truth series (e.g. ERA5 for ERA5 runs, INMET for INMET runs).
    y_pred : np.ndarray
        Model predictions in the same space as ``y_main``.
    y_ref : np.ndarray | None, optional
        Optional secondary reference series (e.g. INMET for ERA5 runs, ERA5 for
        INMET runs). When ``None``, only ``y_main`` and ``y_pred`` are plotted.
    title : str, optional
        Plot title.
    metadata : dict | None, optional
        Metadata used to reconstruct dates.
    main_label : str, optional
        Label for the main ground-truth series.
    ref_label : str, optional
        Label for the secondary reference series.
    """

    dates = _build_dates_from_metadata(metadata, title, len(y_main))

    fig, ax = plt.subplots(figsize=(12, 5))

    if dates is not None:
        ax.plot(dates, y_main, label=main_label, linewidth=2)
        ax.plot(
            dates,
            y_pred,
            label="Prediction",
            linewidth=2,
            linestyle="--",
        )

        if y_ref is not None:
            ax.plot(
                dates,
                y_ref,
                label=ref_label,
                linewidth=2,
                alpha=0.8,
            )

        ax.set_xlabel("Date")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    else:
        ax.plot(y_main, label=main_label, linewidth=2)
        ax.plot(y_pred, label="Prediction", linewidth=2, linestyle="--")

        if y_ref is not None:
            ax.plot(y_ref, label=ref_label, linewidth=2, alpha=0.8)

        ax.set_xlabel("Time Step")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if metadata:
        meta_text = " | ".join(f"{k}: {v}" for k, v in metadata.items())
        fig.text(
            0.5,
            0.93,
            meta_text,
            ha="center",
            fontsize=9,
            alpha=0.8,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig


def plot_real_and_predicted_separate(
    y_main: np.ndarray,
    y_pred: np.ndarray,
    y_ref: np.ndarray | None = None,
    title: str = "Prediction vs Reference",
    metadata: dict | None = None,
    main_label: str = "Main",
    ref_label: str = "Reference",
):
    """Plot main, prediction and optional reference in stacked subplots."""

    dates = _build_dates_from_metadata(metadata, title, len(y_main))

    n_plots = 3 if y_ref is not None else 2

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(12, 3.5 * n_plots),
        sharex=True,
    )

    if n_plots == 2:
        ax_main, ax_pred = axes
    else:
        ax_main, ax_pred, ax_ref = axes

    # --- MAIN ---
    if dates is not None:
        ax_main.plot(dates, y_main, label=main_label, linewidth=2)
        ax_main.set_ylabel(main_label)
    else:
        ax_main.plot(y_main, label=main_label, linewidth=2)
        ax_main.set_ylabel(main_label)

    ax_main.set_title(f"{title} - {main_label}")
    ax_main.grid(True, alpha=0.3)

    # --- PRED ---
    if dates is not None:
        ax_pred.plot(
            dates,
            y_pred,
            label="Prediction",
            linewidth=2,
            linestyle="--",
        )
    else:
        ax_pred.plot(y_pred, label="Prediction", linewidth=2, linestyle="--")

    ax_pred.set_title(f"{title} - Prediction")
    ax_pred.grid(True, alpha=0.3)

    # --- REF (optional) ---
    if y_ref is not None:
        if dates is not None:
            ax_ref.plot(dates, y_ref, label=ref_label, linewidth=2)
        else:
            ax_ref.plot(y_ref, label=ref_label, linewidth=2)

        ax_ref.set_title(f"{title} - {ref_label}")
        ax_ref.grid(True, alpha=0.3)

    if dates is not None:
        for ax in (axes if isinstance(axes, np.ndarray) else [axes]):
            ax.set_xlabel("Date")
        fig.autofmt_xdate()
    else:
        for ax in (axes if isinstance(axes, np.ndarray) else [axes]):
            ax.set_xlabel("Time Step")

    fig.tight_layout()

    return fig


def plot_error_histogram(
    y_true,
    y_pred,
    title: str = "Error Distribution",
    bins: int = 50,
):
    """
    Plots histogram of prediction errors (y_true - y_pred).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    title : str
        Title of the plot.
    bins : int
        Number of histogram bins.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(errors, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(0, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Error (True - Predicted)")
    ax.set_ylabel("Frequency")

    ax.grid(alpha=0.3)

    return fig

import numpy as np
import matplotlib.pyplot as plt


def plot_absolute_error_timeseries(
    y_true,
    y_pred,
    title: str = "Absolute Error Over Time",
    metadata: dict | None = None,
):
    """
    Plots absolute error |y_true - y_pred| as a time series.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    title : str
    metadata : dict (optional)
        Can contain date index under key 'dates'
    """

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    abs_error = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(12, 5))

    if metadata and "dates" in metadata:
        x_axis = metadata["dates"]
    else:
        x_axis = np.arange(len(abs_error))

    ax.plot(x_axis, abs_error)

    ax.set_title(title)
    ax.set_ylabel("Absolute Error")
    ax.set_xlabel("Time")

    ax.grid(alpha=0.3)

    return fig