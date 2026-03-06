from pathlib import Path
import torch

from src.evaluation.plots import (
    plot_real_vs_predicted_scatter,
    plot_training_history_torch,
    plot_real_vs_predicted_timeseries,
    plot_real_and_predicted_separate,
    plot_error_histogram,
    plot_absolute_error_timeseries
)
from src.evaluation.metrics import mae, smape
from src.utils.files import save_plot, save_json


def save_station_artifacts(
    cidade: str,
    run_name: str,
    base_dir: Path,
    history,
    model,
    val_loss: float,
    y_true_val,
    y_pred_val,
    y_true_test,
    y_pred_test,
    y_inmet_test,
    metadata: dict,
    config: dict,
):
    """Saves model, plots and metrics for a single station.

    Notes
    -----
    ``y_true_*`` always refers to the *main* dataset defined by
    ``config["data"]["source"]`` (ERA5 or INMET). ``y_inmet_test`` is kept
    for backward compatibility and represents the *reference* test series
    (INMET when source=ERA5, ERA5 when source=INMET).
    """

    base_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{cidade}_{run_name}" if run_name else cidade
    plot_id = f"{cidade} - {run_name}" if run_name else cidade

    source = config["data"].get("source", "era5").lower()

    # ======================================================
    # SAVE MODEL
    # ======================================================
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "station": cidade,
            "val_loss": float(val_loss),
        },
        model_dir / f"{file_prefix}_model.pt",
    )

    # ======================================================
    # TRAINING HISTORY
    # ======================================================
    history_dir = base_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    fig_hist = plot_training_history_torch(history, identifier=plot_id)
    save_plot(fig_hist, history_dir / f"{file_prefix}_history.png")

    # ======================================================
    # VALIDATION PLOTS (no external reference)
    # ======================================================
    _save_prediction_plots(
        y_true_val,
        y_pred_val,
        "val",
        plot_id,
        file_prefix,
        base_dir,
        metadata,
        source=source,
        y_ref=None,
    )

    # ======================================================
    # TEST PLOTS
    # ======================================================
    # For ERA5 runs, y_true_test = ERA5 and y_inmet_test = INMET reference.
    # For INMET runs, y_true_test = INMET and y_inmet_test = ERA5 reference.
    # Requirement: INMET tests should only show Prediction + INMET in the
    # main timeseries plot, while ERA5 tests keep the triple plot.
    if source == "era5":
        # main = ERA5, ref = INMET (if available)
        _save_prediction_plots(
            y_true_test,
            y_pred_test,
            "test",
            plot_id,
            file_prefix,
            base_dir,
            metadata,
            source=source,
            y_ref=y_inmet_test,
        )
    else:
        # source == "inmet": main = INMET, ref = ERA5 (optional)
        # We pass y_ref so that secondary/separate plots can still use it,
        # but the main overlay plot will only draw main+prediction.
        _save_prediction_plots(
            y_true_test,
            y_pred_test,
            "test",
            plot_id,
            file_prefix,
            base_dir,
            metadata,
            source=source,
            y_ref=y_inmet_test,
        )

    # ======================================================
    # METRICS
    # ======================================================
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "val_loss": float(val_loss),
        "mae": float(mae(y_true_test, y_pred_test)),
        "smape": float(smape(y_true_test, y_pred_test)),
    }

    save_json(metrics, metrics_dir / f"{file_prefix}_metrics.json")

    return metrics


# ==========================================================
# INTERNAL HELPER
# ==========================================================

def _save_prediction_plots(
    y_true,
    y_pred,
    split_name,
    plot_id,
    file_prefix,
    base_dir,
    metadata,
    source: str = "era5",
    y_ref=None,
):
    """Saves scatter + timeseries + separate plots inside split folders.

    Parameters
    ----------
    source : {"era5", "inmet"}
        Main data source. Controls labeling and whether the optional
        reference series is drawn in the combined timeseries plot.
    y_ref : array-like or None
        Optional secondary reference series. For ERA5 runs this is INMET,
        for INMET runs this is ERA5.
    """

    # Create base split directory
    split_dir = base_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # SCATTER
    # ======================================================
    scatter_dir = split_dir / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    fig_scatter = plot_real_vs_predicted_scatter(
        y_true,
        y_pred,
        identifier=plot_id,
    )

    save_plot(
        fig_scatter,
        scatter_dir / f"{file_prefix}_scatter.png",
    )

    # ======================================================
    # TIMESERIES (combined)
    # ======================================================
    ts_dir = split_dir / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Decide labels based on source
    if source == "era5":
        main_label = "ERA5"
        ref_label = "INMET"
    else:
        main_label = "INMET"
        ref_label = "ERA5"

    # For INMET runs, requirement is to show only Prediction + INMET in
    # the main timeseries plot; we simply do not pass the ref series.
    if split_name == "test" and source == "inmet":
        fig_ts = plot_real_vs_predicted_timeseries(
            y_true,
            y_pred,
            title=f"{split_name.capitalize()} - Real vs Predicted",
            metadata=metadata,
            main_label=main_label,
        )
    else:
        fig_ts = plot_real_vs_predicted_timeseries(
            y_true,
            y_pred,
            y_ref,
            title=f"{split_name.capitalize()} - Real vs Predicted",
            metadata=metadata,
            main_label=main_label,
            ref_label=ref_label,
        )

    save_plot(
        fig_ts,
        ts_dir / f"{file_prefix}_timeseries.png",
    )

    # ======================================================
    # SEPARATE TIMESERIES
    # ======================================================
    ts_sep_dir = split_dir / "timeseries_separate"
    ts_sep_dir.mkdir(parents=True, exist_ok=True)

    fig_sep = plot_real_and_predicted_separate(
        y_true,
        y_pred,
        y_ref,
        title=f"{split_name.capitalize()} - Real and Predicted (Separate)",
        metadata=metadata,
        main_label=main_label,
        ref_label=ref_label,
    )

    save_plot(
        fig_sep,
        ts_sep_dir / f"{file_prefix}_timeseries_separate.png",
    )

    # ======================================================
    # ERROR HISTOGRAM
    # ======================================================
    error_dir = split_dir / "error_histogram"
    error_dir.mkdir(parents=True, exist_ok=True)

    fig_error = plot_error_histogram(
        y_true,
        y_pred,
        title=f"{split_name.capitalize()} - Error Distribution",
    )

    save_plot(
        fig_error,
        error_dir / f"{file_prefix}_error_histogram.png",
    )

    # ======================================================
    # ABS ERROR TIMESERIES
    # ======================================================
    error_dir = split_dir / "abs_error_timeseries"
    error_dir.mkdir(parents=True, exist_ok=True)

    fig_error = plot_absolute_error_timeseries(
        y_true,
        y_pred,
        title=f"{split_name.capitalize()} - Absolute Error Timeseries",
    )

    save_plot(
        fig_error,
        error_dir / f"{file_prefix}_abs_error_timeseries.png",
    )