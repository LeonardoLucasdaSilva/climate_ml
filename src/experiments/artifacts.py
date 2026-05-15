from pathlib import Path
import torch

import numpy as np

from src.evaluation.plots import (
    plot_real_vs_predicted_scatter,
    plot_training_history_torch,
    plot_real_vs_predicted_timeseries,
    plot_real_and_predicted_separate,
    plot_error_histogram,
    plot_absolute_error_timeseries,
    plot_multistep_horizon_timeseries,
)
from src.evaluation.metrics import mae, smape, mae_per_step, smape_per_step
from src.utils.files import save_plot, save_json, ensure_dir


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
    global_debug_dir: Path | None = None,
    test_dates=None,
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
    # SAVE RAW TEST PREDICTIONS (station-level NPZ)
    # ======================================================
    # These are saved under the station's base_dir (which may be a per-station
    # folder in standard mode or a shared GLOBAL_DEBUG_* folder in global_debug).
    preds_dir = base_dir / "predictions"
    ensure_dir(preds_dir)

    import numpy as np
    # Ensure independent copies to avoid accidental aliasing across runs/stations
    y_true_np = np.asarray(y_true_test).copy()
    y_pred_np = np.asarray(y_pred_test).copy()
    y_ref_np = np.asarray(y_inmet_test).copy() if y_inmet_test is not None else np.array([])

    # Lightweight configuration fingerprint for traceability
    import json, hashlib
    try:
        cfg_text = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        config_id = hashlib.sha1(cfg_text).hexdigest()[:10]
    except Exception:
        config_id = "unknown"

    def _save_npz(dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        npz_data = dict(
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_ref=y_ref_np,
            source=source,
            run_name=run_name,
            config_id=config_id,
        )
        if test_dates is not None:
            npz_data["dates"] = np.asarray(test_dates, dtype="U10")  # ISO strings YYYY-MM-DD
        np.savez_compressed(
            dest_dir / f"{file_prefix}_test_predictions.npz",
            **npz_data,
        )

    _save_npz(preds_dir)
    if global_debug_dir is not None:
        _save_npz(global_debug_dir / "predictions")

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
            global_debug_dir=global_debug_dir,
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
            global_debug_dir=global_debug_dir,
        )

    # ======================================================
    # METRICS
    # ======================================================
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    y_flat_true = np.asarray(y_true_test).flatten()
    y_flat_pred = np.asarray(y_pred_test).flatten()
    horizon = np.asarray(y_pred_test).shape[1] if np.asarray(y_pred_test).ndim > 1 else 1

    metrics = {
        "val_loss": float(val_loss),
        "mae": float(mae(y_flat_true, y_flat_pred)),
        "smape": float(smape(y_flat_true, y_flat_pred)),
    }

    if horizon > 1:
        metrics["mae_per_step"] = mae_per_step(y_true_test, y_pred_test)
        metrics["smape_per_step"] = smape_per_step(y_true_test, y_pred_test)

    save_json(metrics, metrics_dir / f"{file_prefix}_metrics.json")

    # Also return the core test predictions so the caller (run_all_stations)
    # can optionally build a run-level NPZ next to summary_metrics.json.
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
    global_debug_dir: Path | None = None,
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
    global_debug_dir : Path or None
        When set, test-split plots are also saved here:
        horizon=1  → global_debug_dir/timeseries/
        horizon>1  → global_debug_dir/multistep_timeseries/
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    horizon = y_true.shape[1] if y_true.ndim > 1 else 1

    is_test = split_name == "test"

    # Create base split directory
    split_dir = base_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # MULTI-STEP: per-horizon-step timeseries plot
    # ======================================================
    if horizon > 1:
        ms_dir = split_dir / "multistep_timeseries"
        ms_dir.mkdir(parents=True, exist_ok=True)
        fig_ms = plot_multistep_horizon_timeseries(
            y_true,
            y_pred,
            title=f"{split_name.capitalize()} - Multi-Step Forecast - {plot_id}",
            metadata=metadata,
        )
        if is_test and global_debug_dir is not None:
            gd = global_debug_dir / "multistep_timeseries"
            gd.mkdir(parents=True, exist_ok=True)
            fig_ms.savefig(gd / f"{file_prefix}_multistep_timeseries.png", dpi=300, bbox_inches="tight")
        save_plot(fig_ms, ms_dir / f"{file_prefix}_multistep_timeseries.png")

    # Flatten for the remaining single-series plots
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_ref_flat = np.asarray(y_ref).flatten() if y_ref is not None else None

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
            y_ref_flat,
            title=f"{split_name.capitalize()} - Real vs Predicted",
            metadata=metadata,
            main_label=main_label,
            ref_label=ref_label,
        )

    if is_test and global_debug_dir is not None and horizon == 1:
        gd = global_debug_dir / "timeseries"
        gd.mkdir(parents=True, exist_ok=True)
        fig_ts.savefig(gd / f"{file_prefix}_timeseries.png", dpi=300, bbox_inches="tight")
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
        y_ref_flat,
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