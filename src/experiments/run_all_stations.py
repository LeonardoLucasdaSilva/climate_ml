from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch
import time

from src.data import load_interim
from src.data.preprocess import create_sliding_windows, prepare_data_seq_to_one
from src.data.split import temporal_train_val_test_split
from src.evaluation.metrics import mae, smape
from src.models.pytorch.train import train_regression_model
from src.models.pytorch.predict import predict_timeseries_model
from src.evaluation.plots import (
    plot_real_vs_predicted_scatter,
    plot_training_history_torch,
    plot_real_vs_predicted_timeseries,
    plot_real_and_predicted_separate,
)
from src.utils.files import save_json, save_plot
from src.config.paths import DATA_DIR, PROJECT_ROOT
from src.models.pytorch.losses import weighted_mse_loss


# ==========================================================
# SINGLE STATION
# ==========================================================

def run_single_station(
    cidade: str,
    model_builder,
    config: dict,
    run_dir: Path,
    base_dir: Path,
):
    base_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # Identifiers
    # ----------------------------------
    run_name = config["experiment"].get("run_name", "")
    plot_id = f"{cidade} - {run_name}" if run_name else cidade
    file_prefix = f"{cidade}_{run_name}" if run_name else cidade

    metadata = {
        "Timesteps": config["data"]["timesteps"],
        "Loss": config["training"]["loss"],
        "Scaler": config["preprocessing"]["use_scaler"],
        "Range": config["data"]["initial_date"]
                 + "-" +
                 config["data"]["end_date"],
    }

    # ----------------------------------
    # LOAD DATA
    # ----------------------------------
    path = load_interim(f"era5_precipitation_timeseries_{cidade}_1D.nc")
    nc_dataset = xr.open_dataset(path, engine="netcdf4")
    tp = nc_dataset["tp"]
    df = tp.to_dataframe(name="total_precipitation")

    series = df.loc[
        config["data"]["initial_date"]:
        config["data"]["end_date"]
    ]["total_precipitation"]

    # ----------------------------------
    # PREPROCESS
    # ----------------------------------
    X, y = create_sliding_windows(
        series,
        config["data"]["timesteps"],
        config["data"]["horizon"],
    )

    use_log = config.get("preprocessing", {}).get("use_log", False)
    use_scaler = config.get("preprocessing", {}).get("use_scaler", True)

    if use_log:
        X_proc = np.log1p(X)
        y_proc = np.log1p(y)
    else:
        X_proc = X
        y_proc = y

    if use_scaler:
        X_scaled, y_scaled, scaler_x, scaler_y = prepare_data_seq_to_one(
            X_proc, y_proc, num_features=1
        )
    else:
        X_scaled = X_proc.reshape(X_proc.shape[0], X_proc.shape[1], 1)
        y_scaled = y_proc.reshape(-1, 1)
        scaler_x, scaler_y = None, None

    X_train, X_val, X_test, y_train, y_val, y_test = (
        temporal_train_val_test_split(
            X_scaled,
            y_scaled,
            config["data"]["train_split"],
            config["data"]["val_split"],
        )
    )

    # ----------------------------------
    # MODEL
    # ----------------------------------
    model = model_builder(config)

    loss_name = config["training"].get("loss", "mse")

    if loss_name == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_name == "huber":
        criterion = torch.nn.SmoothL1Loss()
    elif loss_name == "mae":
        criterion = torch.nn.L1Loss()
    elif loss_name == "weighted_mse":
        criterion = lambda yp, yt: weighted_mse_loss(
            yp, yt, extreme_weight=5.0, is_log=use_log
        )
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    # ----------------------------------
    # TRAIN
    # ----------------------------------
    history, val_loss = train_regression_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        criterion=criterion,
    )

    # ----------------------------------
    # SAVE MODEL
    # ----------------------------------
    model_path = base_dir / "models" / f"{file_prefix}_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "station": cidade,
            "val_loss": float(val_loss),
        },
        model_path,
    )

    # ----------------------------------
    # HISTORY PLOT
    # ----------------------------------
    fig = plot_training_history_torch(history, identifier=plot_id)
    save_plot(
        fig,
        base_dir / "history" / f"{file_prefix}_history.png"
    )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    y_pred_val = predict_timeseries_model(model, X_val)

    if use_scaler and scaler_y is not None:
        y_pred_val = scaler_y.inverse_transform(
            y_pred_val.reshape(-1, 1)
        )
        y_true_val = scaler_y.inverse_transform(
            y_val.reshape(-1, 1)
        )
    else:
        y_true_val = y_val.copy()

    if use_log:
        y_pred_val = np.expm1(y_pred_val)
        y_true_val = np.expm1(y_true_val)

    y_pred_val = np.clip(y_pred_val, 0.0, None).squeeze()
    y_true_val = y_true_val.squeeze()

    fig_val = plot_real_vs_predicted_scatter(
        y_true_val, y_pred_val, identifier=plot_id
    )
    save_plot(
        fig_val,
        base_dir / "scatter" / f"{file_prefix}_val_scatter.png"
    )

    fig_val_ts = plot_real_vs_predicted_timeseries(
        y_true_val,
        y_pred_val,
        title="Validation - Real vs Predicted",
        metadata=metadata,
    )
    save_plot(
        fig_val_ts,
        base_dir / "timeseries" / f"{file_prefix}_val_timeseries.png"
    )

    fig_val_ts_sep = plot_real_and_predicted_separate(
        y_true_val,
        y_pred_val,
        title="Validation - Real and Predicted (Separate)",
        metadata=metadata,
    )
    save_plot(
        fig_val_ts_sep,
        base_dir / "timeseries_separate"
        / f"{file_prefix}_val_timeseries_separate.png"
    )

    # ==========================================================
    # TEST
    # ==========================================================

    y_pred_test = predict_timeseries_model(model, X_test)

    if use_scaler and scaler_y is not None:
        y_pred_test = scaler_y.inverse_transform(
            y_pred_test.reshape(-1, 1)
        )
        y_true_test = scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        )
    else:
        y_true_test = y_test.copy()

    if use_log:
        y_pred_test = np.expm1(y_pred_test)
        y_true_test = np.expm1(y_true_test)

    y_pred_test = np.clip(y_pred_test, 0.0, None).squeeze()
    y_true_test = y_true_test.squeeze()

    fig_test = plot_real_vs_predicted_scatter(
        y_true_test, y_pred_test, identifier=plot_id
    )
    save_plot(
        fig_test,
        base_dir / "scatter" / f"{file_prefix}_test_scatter.png"
    )

    fig_test_ts = plot_real_vs_predicted_timeseries(
        y_true_test,
        y_pred_test,
        title="Test - Real vs Predicted",
        metadata=metadata,
    )
    save_plot(
        fig_test_ts,
        base_dir / "timeseries" / f"{file_prefix}_test_timeseries.png"
    )

    fig_test_ts_sep = plot_real_and_predicted_separate(
        y_true_test,
        y_pred_test,
        title="Test - Real and Predicted (Separate)",
        metadata=metadata,
    )
    save_plot(
        fig_test_ts_sep,
        base_dir / "timeseries_separate"
        / f"{file_prefix}_test_timeseries_separate.png"
    )

    # ----------------------------------
    # METRICS
    # ----------------------------------
    metrics = {
        "val_loss": float(val_loss),
        "mae": float(mae(y_true_test, y_pred_test)),
        "smape": float(smape(y_true_test, y_pred_test)),
    }

    save_json(
        metrics,
        base_dir / "metrics" / f"{file_prefix}_metrics.json"
    )

    return metrics


# ==========================================================
# ALL STATIONS
# ==========================================================

def run_all_stations(model_builder, config: dict, run_dir: Path):

    summary = {}
    benchmark = {}

    df_stations = pd.read_csv(DATA_DIR / config["data"]["stations_csv"])
    df_stations.columns = ["cidade", "lat", "lon"]

    stations_mode = config.get("experiment", {}).get("stations_mode", "all")
    output_mode = config.get("experiment", {}).get("output_mode", "standard")

    if stations_mode == "single":
        single_station = config["experiment"].get("single_station_name")

        if single_station is None:
            raise ValueError(
                "stations_mode is 'single' but no 'single_station_name' provided."
            )

        df_stations = df_stations[
            df_stations["cidade"] == single_station
        ]

        if df_stations.empty:
            raise ValueError(
                f"Station '{single_station}' not found in stations CSV."
            )

        print(f"Running ONLY station: {single_station}")
    else:
        print("Running ALL stations")

    if output_mode == "standard":
        def get_base_dir(cidade):
            return run_dir / "locations" / cidade

    elif output_mode == "debug":
        def get_base_dir(cidade):
            return run_dir / "debug_outputs"

    elif output_mode == "global_debug":
        global_debug_root = PROJECT_ROOT / "runs" / "_GLOBAL_DEBUG"
        global_debug_root.mkdir(parents=True, exist_ok=True)

        def get_base_dir(cidade):
            return global_debug_root
    else:
        raise ValueError("Unsupported output_mode")

    for _, row in df_stations.iterrows():
        cidade = row["cidade"]
        print(f"Running station: {cidade}")

        start_time = time.time()
        base_dir = get_base_dir(cidade)

        metrics = run_single_station(
            cidade,
            model_builder,
            config,
            run_dir,
            base_dir=base_dir,
        )

        elapsed = time.time() - start_time

        summary[cidade] = metrics
        benchmark[cidade] = {"time_seconds": round(elapsed, 4)}

    save_json(summary, run_dir / "summary_metrics.json")
    save_json(benchmark, run_dir / "benchmark_times.json")