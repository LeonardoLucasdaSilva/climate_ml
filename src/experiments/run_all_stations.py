from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
import torch

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
)
from src.utils.files import save_figure
from src.config.paths import DATA_DIR

def run_single_station(
    cidade: str,
    model_builder,
    config: dict,
    run_dir: Path,
):
    station_dir = run_dir / "locations" / cidade
    station_dir.mkdir(parents=True, exist_ok=True)

    # ---------- LOAD DATA ----------
    path = load_interim(f"era5_precipitation_timeseries_{cidade}_1D.nc")
    nc_dataset = xr.open_dataset(path, engine="netcdf4")

    tp = nc_dataset["tp"]
    df = tp.to_dataframe(name="total_precipitation")

    series = df.loc[
        config["data"]["initial_date"]:
        config["data"]["end_date"]
    ]["total_precipitation"]

    # ---------- PREPROCESS ----------
    X, y = create_sliding_windows(
        series,
        config["data"]["timesteps"],
        config["data"]["horizon"],
    )

    use_log = config.get("preprocessing", {}).get("use_log", False)

    if use_log:
        X_proc = np.log1p(X)
        y_proc = np.log1p(y)
    else:
        X_proc = X
        y_proc = y

    X_scaled, y_scaled, scaler_x, scaler_y = prepare_data_seq_to_one(
        X_proc,
        y_proc,
        num_features=1
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        temporal_train_val_test_split(
            X_scaled,
            y_scaled,
            config["data"]["train_split"],
            config["data"]["val_split"],
        )
    )

    # ---------- MODEL ----------
    model = model_builder(config)

    loss_name = config["training"].get("loss", "mse")

    if loss_name == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_name == "huber":
        criterion = torch.nn.SmoothL1Loss()
    elif loss_name == "mae":
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

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

    # ---------- SAVE MODEL ----------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "station": cidade,
            "val_loss": float(val_loss),
        },
        station_dir / "model.pt",
    )

    # ---------- SAVE HISTORY ----------
    fig = plot_training_history_torch(history)
    save_figure(fig, station_dir / "history.png")

    # ---------- VALIDATION ----------
    y_pred_val = predict_timeseries_model(model, X_val)
    y_pred_val = scaler_y.inverse_transform(y_pred_val.reshape(-1, 1))

    y_true_val = scaler_y.inverse_transform(y_val.reshape(-1, 1))

    if use_log:
        y_pred_val = np.expm1(y_pred_val)
        y_true_val = np.expm1(y_true_val)

    y_pred_val = y_pred_val.squeeze()
    y_true_val = y_true_val.squeeze()

    fig_val = plot_real_vs_predicted_scatter(y_true_val, y_pred_val)
    save_figure(fig_val, station_dir / "val_scatter.png")

    fig_val_ts = plot_real_vs_predicted_timeseries(
        y_true_val,
        y_pred_val,
        title="Validation - Real vs Predicted",
    )
    save_figure(fig_val_ts, station_dir / "val_timeseries.png")

    # ---------- TEST ----------
    y_pred_test = predict_timeseries_model(model, X_test)
    y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))

    y_true_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    if use_log:
        y_pred_test = np.expm1(y_pred_test)
        y_true_test = np.expm1(y_true_test)

    y_pred_test = y_pred_test.squeeze()
    y_true_test = y_true_test.squeeze()

    fig_test = plot_real_vs_predicted_scatter(y_true_test, y_pred_test)
    save_figure(fig_test, station_dir / "test_scatter.png")

    fig_test_ts = plot_real_vs_predicted_timeseries(
        y_true_test,
        y_pred_test,
        title="Test - Real vs Predicted",
    )
    save_figure(fig_test_ts, station_dir / "test_timeseries.png")

    metrics = {
        "val_loss": float(val_loss),
        "mae": float(mae(y_true_test, y_pred_test)),
        "smape": float(smape(y_true_test, y_pred_test)),
    }

    with open(station_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

import time
import json


def run_all_stations(model_builder, config: dict, run_dir: Path):

    summary = {}
    benchmark = {}

    df_stations = pd.read_csv(
        DATA_DIR / config["data"]["stations_csv"]
    )
    df_stations.columns = ["cidade", "lat", "lon"]

    for _, row in df_stations.iterrows():
        cidade = row["cidade"]
        print(f"Running station: {cidade}")

        start_time = time.time()

        metrics = run_single_station(
            cidade,
            model_builder,
            config,
            run_dir,
        )

        end_time = time.time()
        elapsed = end_time - start_time

        summary[cidade] = metrics
        benchmark[cidade] = {
            "time_seconds": round(elapsed, 4)
        }

    # Save metrics summary
    with open(run_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Save benchmark times
    with open(run_dir / "benchmark_times.json", "w") as f:
        json.dump(benchmark, f, indent=4)