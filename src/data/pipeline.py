from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.data import load_interim
from src.data.preprocess import create_sliding_windows, prepare_data_seq_to_one
from src.data.split import temporal_train_val_test_split


def prepare_station_data(cidade: str, config: dict):
    variables = config["data"]["variables"] or ["total_precipitation"]
    target_column = config["data"]["target"]

    # -----------------------
    # LOAD
    # -----------------------
    data_frames = []

    for var in variables:
        filename = f"era5_{var}_timeseries_{cidade.lower()}_1D.nc"
        path = load_interim(filename)

        ds = xr.open_dataset(path)
        da = list(ds.data_vars.values())[0]
        df_var = da.to_dataframe(name=var)
        data_frames.append(df_var)

    df = pd.concat(data_frames, axis=1)

    df = df.loc[
        config["data"]["initial_date"]:
        config["data"]["end_date"]
    ]

    X_raw = df.values
    y_raw = df[target_column].values

    # -----------------------
    # SLIDING WINDOWS
    # -----------------------
    X, y = create_sliding_windows(
        X_raw,
        y_raw,
        config["data"]["timesteps"],
        config["data"]["horizon"],
    )

    # -----------------------
    # LOG
    # -----------------------
    use_log = config.get("preprocessing", {}).get("use_log", False)
    if use_log:
        X = np.log1p(X)
        y = np.log1p(y)

    # -----------------------
    # SCALING
    # -----------------------
    use_scaler = config.get("preprocessing", {}).get("use_scaler", True)

    if use_scaler:
        num_features = X.shape[2]
        X, y, scaler_x, scaler_y = prepare_data_seq_to_one(
            X, y, num_features=num_features
        )
    else:
        y = y.reshape(-1, 1)
        scaler_x, scaler_y = None, None

    # -----------------------
    # SPLIT
    # -----------------------
    splits = temporal_train_val_test_split(
        X,
        y,
        config["data"]["train_split"],
        config["data"]["val_split"],
    )

    return splits, scaler_y, use_log