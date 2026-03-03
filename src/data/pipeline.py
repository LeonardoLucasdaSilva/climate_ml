from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.data import load_interim
from src.data.preprocess import create_sliding_windows, prepare_data_seq_to_one
from src.data.split import temporal_train_val_test_split
from src.config.paths import PROCESSED_DATA_DIR

def load_era5_timeseries(cidade: str, config: dict) -> pd.DataFrame:
    """
    Carrega séries temporais ERA5 para uma cidade, combinando variáveis em um DataFrame.
    Index: DatetimeIndex diário.
    """
    variables = config["data"]["variables"] or ["total_precipitation"]

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

    return df


def load_inmet_timeseries(cidade: str, config: dict) -> pd.DataFrame:
    """
    Carrega séries temporais INMET para uma estação (cidade),
    usando o código da estação e o estado do config.
    """
    state_acronym = config["experiment"]["state_acronym"]
    station_code = config["experiment"]["single_station_code"]

    csv_path = (
        PROCESSED_DATA_DIR /
        "inmet" /
        state_acronym /
        station_code /
        f"{station_code}_2000_2025_daily.csv"
    )

    df = pd.read_csv(csv_path, sep=";")
    df["date"] = pd.to_datetime(df["DATA"])
    df = df.set_index("date")

    df = df.loc[
        config["data"]["initial_date"]:
        config["data"]["end_date"]
    ]

    # Mapeia nome lógico -> coluna INMET
    mapping = {
        "total_precipitation": "PRECIPITACAO_TOTAL",
    }

    variables = config["data"]["variables"] or ["total_precipitation"]
    cols = {}
    for var in variables:
        inmet_col = mapping.get(var)
        if inmet_col not in df.columns:
            raise KeyError(
                f"Coluna INMET `{inmet_col}` não encontrada para variável `{var}`."
            )
        cols[var] = df[inmet_col]

    df_out = pd.DataFrame(cols, index=df.index)

    return df_out

def prepare_station_data(cidade: str, config: dict):
    """
    Prepara dados para uma estação, permitindo usar ERA5 ou INMET
    como dataset principal, com possibilidade de usar a outra fonte
    como série de referência de teste.
    """
    source = config["data"].get("source", "era5")
    target_column = config["data"]["target"]

    # -----------------------
    # LOAD
    # -----------------------
    if source == "era5":
        df = load_era5_timeseries(cidade, config)
    elif source == "inmet":
        df = load_inmet_timeseries(cidade, config)
    else:
        raise ValueError(f"Fonte de dados desconhecida: {source}")

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
    y_test_ref = None   # referência externa (ERA5 ou INMET)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_end,
        val_end,
    ) = temporal_train_val_test_split(
        X,
        y,
        config["data"]["train_split"],
        config["data"]["val_split"],
    )

    test_start_idx = val_end
    test_end_idx = len(y)

    use_ref = config["data"].get("use_inmet_test", False)

    if use_ref:
        if source == "era5":
            # dataset principal = ERA5
            # referência externa = INMET (como já fazia antes)
            y_test_ref = load_and_prepare_inmet(
                cidade,
                config,
                test_start_idx,
                test_end_idx,
                scaler_y,
                use_log,
            )
        elif source == "inmet":
            # dataset principal = INMET
            # referência externa = ERA5
            y_test_ref = load_and_prepare_era5_reference(
                cidade,
                config,
                test_start_idx,
                test_end_idx,
                scaler_y,
                use_log,
            )

    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_test_inmet": y_test_ref,  # mantém chave para não quebrar o resto
        "train_end": train_end,
        "val_end": val_end,
    }

    return splits, scaler_y, use_log

def load_and_prepare_inmet(
    cidade: str,
    config: dict,
    test_start_idx: int,
    test_end_idx: int,
    scaler_y,
    use_log: bool,
):

    timesteps = config["data"]["timesteps"]
    horizon = config["data"]["horizon"]
    target_column = config["data"]["target"]

    # -----------------------
    # LOAD FULL CSV
    # -----------------------
    csv_path = (
        PROCESSED_DATA_DIR /
        "inmet" /
        config["experiment"]["state_acronym"] /
        config["experiment"]["single_station_code"] /
        f"{config['experiment']['single_station_code']}_2000_2025_daily.csv"
    )

    df = pd.read_csv(csv_path, sep=";")
    df["date"] = pd.to_datetime(df["DATA"])
    df = df.set_index("date")

    # Apply same date filtering as ERA5
    df = df.loc[
        config["data"]["initial_date"]:
        config["data"]["end_date"]
    ]

    if config["data"]["target"] == "total_precipitation":
        y_raw = df["PRECIPITACAO_TOTAL"].values

    # -----------------------
    # SLIDING WINDOWS (FULL SERIES)
    # -----------------------
    _, y_full = create_sliding_windows(
        y_raw.reshape(-1, 1),  # dummy X
        y_raw,
        timesteps,
        horizon,
    )

    # -----------------------
    # LOG TRANSFORM
    # -----------------------
    if use_log:
        y_full = np.log1p(y_full)

    # -----------------------
    # SCALING (USE ERA5 SCALER)
    # -----------------------
    # if scaler_y is not None:
    #    y_full = scaler_y.transform(y_full.reshape(-1, 1))

    # -----------------------
    # SLICE USING ERA5 TEST INDICES
    # -----------------------
    y_test_inmet = y_full[test_start_idx:test_end_idx]

    # Safety check
    if len(y_test_inmet) != (test_end_idx - test_start_idx):
        raise ValueError("INMET test length does not match ERA5 test length.")

    return y_test_inmet

def load_and_prepare_era5_reference(
    cidade: str,
    config: dict,
    test_start_idx: int,
    test_end_idx: int,
    scaler_y,
    use_log: bool,
):
    """
    Carrega ERA5 completo para a mesma estação e converte para
    a mesma estrutura de janelas para ser usado como referência
    no conjunto de teste, quando o dataset principal é INMET.
    """
    timesteps = config["data"]["timesteps"]
    horizon = config["data"]["horizon"]
    target_column = config["data"]["target"]

    df = load_era5_timeseries(cidade, config)

    y_raw = df[target_column].values

    _, y_full = create_sliding_windows(
        y_raw.reshape(-1, 1),
        y_raw,
        timesteps,
        horizon,
    )

    if use_log:
        y_full = np.log1p(y_full)

    y_test_ref = y_full[test_start_idx:test_end_idx]

    if len(y_test_ref) != (test_end_idx - test_start_idx):
        raise ValueError("ERA5 test length does not match main dataset test length.")

    return y_test_ref