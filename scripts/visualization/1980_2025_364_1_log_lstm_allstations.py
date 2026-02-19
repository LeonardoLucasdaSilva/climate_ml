# Imports

# Libraries
import numpy as np
import xarray as xr
import pandas as pd

# Built-in functions
from src.data import load_interim
from src.data.preprocess import create_sliding_windows, prepare_data_seq_to_one
from src.data.split import temporal_train_val_test_split
from src.evaluation.metrics import mae, smape
from src.models.train import train_regression_model
from src.evaluation.plots import plot_real_vs_predicted_scatter, plot_training_history_torch, save_table_as_image
from src.models.architectures import LSTMSeqToVec
from src.models.predict import predict_timeseries_model
from src.config.paths import PROJECT_ROOT, IMAGE_DATA_DIR
from src.utils.files import save_figure

# See if this is maintained later
np.random.seed(42)

# Read the positions from inmet stations from a csv file and store them in a pandas dataframe
df = pd.read_csv(PROJECT_ROOT/ 'data'/ 'inmet_pos.csv')
df.columns = ['cidade', 'lat', 'lon']

overwrite = True

val_losses = []
cidades = []
maes = []
smapes = []

for index, row in df.iterrows():

    filename = row['cidade'] + ".png"

    print("Downloading", filename)

    save_path_history = (
            IMAGE_DATA_DIR
            / "1980_2025_364_1_log_lstm_allstations"
            / "history"
            / filename
    )

    save_path_val = (
            IMAGE_DATA_DIR
            / "1980_2025_364_1_log_lstm_allstations"
            / "scatter_val"
            / filename
    )

    save_path_test = (
            IMAGE_DATA_DIR
            / "1980_2025_364_1_log_lstm_allstations"
            / "scatter_test"
            / filename
    )

    if save_path_history.exists() and save_path_val.exists() and save_path_test.exists() and not overwrite:
        print(f"All the graphs from {filename} file already exist")
        continue

    path = load_interim('era5_precipitation_timeseries_'+str(row['cidade'])+'_1D.nc')

    nc_dataset = xr.open_dataset(path, engine="netcdf4")

    # Saves the variable tp = Total Precipitation
    tp = nc_dataset["tp"]
    df = tp.to_dataframe(name="total_precipitation")

    TIMESTEPS = 364  # past days used as input
    HORIZON = 1  # predict 1 day ahead

    # series = df["total_precipitation"]
    initial_date = "1980-01-01"
    end_date = "2025-12-31"

    series = df.loc[initial_date:end_date]["total_precipitation"]

    X, y = create_sliding_windows(series, TIMESTEPS, HORIZON)

    # Log transform
    X_log = np.log1p(X)
    y_log = np.log1p(y)

    # Scale the data
    X_scaled, y_scaled, scaler_x, scaler_y = prepare_data_seq_to_one(X_log, y_log, num_features=1)

    # Split the data into Train, Validation and Test
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(X_scaled, y_scaled, 0.7, 0.2)

    model = LSTMSeqToVec(
        horizon=HORIZON,
        timesteps=TIMESTEPS,
        num_features=1,
    )

    history, val_loss = train_regression_model(model, X_train, y_train, X_val, y_val, epochs=250, patience=10)

    val_losses.append(round(val_loss, 5))
    cidades.append(row['cidade'])

    if save_path_history.exists() and not overwrite:
        print(f"File {save_path_history} already exists")
    else:
        fig_history = plot_training_history_torch(history)

        save_figure(fig_history, save_path_history)

    y_pred_log_scaled = predict_timeseries_model(model, X_val)

    y_pred_log = scaler_y.inverse_transform(y_pred_log_scaled)
    y_pred = np.expm1(y_pred_log)

    y_true_log = scaler_y.inverse_transform(y_val.squeeze(-1))
    y_true = np.expm1(y_true_log)

    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    if save_path_val.exists() and not overwrite:
        print(f"File {save_path_val} already exists")
    else:
        fig_scatter_test = plot_real_vs_predicted_scatter(y_true, y_pred)
        save_figure(fig_scatter_test, save_path_val)


    y_pred_log_scaled = predict_timeseries_model(model, X_test)
    y_pred_log = scaler_y.inverse_transform(y_pred_log_scaled.reshape(-1, 1))
    y_pred = np.expm1(y_pred_log).squeeze()

    y_true_log = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_true = np.expm1(y_true_log).squeeze()

    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    maes.append(mae(y_true, y_pred))
    smapes.append(smape(y_true, y_pred))

    if save_path_test.exists() and not overwrite:
        print(f"File {save_path_test} already exists")
    else:
        fig_scatter_test = plot_real_vs_predicted_scatter(y_true, y_pred)
        save_figure(fig_scatter_test, save_path_test)

# ---------- TABLES -------------

# Validation loss
loss_path = (
        IMAGE_DATA_DIR
        / "1980_2025_364_1_log_lstm_allstations"
        / 'validation_loss.png'
)

df = pd.DataFrame({
    "Estação": cidades,
    "Validation Loss": val_losses,
})

# MAE
mae_path = (
        IMAGE_DATA_DIR
        / "1980_2025_364_1_log_lstm_allstations"
        / 'mae.png'
)

smape_path = (
        IMAGE_DATA_DIR
        / "1980_2025_364_1_log_lstm_allstations"
        / 'smape.png'
)

# MAPE

save_table_as_image(df,loss_path)

df = pd.DataFrame({
    "Estação": cidades,
    "MAE": maes,
})

save_table_as_image(df,mae_path)

df = pd.DataFrame({
    "Estação": cidades,
    "SMAPE": smapes,
})

save_table_as_image(df,smape_path)













