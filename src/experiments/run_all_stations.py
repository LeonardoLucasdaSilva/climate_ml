from pathlib import Path
import pandas as pd
import time

from src.experiments.utils import resolve_output_directory, filter_stations
from src.utils.files import save_json
from src.config.paths import DATA_DIR
from src.data.pipeline import prepare_station_data
from src.models.pytorch.factory import build_model_and_loss
from src.models.pytorch.train import train_regression_model
from src.models.pytorch.evaluator import evaluate_model
from src.evaluation.metrics import mae, smape


def run_single_station(cidade, model_builder, config, run_dir, base_dir):

    # -----------------------
    # DATA
    # -----------------------
    (X_train, X_val, X_test,
     y_train, y_val, y_test), scaler_y, use_log = (
        prepare_station_data(cidade, config)
    )

    # -----------------------
    # MODEL
    # -----------------------
    num_features = X_train.shape[2]
    model, criterion = build_model_and_loss(
        model_builder, config, num_features
    )

    # -----------------------
    # TRAIN
    # -----------------------
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

    # -----------------------
    # VALIDATION
    # -----------------------
    y_true_val, y_pred_val = evaluate_model(
        model, X_val, y_val, scaler_y, use_log
    )

    # -----------------------
    # TEST
    # -----------------------
    y_true_test, y_pred_test = evaluate_model(
        model, X_test, y_test, scaler_y, use_log
    )

    metrics = {
        "val_loss": float(val_loss),
        "mae": float(mae(y_true_test, y_pred_test)),
        "smape": float(smape(y_true_test, y_pred_test)),
    }

    return metrics

def run_all_stations(model_builder, config: dict, run_dir: Path):

    summary = {}
    benchmark = {}

    df_stations = pd.read_csv(DATA_DIR / config["data"]["stations_csv"])
    df_stations.columns = ["cidade", "lat", "lon"]

    df_stations = filter_stations(df_stations, config)

    for cidade in df_stations["cidade"]:

        start_time = time.time()

        base_dir = resolve_output_directory(
            cidade, config, run_dir
        )

        metrics = run_single_station(
            cidade,
            model_builder,
            config,
            run_dir,
            base_dir=base_dir,
        )

        summary[cidade] = metrics
        benchmark[cidade] = {
            "time_seconds": round(time.time() - start_time, 4)
        }

    save_json(summary, run_dir / "summary_metrics.json")
    save_json(benchmark, run_dir / "benchmark_times.json")