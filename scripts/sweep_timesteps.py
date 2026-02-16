import copy
import yaml
from datetime import datetime

from src.utils.config import load_config
from src.models.builders import lstm_builder
from src.experiments.run_all_stations import run_all_stations
from src.utils.files import ensure_dir
from src.config.paths import PROJECT_ROOT


# ----------------------------------
# Ensure main runs folder exists
# ----------------------------------
runs_root = PROJECT_ROOT / "runs"
ensure_dir(runs_root)


# ----------------------------------
# Sweep parameters
# ----------------------------------
timesteps_list = [7, 10, 15, 20, 25, 30, 45, 60, 75, 90, 105, 120]
loss_list = ["weighted_mse", "huber", "mse", "mae"]
use_scaler_list = [True, False]
horizon_list = [1]
dataset_starts = [
    "1980-01-01",
    "1990-01-01",
    "2000-01-01",
    "2010-01-01",
    "2015-01-01",
]
dataset_end = "2020-12-31"


# ----------------------------------
# Load base config
# ----------------------------------
base_config = load_config("lstm_daily_base.yaml")


# ----------------------------------
# Run sweep
# ----------------------------------
for start_date in dataset_starts:
    for ts in timesteps_list:
        for loss in loss_list:
            for use_scaler in use_scaler_list:
                for horizon in horizon_list:

                    config = copy.deepcopy(base_config)

                    # ----------------------------------
                    # Update config for this experiment
                    # ----------------------------------
                    config["data"]["initial_date"] = start_date
                    config["data"]["end_date"] = dataset_end
                    config["data"]["timesteps"] = ts
                    config["data"]["horizon"] = horizon
                    config["training"]["loss"] = loss
                    config["preprocessing"]["use_scaler"] = use_scaler

                    # ----------------------------------
                    # Build run name
                    # ----------------------------------
                    model_name = config["experiment"]["model"]
                    freq = config["experiment"]["frequency"]

                    start_year = start_date[:4]
                    end_year = dataset_end[:4]
                    date_range_str = f"{start_year}-{end_year}"

                    base_run_name = (
                        f"{model_name}_{freq}"
                        f"_{date_range_str}"
                        f"_ts{ts}"
                        f"_h{horizon}"
                        f"_loss{loss}"
                        f"_scaler{use_scaler}"
                    )

                    stations_mode = config["experiment"].get("stations_mode", "all")
                    output_mode = config["experiment"].get("output_mode", "standard")

                    if output_mode == "debug":
                        if stations_mode == "single":
                            station_name = config["experiment"].get(
                                "single_station_name", "unknown"
                            )
                            run_name = f"{base_run_name}_DEBUG_{station_name}"
                        else:
                            run_name = f"{base_run_name}_DEBUG"
                    else:
                        run_name = base_run_name

                    # Optional: add timestamp to prevent overwriting
                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # run_name = f"{run_name}_{timestamp}"

                    # ----------------------------------
                    # Create run directory
                    # ----------------------------------
                    run_dir = PROJECT_ROOT / "runs" / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)

                    # ----------------------------------
                    # Save frozen config
                    # ----------------------------------
                    config["experiment"]["run_name"] = run_name
                    config["experiment"]["date_range"] = date_range_str

                    with open(run_dir / "config.yaml", "w") as f:
                        yaml.dump(config, f)

                    # ----------------------------------
                    # Logging
                    # ----------------------------------
                    print("=" * 90)
                    print(f"Running experiment: {run_name}")
                    print(f"Date range: {start_date} → {dataset_end}")
                    print(f"Timesteps: {ts}")
                    print(f"Horizon: {horizon}")
                    print(f"Loss: {loss}")
                    print(f"Use scaler: {use_scaler}")
                    print(f"Stations mode: {stations_mode}")
                    print(f"Output mode: {output_mode}")
                    print("=" * 90)

                    # ----------------------------------
                    # Run experiment
                    # ----------------------------------
                    run_all_stations(lstm_builder, config, run_dir)