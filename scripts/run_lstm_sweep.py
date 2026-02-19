from src.utils.config import load_config
from src.models.builders import lstm_builder
from src.experiments.run_all_stations import run_all_stations
from src.experiments.sweep_runner import (
    generate_sweep_configs,
    initialize_run_directory,
)
from src.config.paths import PROJECT_ROOT
from src.utils.files import ensure_dir


# ----------------------------------
# Ensure runs folder exists
# ----------------------------------
runs_root = PROJECT_ROOT / "runs"
ensure_dir(runs_root)


# ==================================
# SWEEP DEFINITION
# ==================================

timesteps_list = [3, 5, 7, 10, 20, 30]
loss_list = ["quantile_0.9","quantile_0.8", "weighted_mse", "huber", "mse", "mae"]
use_scaler_list = [True, False]
horizon_list = [1]

days_before_list = [365, 720, 1440, 2800]
dataset_end = "2020-12-31"


# ----------------------------------
# Load base config
# ----------------------------------
base_config = load_config("lstm_daily_base.yaml")


sweep_params = {
    "timesteps_list": timesteps_list,
    "loss_list": loss_list,
    "use_scaler_list": use_scaler_list,
    "horizon_list": horizon_list,
    "days_before_list": days_before_list,
}


# ----------------------------------
# Run sweep
# ----------------------------------
for config in generate_sweep_configs(
    base_config,
    sweep_params,
    dataset_end,
):

    run_dir = initialize_run_directory(config, runs_root)

    print("=" * 80)
    print(f"Running experiment: {config['experiment']['run_name']}")
    print("=" * 80)

    run_all_stations(lstm_builder, config, run_dir)