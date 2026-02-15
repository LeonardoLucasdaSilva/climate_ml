import copy
import yaml

from src.utils.config import load_config
from src.models.builders import lstm_builder
from src.experiments.run_all_stations import run_all_stations
from src.utils.files import ensure_dir
from src.config.paths import PROJECT_ROOT

# Ensure main runs folder exists
runs_root = PROJECT_ROOT / "runs"
ensure_dir(runs_root)

# ----------------------------------
# Sweep parameters
# ----------------------------------

timesteps_list = [7, 10, 15, 20, 25, 30, 45, 60, 75, 90, 105, 120]
loss_list = ["huber", "mse", "mae"]
horizon = 1


# ----------------------------------
# Load base config
# ----------------------------------

base_config = load_config("lstm_daily_base.yaml")

# Extract dataset boundaries
initial_date = base_config["data"]["initial_date"]
end_date = base_config["data"]["end_date"]

start_year = initial_date[:4]
end_year = end_date[:4]

date_range_str = f"{start_year}-{end_year}"


# ----------------------------------
# Run sweep
# ----------------------------------

for ts in timesteps_list:
    for loss in loss_list:

        config = copy.deepcopy(base_config)

        config["data"]["timesteps"] = ts
        config["data"]["horizon"] = horizon
        config["training"]["loss"] = loss  # NEW

        model_name = config["experiment"]["model"]
        freq = config["experiment"]["frequency"]

        run_name = (
            f"{model_name}_{freq}"
            f"_{date_range_str}"
            f"_ts{ts}"
            f"_h{horizon}"
            f"_loss{loss}"   # NEW
        )

        run_dir = PROJECT_ROOT / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save frozen config
        config["experiment"]["run_name"] = run_name
        config["experiment"]["date_range"] = date_range_str

        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        print("=" * 60)
        print(f"Running experiment: {run_name}")
        print(f"Date range: {initial_date} → {end_date}")
        print(f"Timesteps: {ts}")
        print(f"Horizon: {horizon}")
        print(f"Loss: {loss}")  # NEW
        print("=" * 60)

        run_all_stations(lstm_builder, config, run_dir)
