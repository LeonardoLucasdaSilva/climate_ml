import copy
import yaml
from itertools import product
from pathlib import Path

from src.utils.dates import days_before


def generate_dataset_starts(dataset_end, days_before_list):
    return [days_before(dataset_end, d) for d in days_before_list]


def generate_sweep_configs(base_config, sweep_params, dataset_end):
    dataset_starts = generate_dataset_starts(
        dataset_end,
        sweep_params["days_before_list"],
    )

    for start_date, ts, loss, use_scaler, horizon in product(
        dataset_starts,
        sweep_params["timesteps_list"],
        sweep_params["loss_list"],
        sweep_params["use_scaler_list"],
        sweep_params["horizon_list"],
    ):

        config = copy.deepcopy(base_config)

        config["data"]["initial_date"] = start_date
        config["data"]["end_date"] = dataset_end
        config["data"]["timesteps"] = ts
        config["data"]["horizon"] = horizon
        config["training"]["loss"] = loss
        config["preprocessing"]["use_scaler"] = use_scaler

        yield config


def build_run_name(config):
    model_name = config["experiment"]["model"]
    freq = config["experiment"]["frequency"]

    start = config["data"]["initial_date"]
    end = config["data"]["end_date"]
    ts = config["data"]["timesteps"]
    horizon = config["data"]["horizon"]
    loss = config["training"]["loss"]
    scaler = config["preprocessing"]["use_scaler"]

    return (
        f"{model_name}_{freq}"
        f"_{start}-{end}"
        f"_ts{ts}"
        f"_h{horizon}"
        f"_loss{loss}"
        f"_scaler{scaler}"
    )


def initialize_run_directory(config, runs_root: Path):
    run_name = build_run_name(config)
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config["experiment"]["run_name"] = run_name
    config["experiment"]["date_range"] = (
        f"{config['data']['initial_date'][:4]}"
        f"-{config['data']['end_date'][:4]}"
    )

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return run_dir
