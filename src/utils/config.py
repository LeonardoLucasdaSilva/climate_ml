import yaml
from src.config.paths import CONFIG_DIR


def load_config(path):
    path = CONFIG_DIR / path
    with open(path, "r") as f:
        return yaml.safe_load(f)