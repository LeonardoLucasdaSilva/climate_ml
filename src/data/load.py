from pathlib import Path
from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

def load_raw (filename: str) -> Path:
    # This function receives a raw data file name and returns it path
    path = RAW_DATA_DIR / filename

    if not(path.exists()):
        raise FileNotFoundError(f"File {filename} does not exist")

    return path

def load_interim(filename: str) -> Path:
    # This function receives an interim data file name and returns it path
    path = INTERIM_DATA_DIR / filename

    if not(path.exists()):
        raise FileNotFoundError(f"File {filename} does not exist")

    return path

def load_processed(filename: str) -> Path:
    # This function receives a processed data file name and returns it path
    path = PROCESSED_DATA_DIR / filename

    if not(path.exists()):
        raise FileNotFoundError(f"File {filename} does not exist")

    return path






