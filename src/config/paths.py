from pathlib import Path

# This line gets the folder that contains the folder src/ and stores it in the variable PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGE_DATA_DIR = DATA_DIR / "images"
CONFIG_DIR = PROJECT_ROOT / "config"

# Create the directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
