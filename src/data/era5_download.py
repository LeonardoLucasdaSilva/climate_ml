import os
from pathlib import Path
import cdsapi

from src.config.paths import RAW_DATA_DIR

def download_era5(
    dataset: str,
    request: dict,
    filename: str,
    overwrite: bool = False,
) -> Path:
    """
    Download an ERA5 dataset and save it to data/raw

    Parameters:
        dataset (str): ERA5 dataset
        request (dict): Dictionary of request parameters
        filename (str): Filename to be saved locally
        overwrite (bool): Whether to overwrite existing file

    Returns:
        Path: Path to downloaded file
    """

    output_path = RAW_DATA_DIR / filename

    if os.path.exists(output_path) and not overwrite:
        print(f"File {output_path} already exists")
        return output_path

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(str(output_path))

    print(f"File {output_path} downloaded")

    return output_path