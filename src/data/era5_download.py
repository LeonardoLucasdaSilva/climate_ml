
from src.config.paths import RAW_DATA_DIR
import zipfile
import shutil
from pathlib import Path
import cdsapi

def download_era5(
    dataset: str,
    request: dict,
    filename: str,
    overwrite: bool = False,
) -> Path:

    output_path = RAW_DATA_DIR / filename

    if output_path.exists() and not overwrite:
        print(f"File {output_path} already exists")
        return output_path

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(str(output_path))

    print(f"Downloaded {output_path}")

    # detect ZIP
    with open(output_path, "rb") as f:
        magic = f.read(4)

    if magic != b"PK\x03\x04":
        return output_path  # real NetCDF

    print("ZIP detected, extracting...")

    extract_dir = output_path.parent / f".tmp_{output_path.stem}"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(output_path, "r") as z:
        z.extractall(extract_dir)

    # find the real data file (random name!)
    data_files = list(
        p for p in extract_dir.rglob("*")
        if p.suffix in [".nc", ".grib", ".grb"]
    )

    if not data_files:
        raise RuntimeError(f"No data file found inside {output_path}")

    src = data_files[0]
    dst = output_path.with_suffix(src.suffix)

    # COPY (never move)
    shutil.copy2(src, dst)

    if not dst.exists():
        raise RuntimeError("Extraction failed: destination file missing")

    # cleanup
    output_path.unlink()        # remove ZIP
    shutil.rmtree(extract_dir) # remove temp dir

    print(f"Saved ERA5 data as {dst}")
    return dst