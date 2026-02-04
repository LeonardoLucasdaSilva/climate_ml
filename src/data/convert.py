# src/data/convert.py
from pathlib import Path
import xarray as xr


def grib_to_netcdf(
    grib_path: Path,
    nc_path: Path,
    *,
    filter_by_keys: dict | None = None
) -> None:
    """
    Convert a GRIB file to NetCDF.

    Parameters
    ----------
    grib_path : Path
        Path to input GRIB file.
    nc_path : Path
        Path to output NetCDF file.
    filter_by_keys : dict, optional
        cfgrib filter keys (e.g. {"shortName": "t2m"}).
    """
    print("Converting GRIB file to NetCDF")
    backend_kwargs = {}
    if filter_by_keys:
        backend_kwargs["filter_by_keys"] = filter_by_keys

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs=backend_kwargs,
    )

    nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(nc_path)

