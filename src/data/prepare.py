import xarray as xr

def sum_column(raw_path, interim_path, timespan = '1D', overwrite = False):

    # Timespan values
    # 1D - daily
    # 1M - month
    # 1Y - year
    # 6H - 6 hours
    # XH - X hours

    if interim_path.exists() and not overwrite:
        print(f"File {interim_path} already exists")

    nc_dataset = xr.open_dataset(raw_path, engine="netcdf4")

    # Converts Total Precipitation (m) to Total Precipitation (mm)
    nc_dataset["tp"].values = nc_dataset["tp"].values * 1000

    concat_dataset = nc_dataset.resample(valid_time=timespan).sum()

    concat_dataset.to_netcdf(interim_path)

def mean_column(raw_path, interim_path, timespan='1D', overwrite=False):

    if interim_path.exists() and not overwrite:
        print(f"File {interim_path} already exists")
        return

    nc_dataset = xr.open_dataset(raw_path, engine="netcdf4")

    # Convert Total Precipitation (m) to (mm)
    nc_dataset["tp"].values = nc_dataset["tp"].values * 1000

    mean_dataset = nc_dataset.resample(valid_time=timespan).mean()

    mean_dataset.to_netcdf(interim_path)