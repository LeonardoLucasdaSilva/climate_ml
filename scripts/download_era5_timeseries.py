from src.data import download_era5
import pandas as pd
from src.spatial.grid import latlon_to_grid

# Read the positions from inmet stations from a csv file and store them in a pandas dataframe
df = pd.read_csv('../data/inmet_pos.csv')
df.columns = ['cidade', 'lat', 'lon']


# Download the timeseries from all stations given the closest era5 grid point
DATASETS = []

for index, row in df.iterrows():

    lat,lon = latlon_to_grid(row['lat'], row['lon'])

    requisition = {
        "name": "era5_precipitation_timeseries_"+row['cidade'].lower(),
        "dataset": "reanalysis-era5-single-levels-timeseries",
        "request": {
            "variable": ["total_precipitation"],
            "location": {"longitude": lon, "latitude": lat},
            "date": ["1979-01-01/2025-12-31"],
            "data_format": "netcdf"
        },
        "filename": "era5_precipitation_timeseries_"+row['cidade'].lower()
    }
    DATASETS.append(requisition)


def main():
    for spec in DATASETS:
        print(f"Downloading {spec['name']}")

        download_era5(
            dataset=spec["dataset"],
            request=spec["request"],
            filename=spec["filename"]+".zip",
        )

if __name__ == "__main__":
    main()



