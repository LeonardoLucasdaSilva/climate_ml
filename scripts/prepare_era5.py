from src.data.prepare import concat_precipitation
import pandas as pd
from src.config.paths import RAW_DATA_DIR, INTERIM_DATA_DIR

def main():

    # Read the positions from inmet stations from a csv file and store them in a pandas dataframe
    df = pd.read_csv('../data/inmet_pos.csv')
    df.columns = ['cidade', 'lat', 'lon']

    for index, row in df.iterrows():

        timespan = '1D'
        raw_filename = "era5_precipitation_timeseries_"+row['cidade'].lower()+".nc"
        interim_filename = "era5_precipitation_timeseries_" + row['cidade'].lower() + "_" + timespan + ".nc"

        print(RAW_DATA_DIR / raw_filename)

        concat_precipitation(
            RAW_DATA_DIR / raw_filename,
            INTERIM_DATA_DIR / interim_filename,
            timespan
        )

if __name__ == "__main__":
    main()