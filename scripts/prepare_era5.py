from src.data.prepare import sum_column, mean_column
import pandas as pd
from src.config.paths import RAW_DATA_DIR, INTERIM_DATA_DIR

def main():

    # Read the positions from inmet stations from a csv file and store them in a pandas dataframe
    df = pd.read_csv('../data/inmet_pos.csv')
    df.columns = ['cidade', 'lat', 'lon']

    era5_columns = ["2m_dewpoint_temperature", "surface_pressure", "surface_solar_radiation_downwards",
                    "2m_temperature", "total_precipitation"]
    era5_columns_mean = ["2m_dewpoint_temperature", "surface_pressure", "2m_temperature"]
    era5_columns_sum = ["surface_solar_radiation_downwards", "total_precipitation"]

    for column in era5_columns:
        for index, row in df.iterrows():

            timespan = '1D'
            raw_filename = "era5_precipitation_timeseries_"+row['cidade'].lower()+".nc"
            interim_filename = "era5_precipitation_timeseries_" + row['cidade'].lower() + "_" + timespan + ".nc"

            print(RAW_DATA_DIR / raw_filename)

            if column in era5_columns_sum:
                sum_column(RAW_DATA_DIR / raw_filename, INTERIM_DATA_DIR / interim_filename, timespan)
            elif column in era5_columns_mean:
                mean_column(RAW_DATA_DIR / raw_filename, INTERIM_DATA_DIR / interim_filename, timespan)
            else:
                raise ValueError(f"Column {column} does not belong to any of the lists: era5_columns_mean, era5_columns_sum")


if __name__ == "__main__":
    main()