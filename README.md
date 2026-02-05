# Climate Change Project

This project aims to understand the usage of Recurrent Neural Networks (RNN) to predict the behavior of climate
variables in the state of Rio Grande do Sul, Brazil.

## Setup

You need to install the requirements through `pip install -r requirements.txt`
<br>
For some of the automatic downloads and preparation (timeseries for example) a file called `inmet_pos.csv`
containing the positional data from the inmet stations is needed (row format: name_of_station, latitude, longitude)

## Folders and data

By default, you need to set up a ERA5 key to be able to download the data from the ERA5 API. With that done, you can run
the project. Then, it will automatically set up the folders for the local data usage.

### Raw

As the name suggested this is the folder designated to raw data, straight from ERA5 API, the only difference between
the files here is the dataset and the coverage of the grid. This folder also ideally does not contain zip folders, just
.grib and .nc files.

### Interim

- `_6H` means 6-hour grouped data
- `_XH` means X-hour grouped data
- `_1D` means daily grouped data
- `_1M` means monthly grouped data
- `_1Y` means yearly grouped data

## Downloading datasets

To download the datasets used here, run the scripts in `\scripts\` folder.
- `\scripts\download_era5` downloads all the ERA5 grid data from the Rio Grande do Sul state given a certain column 
    (this is very slow averaging 16min for yearly data).
- `\scripts\download_era5_timeseries` downloads the timeseries data for the grid points closest to each inmet station
    using experimental data from ERA5 reanalysis dataset.

## Notebooks

There are different types of notebooks in this project that are divided by folders. They behave as follows:

- __Concepts__: are notebooks containing theory and conceptual examples, useful to understand the behavior of the code and
data science in general
- __Exploration__: are notebooks containing tests and real usage of the algorithms implemented in the project

