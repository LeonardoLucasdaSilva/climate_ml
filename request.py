import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
"product_type": ["reanalysis"],
"variable": [
"2m_dewpoint_temperature",
"2m_temperature",
"mean_sea_level_pressure",
"sea_surface_temperature",
"surface_pressure",
"ice_temperature_layer_1",
"ice_temperature_layer_2",
"ice_temperature_layer_3",
"ice_temperature_layer_4",
"maximum_2m_temperature_since_previous_post_processing",
"minimum_2m_temperature_since_previous_post_processing",
"skin_temperature",
"total_cloud_cover"
],
"year": ["1940", "1941", "1942"],
"month": ["01", "02", "03"],
"day": ["01", "02", "03"],
"time": ["00:00", "01:00", "02:00"],
"data_format": "netcdf",
"download_format": "unarchived",
"area": [10, -10, -10, 10],
"grid": '0.25/0.25'
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

