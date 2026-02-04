import math as m
import numpy as np

def latlon_to_grid(latitude, longitude):

    lat_array = np.linspace(m.floor(latitude),m.ceil(latitude), num=5)
    lon_array = np.linspace(m.floor(longitude), m.ceil(longitude), num=5)

    diff = list(abs(lat_array - latitude))

    # Find the minimum value
    min_value = min(diff)

    # Find the index of the first occurrence of the minimum value
    min_index = diff.index(min_value)

    lat_grid = lat_array[min_index]

    diff = list(abs(lon_array - longitude))

    # Find the minimum value
    min_value = min(diff)

    # Find the index of the first occurrence of the minimum value
    min_index = diff.index(min_value)

    lon_grid = lon_array[min_index]

    return lat_grid, lon_grid

