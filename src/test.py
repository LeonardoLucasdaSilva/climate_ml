import xarray as xr

print("AAA")
xrds = xr.open_dataset('../temperature.nc')
#print(xrds)

print(xrds['latitude'].values)