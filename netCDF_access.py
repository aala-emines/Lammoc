import xarray as xr
# Load the NetCDF file
ds = xr.open_dataset('PT_all_grid_Zenodo.nc')
# Coordinates
lon_value = -36.5625
lat_value = -7.704221481600492
month_value = 12  # December
# Access the data for December at the specified coordinates
a = ds['a'].sel(lon=lon_value, lat=lat_value, time=month_value).item()
b = ds['b'].sel(lon=lon_value, lat=lat_value, time=month_value).item()
#Or access the data for December at the nearest point
lon_value = -36.9
lat_value = -7.9
month_value = 12  # December
# Access the data for December at the specified coordinates
a_nearest = ds['a'].sel(lon=lon_value, lat=lat_value, time=month_value,method = 'nearest').item()
b_nearest = ds['b'].sel(lon=lon_value, lat=lat_value, time=month_value, method = 'nearest').item()
