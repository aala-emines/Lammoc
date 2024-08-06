import numpy as np
import xarray as xr
import pandas as pd
import dask
import os

"""
Exportando dados MENSAIS de todas as variaveis para determinadas posicoes/pontos geograficas.

Para as variáveis pr e ETo, os valores são os acumulados,
para as demais, é média dos mês 
"""
# periodo para ser exportado
date_start, date_end = '1961-01-01', '2022-12-31'

# set correct path of the netcdf files
path_var = 'pr_Tmax_Tmin_NetCDF_Files/'

# Posicoes: Colocar em ordem, separando por virgula. Neste exemplo temos dois pontos em que as coordenadas
# (lat, lon) sao (-20.6,-44.6) e  (-21.0, -44.1), respectivamente para o primeiro e segundo ponto.
# Pode-se colocar quantos pontos quiser, apenas separe por virgula.

# variables names
var_names = ['pr']
def rapperFunc(path_var, date_start, date_end, var_names, miroc6_filename):
    lat = [ eval(miroc6_filename[22:][:-4].split('_')[1]) ]
    lon = [ eval(miroc6_filename[22:][:-4].split('_')[0]) ]
    # function to read the netcdf files
    def rawData(var2get_xr, var_name2get):
        return var2get_xr[var_name2get].loc[dict(time=slice(date_start, date_end))].sel(longitude=xr.DataArray(lon, dims='z'),
                                              latitude=xr.DataArray(lat, dims='z'),
                                              method='nearest').values

    # getting data from NetCDF files
    for n, var_name2get in enumerate(var_names):
        print("getting " + var_name2get)
        if var_name2get in ["pr"]:
            var2get_xr = xr.open_mfdataset(path_var + var_name2get + '*.nc', parallel=True).resample(time="ME").sum("time")
            # var2get_xr[var_name2get].sel(latitude=lat[0], longitude=lon[0], method='nearest').plot()
        else:
            var2get_xr = xr.open_mfdataset(path_var + var_name2get + '*.nc', parallel=True).resample(time="ME").mean("time")

        if n == 0:
            var_ar = rawData(var2get_xr, var_name2get)
            n_lines = var_ar.shape[0]
            time = var2get_xr.time.values
        else:
            var_ar = np.c_[var_ar, rawData(var2get_xr, var_name2get)]

    # saving
    for n in range(len(lat)):
        name_file = 'lat{:.2f}_lon{:.2f}.csv'.format(lat[n], lon[n])
        print(f'arquivo {n + 1} de um total de {len(lat)}; nome do arquivo: {name_file}')
        if ~np.isnan(var_ar[0, n]):
            file = var_ar[:, n::len(lon)]
            df_obs = pd.DataFrame(file, index=time, columns=var_names)
            df_obs['Date'] = df_obs.index
            #pd.DataFrame(file, index=time, columns=var_names).to_csv(name_file, float_format='%.1f')
            df_miroc6 = pd.read_csv('miroc6_historico/' + miroc6_filename)
            df_obs['Date'] = pd.to_datetime(df_obs['Date'])
            df_miroc6['Date'] = pd.to_datetime(df_miroc6['time'])
            merged_df = pd.merge(df_obs, df_miroc6, on='Date', suffixes=('obs', 'MIROC6'))
            merged_df.drop(columns=['Date'], inplace=True)
            merged_df = merged_df[['time', 'pr', 'prec']]
            merged_df.rename(columns={'pr': 'obs_Pr', 'prec': 'MIROC6_Pr'}, inplace=True)
            file_path = os.path.join("PT_results", f"{lat} , {lon} merged.csv")
            merged_df.to_csv(file_path,index = False, float_format='%.6f')


rapperFunc(path_var, date_start, date_end, var_names,"prec_MIROC6_historico_-43.59375_-23.112655356577648.csv")