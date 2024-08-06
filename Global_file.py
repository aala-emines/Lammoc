import numpy as np
import xarray as xr
import dask
import pandas as pd
from scipy.optimize import curve_fit
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
def power_function(x, a, b):
    return a * x ** b
def wrapperFunc(path_var, date_start, date_end, var_names, miroc6_filename):
    lat = [eval(miroc6_filename[22:][:-4].split('_')[1]) ]
    lon = [eval(miroc6_filename[22:][:-4].split('_')[0]) ]
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

            # Assuming 'time' column is in datetime format, extract year
            merged_df['year'] = pd.to_datetime(merged_df['time']).dt.year

            # Initialize empty list to store fitted parameters and transformation results
            fitted_params = []
            transformed_MIROC6_Pr = []
            a_used = []
            b_used = []
            P_bias_per_year_original = {}
            P_bias_per_year_PT = {}
            # Iterate over each year
            for year in merged_df['year'].unique():
                # Filter data for the current year
                year_data = merged_df[merged_df['year'] == year]

                # Extract obs_Pr and MIROC6_Pr data for the current year
                obs_Pr_data = year_data['obs_Pr'].values
                MIROC6_Pr_data = year_data['MIROC6_Pr'].values

                # Perform curve fitting
                popt, pcov = curve_fit(power_function, MIROC6_Pr_data, obs_Pr_data)

                # Store the fitted parameters
                fitted_params.append({
                    'Year': year,
                    'Fitted Parameters': popt
                })

                # Transform MIROC6_Pr using the fitted parameters for this year
                for index, row in year_data.iterrows():
                    MIROC6_Pr = row['MIROC6_Pr']

                    # Transform using fitted parameters
                    transformed_value = power_function(MIROC6_Pr, *popt)

                    # Append transformed value to list
                    transformed_MIROC6_Pr.append(transformed_value)

                    # Append a and b used for this transformation
                    a_used.append(popt[0])
                    b_used.append(popt[1])

                # Calculate mean of obs_Pr and MIROC6_Pr for the current year
                sum_obs_Pr = year_data['obs_Pr'].sum()
                sum_MIROC6_Pr = year_data['MIROC6_Pr'].sum()
                sum_transformed_MIROC6_Pr = sum(transformed_MIROC6_Pr)
                # Compute P_bias for the current year
                P_bias_original = ((sum_obs_Pr - sum_MIROC6_Pr) / sum_obs_Pr) * 100
                P_bias_PT = ((sum_obs_Pr - sum_transformed_MIROC6_Pr) / sum_obs_Pr) * 100
                # Store P_bias in the dictionary
                P_bias_per_year_original[year] = P_bias_original
                P_bias_per_year_PT[year] = P_bias_PT


            # Optionally, convert the dictionary to a DataFrame for easier manipulation or export
            P_bias_df_original = pd.DataFrame(list(P_bias_per_year_original.items()), columns=['Year', 'P_bias_original'])
            P_bias_df_PT = pd.DataFrame(list(P_bias_per_year_PT.items()), columns=['Year', 'P_bias_PT'])
            P_bias_df_original['P_bias_PT'] = P_bias_df_PT['P_bias_PT']
            file_path = os.path.join("P_bias", f"{lat} , {lon} per year P_bias.csv")
            P_bias_df_original.to_csv(file_path, index=False, float_format='%.2f')
            # Add transformed MIROC6_Pr, a_used, and b_used columns to DataFrame
            merged_df['Transformed_MIROC6_Pr'] = transformed_MIROC6_Pr
            merged_df['a'] = a_used
            merged_df['b'] = b_used
            file_path = os.path.join("PT_results2", f"{lon} , {lat} Merged_Transformed.csv")
            merged_df.to_csv(file_path,index = False, float_format='%.6f')
            print("registred")

wrapperFunc(path_var, date_start, date_end, var_names,"prec_MIROC6_historico_-73.125_-4.902686531826539.csv")
# directory_path = 'miroc6_historico'
#
# # List all entries in the directory
# entries = os.listdir(directory_path)
#
# # Filter out directories, leaving only files
# files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
#
# for file_name in files:
#     try:
#         print(file_name)
#         rapperFunc(path_var, date_start, date_end, var_names, file_name)
#     except:
#         continue
