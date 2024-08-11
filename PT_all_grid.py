import numpy as np
import xarray as xr
import dask
import pandas as pd
from scipy.optimize import curve_fit
import os

# Exportando dados MENSAIS de todas as variaveis para determinadas posicoes/pontos geograficas.
# Para as variáveis pr e ETo, os valores são os acumulados,
# para as demais, é média dos mês

# periodo para ser exportado
date_start, date_end = '1961-01-01', '2022-12-31'

# set correct path of the netcdf files
path_var = 'pr_Tmax_Tmin_NetCDF_Files/'

# variables names
var_names = ['pr']

def power_function(x, a, b):
    return a * x ** b

def wrapperFunc(path_var, date_start, date_end, var_names, miroc6_filename):
    lat = [eval(miroc6_filename[22:][:-4].split('_')[1])]
    lon = [eval(miroc6_filename[22:][:-4].split('_')[0])]

    # function to read the netcdf files
    def rawData(var2get_xr, var_name2get):
        return var2get_xr[var_name2get].loc[dict(time=slice(date_start, date_end))].sel(
            longitude=xr.DataArray(lon, dims='z'),
            latitude=xr.DataArray(lat, dims='z'),
            method='nearest').values

    # getting data from NetCDF files
    for n, var_name2get in enumerate(var_names):
        print("getting " + var_name2get)
        var2get_xr = var2get_xr = xr.open_mfdataset(path_var + var_name2get + '*.nc', parallel=True).resample(time="ME").sum("time")
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
            df_miroc6 = pd.read_csv('miroc6_historico/' + miroc6_filename)
            df_obs['Date'] = pd.to_datetime(df_obs['Date'])
            print(df_obs)
            df_miroc6['Date'] = pd.to_datetime(df_miroc6['time'])
            merged_df = pd.merge(df_obs, df_miroc6, on='Date', suffixes=('obs', 'MIROC6'))
            merged_df.drop(columns=['Date'], inplace=True)
            merged_df = merged_df[['time', 'pr', 'prec']]
            merged_df.rename(columns={'pr': 'obs_Pr', 'prec': 'MIROC6_Pr'}, inplace=True)
            # Extract year and month from 'time' column
            merged_df['month'] = pd.to_datetime(merged_df['time']).dt.month
            print(merged_df)
            # Initialize empty list to store fitted parameters and transformation results
            fitted_params = []
            transformed_MIROC6_Pr = []
            a_used = []
            b_used = []
            P_bias_per_month_original = {}
            P_bias_per_month_PT = {}
            RMSE_per_month_original = {}
            RMSE_per_month_PT = {}

            # Iterate over each month
            for month in range(1,13):
                # Filter data for the current year and month
                month_data = merged_df[(merged_df['month'] == month)]
                print(month_data)
                # Extract obs_Pr and MIROC6_Pr data for the current year and month
                obs_Pr_data = month_data['obs_Pr'].values
                MIROC6_Pr_data = month_data['MIROC6_Pr'].values

                # Check if there are enough data points to fit the curve
                if len(obs_Pr_data) < 2 or len(MIROC6_Pr_data) < 2:
                    print(f"Not enough data for month {month}. Skipping...")
                    continue
                print(obs_Pr_data, MIROC6_Pr_data)
                # Perform curve fitting
                popt, pcov = curve_fit(power_function, MIROC6_Pr_data, obs_Pr_data, maxfev=2000)

                # Store the fitted parameters
                fitted_params.append({
                    'Month': month,
                    'a': popt[0],
                    'b': popt[1]
                })

            # Transform MIROC6_Pr using the fitted parameters for this month
            for index, row in merged_df.iterrows():
                MIROC6_Pr = row['MIROC6_Pr']
                its_month = row['month']
                a = fitted_params[its_month-1]['a']
                b = fitted_params[its_month - 1]['b']
                # Transform using fitted parameters
                transformed_value = power_function(MIROC6_Pr, a,b)

                # Append transformed value to list
                transformed_MIROC6_Pr.append(transformed_value)

                # Append a and b used for this transformation
                a_used.append(a)
                b_used.append(b)
            # Add transformed MIROC6_Pr, a_used, and b_used columns to DataFrame
            merged_df['Transformed_MIROC6_Pr'] = transformed_MIROC6_Pr
            merged_df['a'] = a_used
            merged_df['b'] = b_used

            fitted_params_df = pd.DataFrame(fitted_params)
            file_path = os.path.join("params_per_pixel",str(lon)+str(lat)+".csv")
            fitted_params_df.to_csv(file_path, index=False)

            for month in range(1,13):
                month_data = merged_df[(merged_df['month'] == month)]
                sum_obs_Pr = month_data['obs_Pr'].sum()
                sum_MIROC6_Pr = month_data['MIROC6_Pr'].sum()
                sum_transformed_MIROC6_Pr = month_data['Transformed_MIROC6_Pr'].sum()
                # Compute P_bias for the current month3
                P_bias_original = ((sum_obs_Pr - sum_MIROC6_Pr) / sum_obs_Pr) * 100
                P_bias_PT = ((sum_obs_Pr - sum_transformed_MIROC6_Pr) / sum_obs_Pr) * 100
                # Store P_bias in the dictionary
                P_bias_per_month_original[month] = P_bias_original
                P_bias_per_month_PT[month] = P_bias_PT

                # Compute RMSE for the current month
                rmse_original = np.sqrt(((month_data['obs_Pr'] - month_data['MIROC6_Pr']) ** 2).mean())
                rmse_transformed = np.sqrt(((month_data['obs_Pr'] - month_data['Transformed_MIROC6_Pr']) ** 2).mean())

                # Store RMSE in the dictionaries
                RMSE_per_month_original[month] = rmse_original
                RMSE_per_month_PT[month] = rmse_transformed
            #convert the dictionary to a DataFrame for easier manipulation or export
            P_bias_df_original = pd.DataFrame(list(P_bias_per_month_original.items()), columns=['Month', 'P_bias_original'])
            P_bias_df_original['P_bias_PT'] = pd.DataFrame(list(P_bias_per_month_PT.items()), columns=['Month', 'P_bias_PT'])['P_bias_PT']
            P_bias_df_original['RMSE_original'] = pd.DataFrame(list(RMSE_per_month_original.items()), columns=['Month', 'RMSE_original'])['RMSE_original']
            P_bias_df_original['RMSE_PT'] = pd.DataFrame(list(RMSE_per_month_PT.items()), columns=['Month', 'RMSE_PT'])['RMSE_PT']
            file_path = os.path.join("P_bias", f"{lon} , {lat} per month P_bias-RMSE.csv")
            P_bias_df_original.to_csv(file_path, index=False, float_format='%.2f')
            file_path = os.path.join("PT_results", f"{lon} , {lat} Merged_Transformed_daily.csv")
            merged_df.to_csv(file_path, index=False, float_format='%.6f')
            print("registred")
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers (mean radius)
    r = 6371.0

    # Calculate the result
    return c * r
df = pd.read_csv('pixels brazil.csv')
def check(target_lon, target_lat, margin = 0.001):
    # Filter the DataFrame to rows with the same longitude
    same_lon_df = df[df['longitude'] == target_lon].reset_index()
    # Find the row with the closest latitude
    closest_row = same_lon_df.iloc[(same_lon_df['latitude'] - target_lat).abs().idxmin()]
    # Calculate the Haversine distance between the target point and the closest point
    distance = haversine(target_lat, target_lon, closest_row['latitude'], closest_row['longitude'])
    # Add the distance to the closest row DataFrame
    closest_row['distance'] = distance
    if closest_row['distance'] <= margin :
        return (target_lon,target_lat)
    else : return False
N = 0
for filename in os.listdir("miroc6_historico"):
    print(N)
    if N>= 505:
        lat = eval(filename[22:][:-4].split('_')[1])
        lon = eval(filename[22:][:-4].split('_')[0])
        print(f"prec_MIROC6_historico_{lon}_{lat}.csv")
        if check(lon,lat):
            wrapperFunc(path_var, date_start, date_end, var_names, f"prec_MIROC6_historico_{lon}_{lat}.csv")
    N += 1
