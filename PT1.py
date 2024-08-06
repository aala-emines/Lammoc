import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Define your power function
def power_function(x, a, b):
    return a * x ** b


# Load data from CSV file
df = pd.read_csv('[2.1011514025789793] , [-33.75] merged.csv')

# Assuming 'time' column is in datetime format, extract year
df['year'] = pd.to_datetime(df['time']).dt.year

# Initialize empty list to store fitted parameters and transformation results
fitted_params = []
transformed_MIROC6_Pr = []
a_used = []
b_used = []

# Iterate over each year
for year in df['year'].unique():
    # Filter data for the current year
    year_data = df[df['year'] == year]

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

# Add transformed MIROC6_Pr, a_used, and b_used columns to DataFrame
df['Transformed_MIROC6_Pr'] = transformed_MIROC6_Pr
df['a'] = a_used
df['b'] = b_used

# Print or further process the updated DataFrame
print(df[['time', 'MIROC6_Pr','obs_Pr', 'Transformed_MIROC6_Pr', 'a', 'b']])
