import pandas as pd
import plotly.express as px
import os
import re
# Directory where the CSV files are located
directory = 'PT_results/validation-trend-amazonia'
# Initialize lists to store data
year_list = []
month_list = []
a_list = []
b_list = []

# Regex pattern to extract year from filename
pattern = re.compile(r'(\d{4})-\d{2}-\d{2}')

# Loop over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)

        # Extract year from filename
        match = pattern.search(filename)
        if match:
            year = int(match.group(1))
        else:
            continue  # Skip files that don't match the pattern

        # Extract values for the first 12 months
        for i in range(12):
            year_list.append(year)
            month_list.append(df['month'].iloc[i])
            a_list.append(df['a'].iloc[i])
            b_list.append(df['b'].iloc[i])

# Create DataFrames for plotting
plot_df_a = pd.DataFrame({'Year': year_list, 'Month': month_list, 'Value': a_list})
plot_df_b = pd.DataFrame({'Year': year_list, 'Month': month_list, 'Value': b_list})

# Plot 'a' values using Plotly
fig_a = px.bar(plot_df_a, x='Year', y='Value', color='Month', title='Values of a for Each Month Over the Years - Amazonia')
fig_a.show()

# Plot 'b' values using Plotly
fig_b = px.bar(plot_df_b, x='Year', y='Value', color='Month', title='Values of b for Each Month Over the Years - Amazonia')
fig_b.show()
# Create a DataFrame for heatmap
heatmap_df_a = pd.pivot_table(plot_df_a, values='Value', index='Year', columns='Month')
heatmap_df_b = pd.pivot_table(plot_df_b, values='Value', index='Year', columns='Month')

# Plot heatmaps
fig_heatmap_a = px.imshow(heatmap_df_a, aspect='auto', title='Heatmap of a Values Over the Years and Months - Amazonia')
fig_heatmap_a.show()

fig_heatmap_b = px.imshow(heatmap_df_b, aspect='auto', title='Heatmap of b Values Over the Years and Months - Amazonia')
fig_heatmap_b.show()

#this part for P_bias ------------------------------------------------------
directory = 'P_bias/validation-trend-amazonia'

# Initialize lists to store data
year_list = []
month_list = []
p_bias_original_list = []
p_bias_pt_list = []

# Regex pattern to extract year from filename
pattern = re.compile(r'(\d{4})-\d{2}-\d{2}')

# Loop over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)

        # Extract year from filename
        match = pattern.search(filename)
        if match:
            year = int(match.group(1))
        else:
            continue  # Skip files that don't match the pattern

        # Extract values for the first 12 months
        for i in range(12):
            year_list.append(year)
            month_list.append(df['Month'].iloc[i])
            p_bias_original_list.append(df['P_bias_original'].iloc[i])
            p_bias_pt_list.append(df['P_bias_PT'].iloc[i])

# Create DataFrames for plotting
plot_df_p_bias_original = pd.DataFrame(
    {'Year': year_list, 'Month': month_list, 'P_bias_original': p_bias_original_list})
plot_df_p_bias_pt = pd.DataFrame({'Year': year_list, 'Month': month_list, 'P_bias_PT': p_bias_pt_list})

# Plot 'P_bias_original' values using Plotly
fig_p_bias_original = px.bar(plot_df_p_bias_original, x='Year', y='P_bias_original', color='Month',
                              title='P_bias_original for Each Month Over the Years - Amazonia')
fig_p_bias_original.show()

# Plot 'P_bias_PT' values using Plotly
fig_p_bias_pt = px.bar(plot_df_p_bias_pt, x='Year', y='P_bias_PT', color='Month',
                        title='P_bias_PT for Each Month Over the Years - Amazonia')
fig_p_bias_pt.show()

# Create a DataFrame for heatmap
heatmap_df_a = pd.pivot_table(plot_df_p_bias_original, values='P_bias_original', index='Year', columns='Month')
heatmap_df_b = pd.pivot_table(plot_df_p_bias_pt, values='P_bias_PT', index='Year', columns='Month')

# Plot heatmaps
fig_heatmap_a = px.imshow(plot_df_p_bias_original, aspect='auto', title='Heatmap of P_bias original Values Over the Years and Months - Amazonia')
fig_heatmap_a.show()

fig_heatmap_b = px.imshow(plot_df_p_bias_pt, aspect='auto', title='Heatmap of P_Bias PT Values Over the Years and Months - Amazonia')
fig_heatmap_b.show()