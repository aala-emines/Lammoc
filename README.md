Before running the code, the working directory should contain MIROC6 historical data and Alexandro's .netcdf files, as well as directories named PT-results and P_bias to store generates .csv files.

- **PT-Monthly.py :**
  
responsible of extracting Observed data from Alexandro's work, applying Power-transformation on MIROC6 data, and computing P_Bias for a specific pixel.
the main function is WrapperFunc

- **Validation.py :**
  
similar to PT-Monthly, but applies Power transformation on the period 1961-2012 and validates the corrected data on 2013-2014.

- **Validation-10years.py :**
  
similar to the previous file. It goes through all 10-years sets of the dataset and validates on the last two years of each set for each pixel.

- **Validation-analisis.py:**
  
a file to generate graphs of PT parameters and P_bias values of the validation files

- **PT_all_grid** :
  a file to apply power transformation over all available pixels over Brazil. Parameters are fitted over the period 1961 - 2014.

Here is how the directories-tree looks like so far : 

  ├── PT_results

  │   ├── validation-trend-amazonia
  
  │   └── validation-trend-mata atlantica
  
  ├── P_bias
  
  │   ├── validation
  
  │   ├── validation-trend-amazonia
  
  │   └── validation-trend-mata atlantica
  
  ├── miroc6_historico
  
  ├── params_per_pixel
  └── pr_Tmax_Tmin_NetCDF_Files
