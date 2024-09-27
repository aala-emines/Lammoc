This Repository hosts the work that has been done for LAMMOC to help devloping vulnerability maps.
Access obtained datasets via Zenodo : https://zenodo.org/uploads/13317002
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

- **PT_all_grid.py** :
  a file to apply power transformation over all available pixels over Brazil. Parameters are fitted over the period 1961 - 2014.

- **netCDF_access.py** :
  a file hosting simple code to access PT fitted parameters.
  It is preferable to set 'method' to 'nearest'.
  Otherwise, the user can compute rainfall for the registred pixels (included in netCDF_access.py) and then interpolate to get rainfall for unregistred pixels.
  We expect the user to use MIROC6 model as raw precipitation data to be corrected by the fitted parameters of power transormation.

- **SSP_correction** :
  a file used to correct raw MIROC6 forcecast with parameters obtained from the fitting process from PT-Monthly.py

- ** SPI_historico** :
- a file used to compute SPI index based on Transformed data from the directory 

Here is how the directories-tree looks like so far : 

    ├── PT_results - validation - cleaned
    │   ├── validation-trend-amazonia
    │   └── validation-trend-mata atlantica
    ├── PT_results - raw
    │   ├── validation-trend-amazonia
    │   └── validation-trend-mata atlantica
    ├── PT_results - cleaned
    ├── P_bias
    │   ├── P_bias_all_grid - cleaned
    │   ├── P_bias_all_grid raw
    │   ├── validation
    │   ├── validation-trend-amazonia
    │   └── validation-trend-mata atlantica
    ├── miroc6_historical_daily
    ├── miroc6_historico
    ├── miroc6_ssp245
    ├── miroc6_ssp370
    ├── miroc6_ssp585
    ├── params_per_pixel
    ├── pr_Tmax_Tmin_NetCDF_Files
    └── ssp_corr
        ├── ssp_corr_245
        ├── ssp_corr_370
        └── ssp_corr_585
Some folders are cleaned from data referring to pixels that are out of Brazilian territory.

