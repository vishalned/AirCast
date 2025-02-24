import xarray as xr
import os

def merge_nc_files():
    # surface var
    for year in range(2003, 2019):
        print(year)
        ds1 = xr.open_dataset(f'../CAMS_download/chemical_data/surface_var/{year}_0.nc')
        ds2 = xr.open_dataset(f'../CAMS_download/chemical_data/surface_var/{year}_1.nc')

        merged = xr.merge([ds1, ds2])

        merged.to_netcdf(f'../CAMS_download/chemical_data/surface_var/{year}.nc')


        # os.remove(f'../CAMS_download/chemical_data/surface_var/{year}_0.nc')
        # os.remove(f'../CAMS_download/chemical_data/surface_var/{year}_1.nc')


    # pressure var
    for year in range(2003, 2019):
        print(year)
        ds1 = xr.open_dataset(f'../CAMS_download/chemical_data/pressure_var/{year}_0.nc')
        ds2 = xr.open_dataset(f'../CAMS_download/chemical_data/pressure_var/{year}_1.nc')

        merged = xr.merge([ds1, ds2])

        merged.to_netcdf(f'../CAMS_download/chemical_data/pressure_var/{year}.nc')


        # os.remove(f'../CAMS_download/chemical_data/pressure_var/{year}_0.nc')
        # os.remove(f'../CAMS_download/chemical_data/pressure_var/{year}_1.nc')


def split_nc_files():
    '''

    The data downloaded from CAMS is one for each year, and many variables are in one file.
    We want to recreate this, by creating a folder for each variable, and in each folder, we have a file for each year.

    '''

    path = '../CAMS_download/chemical_data/surface_var/'
    output_path = '../CAMS_download/chemical_data/surface_var_split/'
    os.makedirs(output_path, exist_ok=True)

    for year in range(2003, 2019):
        print(year)
        ds = xr.open_dataset(f'{path}{year}.nc')

        for var in ds.data_vars:
            os.makedirs(f'{output_path}{var}', exist_ok=True)
            var_ds = ds[var]
            var_ds.to_netcdf(f'{output_path}{var}/{year}.nc')

    path = '../CAMS_download/chemical_data/pressure_var/'
    output_path = '../CAMS_download/chemical_data/pressure_var_split/'

    for year in range(2003, 2019):
        print(year)
        ds = xr.open_dataset(f'{path}{year}.nc')

        for var in ds.data_vars:
            os.makedirs(f'{output_path}{var}', exist_ok=True)
            var_ds = ds[var]
            var_ds.to_netcdf(f'{output_path}{var}/{year}.nc')


def run_regrid():
    '''
    a function to call the python script regrid.py for all directories
    '''

    surface_var = '../CAMS_download/chemical_data_raw/surface_var_split/'
    pressure_var = '../CAMS_download/chemical_data_raw/pressure_var_split/'

    output_dir = '../CAMS_download/chemical_data_5.625/'

    for var in os.listdir(surface_var):
        print(f'----------------- Running regrid for surface variable: {var} -----------------')
        os.system(f'python src/data_preprocessing/regrid.py --input_fns {surface_var}{var}/* --output_dir {output_dir}{var} --ddeg_out 5.625')

    for var in os.listdir(pressure_var):
        print(f'----------------- Running regrid for pressure variable: {var} -----------------')
        os.system(f'python src/data_preprocessing/regrid.py --input_fns {pressure_var}{var}/* --output_dir {output_dir}{var} --ddeg_out 5.625')



import numpy as np

def find_bin_index(data, bins):

  data_array = np.array(data)
  bins_array = np.array(bins)

  # Use np.digitize to find bin indices
  bin_indices = np.digitize(data_array, bins_array, right=True) - 1
  bin_indices[bin_indices == len(bins) - 1] = len(bins) - 2 # incase anything is more than the last bin

  return bin_indices.tolist()

# Example usage
bins = [0, 5, 10, 15]
counts = [10, 20, 5]  # One less element than bins
data = [2, 8, 12, 15]

bin_indices = find_bin_index(data, bins)
print(bin_indices)  # Output: [0, 1, 2, 2]

# Use counts based on bin indices
for i, data_point in enumerate(data):
  bin_index = bin_indices[i]
  count = counts[bin_index]
  print(f"Data point: {data_point}, Bin index: {bin_index}, Count: {count}")


# run_regrid()

#split_nc_files()

#merge_nc_files()        


