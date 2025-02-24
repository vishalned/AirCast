
'''
    Some functions to process the data to be compatible with the pm2.5 data. Also includes visualization of the pm2.5 data.
'''
CHANNEL_NAMES = [
    "co_50",
    "co_250",
    "co_500",
    "co_600",
    "co_700",
    "co_850",
    "co_925",
    "go3_50",
    "go3_250",
    "go3_500",
    "go3_600",
    "go3_700",
    "go3_850",
    "go3_925",
    "no_50",
    "no_250",
    "no_500",
    "no_600",
    "no_700",
    "no_850",
    "no_925",
    "no2_50",
    "no2_250",
    "no2_500",
    "no2_600",
    "no2_700",
    "no2_850",
    "no2_925",
    "so2_50",
    "so2_250",
    "so2_500",
    "so2_600",
    "so2_700",
    "so2_850",
    "so2_925",      
    "pm1",
    "pm10",
    "pm2p5",
    "tcco",
    "tc_no",
    "tcno2",
    "tcso2",
    "gtco3",
]


def process_era5():
    '''
        This code was used to get 3 hour data from era5. This is done to match with the pm2.5 data.
    '''
    
    
    import xarray as xr
    import os

    # Step 1: Load the data
    directory = "./"
    output_directory = "../new/"

    os.makedirs(output_directory, exist_ok=True)

    # List all NetCDF files in the directory
    dirs = os.listdir(directory)
    for j in dirs:
        if j in os.listdir(output_directory) or j in "constants":
            print(j, " exists")
            continue
        os.makedirs(os.path.join(output_directory, j), exist_ok=True)
        files = [f for f in os.listdir(j) if f.endswith('.nc')]
        
        # Iterate over each file
        for file in files:
            # Load the dataset
            ds = xr.open_dataset(os.path.join(j, file))

            # Step 2: Resample the data
            # Select every 3rd hour starting from 00:00:00 to 21:00:00
            ds_resampled = ds.sel(time=ds.time.dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21]))

            # Step 3: Save the modified dataset
            output_filename = os.path.join(output_directory, j, file)
            ds_resampled.to_netcdf(output_filename)

            # Close the dataset
            ds.close()


def visualize_pm2p5():
    '''
        This code was used to visualize the pm2.5 data in the form of a map.

        Works on the xarray dataset.
    '''


    import xarray as xr

    from zipfile import ZipFile

    # Libraries for reading and working with multidimensional arrays
    import numpy as np
    import xarray as xr


    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Disable warnings for data download via API
    import urllib3 
    urllib3.disable_warnings()



    ds = xr.open_dataset("5.625deg_3hr/pm2p5/2018.nc")
    da = ds["pm2p5"][500]

    dt_values = da["time"].values.astype(str)
    date = dt_values.split('T')[0]
    time = dt_values.split('T')[1][:-10]



    print("--- printing min and max ---")
    print(da.min(), da.max())

    print("--- printing shape ---")
    print(da.shape)

    fig = plt.figure(figsize=(15, 10))

    # create the map using the cartopy PlateCarree projection
    ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())

    # Add lat/lon grid
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # Set figure title
    ax.set_title(f'date:{date}, time:{time}', fontsize=12) 

    # Plot the data
    im = plt.pcolormesh(da.lon, da.lat, da[:,:], cmap='YlOrRd', vmin=0, vmax=4e-7) 

    # Add coastlines
    ax.coastlines(color='black') 

    # Specify the colourbar, including fraction of original axes to use for colorbar, 
    # and fraction of original axes between colorbar and new image axes
    cbar = plt.colorbar(im, fraction=0.025, pad=0.05) 

    # Define the colourbar label
    cbar.set_label('pm2p5') 

    # Save the figure
    fig.savefig(f'./{date}T{time}.png')



def convert_pm2p5():
    '''
        This function is used to convert the pm2.5 data from kg per cubic meter to microgram per cubic meter which is the default real-world unit for pm2.5 data.
        1kg = 1e9 mcg 

        We also do the conversion assuming the data is already in npy format. Hence, we load each data and multiply it by 1e9 to get the microgram per cubic meter data.

        Works on the npz files.
    '''

    import numpy as np
    import os


    path = '../5.625deg_3hr_8shards_npz/'

    for file in os.listdir(path):
        if file.endswith('.npz'):
            data = np.load(os.path.join(path, file))
            if 'pm2p5' in data.keys():
                tmp = data['pm2p5'] * 1e9
                # remove the old key
                data = {key: value for key, value in data.items() if key != 'pm2p5'}
                data['pm2p5'] = tmp
                np.savez_compressed(os.path.join(path, file), **data)
            else:
                print(f'pm2p5 not in {path, file}')
        
        # if it is a directory
        elif os.path.isdir(os.path.join(path, file)):
            for file2 in os.listdir(os.path.join(path, file)):
                if file2.endswith('.npz'):
                    data = np.load(os.path.join(path, file, file2))
                    if 'pm2p5' in data.keys():
                        tmp = data['pm2p5'] * 1e9
                        # remove the old key
                        data = {key: value for key, value in data.items() if key != 'pm2p5'}
                        data['pm2p5'] = tmp

                        np.savez_compressed(os.path.join(path, file, file2), **data)
                    else:
                        print(f'pm2p5 not in {file2}')
                else:
                    print(f'{path,file2} is not a npz file')

                    


def interpolate_pm2p5():
    '''
    ERA5 data is hourly, while the pm2.5 data is 3 hourly. This function is to interpolate the pm2.5 data to match the era5 data.

    works on the xarray dataset
    '''

    import numpy as np
    import xarray as xr
    import os

    for var in os.listdir('../CAMS_download/chemical_data_5.625/'):
        print('processing: ', var)
        path = f"../CAMS_download/chemical_data_5.625/{var}"
        new_path = f"../CAMS_download/chemical_data_5.625_interpolated/{var}"

        os.makedirs(new_path, exist_ok=True)

        for nc in os.listdir(path):
            print('processing file ',nc)
            if nc.endswith('.nc'):
                ds = xr.open_dataset(os.path.join(path, nc))

                # we interpolate the data to make it 1 hourly based on the 3 hour data.
                ds = ds.resample(time='1H').interpolate('linear')
                ds.to_netcdf(os.path.join(new_path, nc))
            else:
                print(f'{nc} is not a netcdf file')

def chem_transform(inp):
    import torch
    import numpy as np

    inp = torch.from_numpy(inp)
    c1, c2 = 0.5, 0.5
    x = inp
    x = c1 * torch.min(x, torch.tensor(2.5)) + c2 * (np.log(torch.max(x, torch.tensor(1e-4))) - np.log(1e-4)) / np.log(1e-4)
    return x.numpy()

def plot_stats():
    '''
    A function to plot and calculate all kinds of statistics for the pm2.5 data.
    
    '''

    # we start with the min and max values of the entire dataset
    print('Calculating min and max values for the entire dataset')
    import numpy as np
    import os
    path = '../data/5.625deg_1hr/'

    min_val = 1e9
    max_val = 0
    mean, std = np.load(path+'normalize_mean.npz')['pm2p5'], np.load(path+'normalize_std.npz')['pm2p5']
    for file in os.listdir(path):
        if file.endswith('.npz') and file != 'climatology.npz' and 'normalize' not in file:
            print(f'processing {file}---------------------')
            data = np.load(os.path.join(path, file))
            if 'pm2p5' in data.keys():
                min_val = min(min_val, np.min(chem_transform(data['pm2p5'])))
                max_val = max(max_val, np.max(chem_transform(data['pm2p5'])))
                #min_val = min(min_val, np.min(data['pm2p5']))
                #max_val = max(max_val, np.max(data['pm2p5']))
        
        # if it is a directory
        elif os.path.isdir(os.path.join(path, file)):
            for file2 in os.listdir(os.path.join(path, file)):
                if file2.endswith('.npz') and file2 != 'climatology.npz':
                    print(f'processing {file2}')
                    data = np.load(os.path.join(path, file, file2))
                    if 'pm2p5' in data.keys():
                        tmp = data['pm2p5']
                        tmp = (tmp - mean[0])/std[0]
                        tmp = chem_transform(tmp)
                        min_val = min(min_val, np.min(tmp))
                        max_val = max(max_val, np.max(tmp))
                        #min_val = min(min_val, np.min(data['pm2p5']))
                        #max_val = max(max_val, np.max(data['pm2p5']))

    print(f'min: {min_val}, max: {max_val}')

    '''
    # we now calculate the min and max for each year. The files are named as year_shard.npz
    print('Calculating min and max values for each year')

    years = [i for i in range(2003, 2019)]
    min_vals = {
        year: 1e9 for year in years
    }
    max_vals = {
        year: 0 for year in years
    }

    for file in os.listdir(path):
        if file.endswith('.npz') and file != 'climatology.npz' and 'normalize' not in file:
            print(f'processing {file}')
            year = int(file.split('_')[0])
            data = np.load(os.path.join(path, file))
            if 'pm2p5' in data.keys():
                min_vals[year] = min(min_vals[year], np.min(data['pm2p5']))
                max_vals[year] = max(max_vals[year], np.max(data['pm2p5']))
        
        # if it is a directory
        elif os.path.isdir(os.path.join(path, file)):
            for file2 in os.listdir(os.path.join(path, file)):
                if file2.endswith('.npz') and file2 != 'climatology.npz':
                    print(f'processing {file2}')
                    year = int(file2.split('_')[0])
                    data = np.load(os.path.join(path, file, file2))
                    if 'pm2p5' in data.keys():
                        min_vals[year] = min(min_vals[year], np.min(data['pm2p5']))
                        max_vals[year] = max(max_vals[year], np.max(data['pm2p5']))

    print('min_vals: ', min_vals)
    print('max_vals: ', max_vals)

    '''


    # we now calculate the historgram of values for the entire dataset to see the distribution of values and outliers
    print('Calculating histogram of values for the entire dataset')
    import matplotlib.pyplot as plt

    bins = np.arange(min_val, max_val+0.1, 0.01)
    #bins = np.arange(-50, max_val+500, 100)
    print(min_val, max_val)
    hist = np.zeros(len(bins), dtype=int)
    print('number of bins: ', len(bins))

    for file in os.listdir(path):
        if file.endswith('.npz') and file != 'climatology.npz' and 'normalize' not in file:
            print(f'processing {file}-----------------------')
            data = np.load(os.path.join(path, file))
            if 'pm2p5' in data.keys():
                tmp = data['pm2p5'].flatten()
                tmp = chem_transform(tmp)
                ind = np.digitize(tmp, bins)
                ind = ind + 1

                for index in ind:
                    hist[index] += 1

        # if it is a directory
        elif os.path.isdir(os.path.join(path, file)):
            for file2 in os.listdir(os.path.join(path, file)):
                if file2.endswith('.npz') and file2 != 'climatology.npz':
                    print(f'processing {file2}')
                    data = np.load(os.path.join(path, file, file2))
                    if 'pm2p5' in data.keys():
                        tmp = data['pm2p5'].flatten()
                        tmp = (tmp - mean[0])/std[0]
                        tmp = chem_transform(tmp)
                        ind = np.digitize(tmp, bins)
                        for index in ind:
                            hist[index] += 1


    fig, ax = plt.subplots()
    print(hist)
    print(bins)
    ax.bar(bins, hist, width=0.01)
    plt.xlabel('pm2.5 values in microgram per cubic meter')
    plt.ylabel('Number of pixels')
    plt.title('Distribution of pm2.5 for all pixels')

    plt.savefig('hist_exp1.png')


def clip_pm2p5():
    '''
    A function to clip the pm2.5 data to a certain range. This is done to remove outliers and make the data more uniform.
    '''
    import numpy as np
    import os
    clip_min = 10
    clip_max = 200

    path = '../data/5.625deg_1hr/'

    for file in os.listdir(path):
        if file.endswith('.npz') and file != 'climatology.npz' and 'normalize' not in file:
            print(f'processing {file}')
            data = np.load(os.path.join(path, file))
            if 'pm2p5' in data.keys():
                tmp = np.clip(data['pm2p5'], clip_min, clip_max)
                # remove the old key
                data = {key: value for key, value in data.items() if key != 'pm2p5'}
                data['pm2p5'] = tmp
                np.savez_compressed(os.path.join(path, file), **data)
        
        # if it is a directory
        elif os.path.isdir(os.path.join(path, file)):
            for file2 in os.listdir(os.path.join(path, file)):
                if file2.endswith('.npz') and file2 != 'climatology.npz':
                    print(f'processing {file2}')
                    data = np.load(os.path.join(path, file, file2))
                    if 'pm2p5' in data.keys():
                        tmp = np.clip(data['pm2p5'], clip_min, clip_max)
                        # remove the old key
                        data = {key: value for key, value in data.items() if key != 'pm2p5'}
                        data['pm2p5'] = tmp
                        np.savez_compressed(os.path.join(path, file, file2), **data)


def compute_norm_chem():
    '''
    For chem data, the standard mean and std are not correct since the data is heterogeneous and skewed. Hence we set the mean to 0, and std to the 
    half of spatial max averaged over time, computed on the entire data. Ref Aurora paper from microsoft.
    '''
                
    import numpy as np
    import os

    path = '../data/5.625_full_npz/'
    normalize_mean_fp = os.path.join(path, 'normalize_mean.npz')
    normalize_std_fp = os.path.join(path, 'normalize_std.npz')

    # first we set the mean of pm2.5 to 0
    file = np.load(normalize_mean_fp)
    file = {key: value for key, value in file.items() if key not in CHANNEL_NAMES}
    for channel in CHANNEL_NAMES:
        file[channel] = [0]

    np.savez_compressed(normalize_mean_fp, **file)

    # now we set the std of pm2.5 to half of spatial max averaged over time

    np_arrays = {}

    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            for file2 in os.listdir(os.path.join(path, file)):
                if file2.endswith('.npz') and file2 != 'climatology.npz':
                    data = np.load(os.path.join(path, file, file2))
                    for channel in CHANNEL_NAMES:
                        if channel not in np_arrays:
                            np_arrays[channel] = []
                        np_arrays[channel].append(data[channel])
    stds = {}
    for channel in CHANNEL_NAMES:
        np_arrays[channel] = np.concatenate(np_arrays[channel], axis=0)

        std = np.mean(np.max(np_arrays[channel], axis=(2, 3)))/2
        stds[channel] = std

    print(stds)

    file = np.load(normalize_std_fp)
    file = {key: value for key, value in file.items() if key not in CHANNEL_NAMES}
    for channel in CHANNEL_NAMES:
        file[channel] = [stds[channel]]

    

    np.savez_compressed(normalize_std_fp, **file)








if __name__ == "__main__":
    # process_era5()
    # visualize_pm2p5()
    # convert_pm2p5()
    # interpolate_pm2p5()
    #plot_stats()
    # clip_pm2p5()
    compute_norm_chem()
