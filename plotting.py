import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from tqdm import tqdm
import climax.utils.data_utils as data_utils

CHANNEL_NAMES = [
    # "co_50",
    # "co_250",
    # "co_500",
    # "co_600",
    # "co_700",
    # "co_850",
    # "co_925",
    # "go3_50",
    # "go3_250",
    # "go3_500",
    # "go3_600",
    # "go3_700",
    # "go3_850",
    # "go3_925",
    # "no_50",
    # "no_250",
    # "no_500",
    # "no_600",
    # "no_700",
    # "no_850",
    # "no_925",
    # "no2_50",
    # "no2_250",
    # "no2_500",
    # "no2_600",
    # "no2_700",
    # "no2_850",
    # "no2_925",
    # "so2_50",
    # "so2_250",
    # "so2_500",
    # "so2_600",
    # "so2_700",
    # "so2_850",
    # "so2_925",      
    "pm1",
    "pm10",
    "pm2p5",
    # "tcco",
    # "tc_no",
    # "tcno2",
    # "tcso2",
    # "gtco3",
]


def load_npz_files(folder_path, variable_name=''):
    data = []
    file_path_lists = []
    for filename in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, filename)):
            for filename2 in os.listdir(os.path.join(folder_path, filename)):
                if filename2.endswith('.npz') and 'climatology' not in filename2:
                    file_path_lists.append(os.path.join(folder_path, filename, filename2))

    for file_path in tqdm(file_path_lists, desc='loading files'):
        data.append(np.load(file_path)[variable_name])

    return np.concatenate(data, axis=0)  



def chem_transform(inp):
    import torch
    import numpy as np
    
    # this is a chemical transformation followed from the aurora paper
    # inp = torch.from_numpy(inp)
    # c1, c2 = 0.5, 0.5
    # x = inp
    # x = c1 * torch.min(x, torch.tensor(2.5)) + c2 * (np.log(torch.max(x, torch.tensor(1e-4))) - np.log(1e-4)) / np.log(1e-4)

    # we also try a new log transformation
    x = torch.from_numpy(inp)
    # x = np.log(torch.max(x, torch.tensor(1e-4)))
    # x = (np.log(torch.max(x, torch.tensor(1e-4))) - np.log(1e-4)) / np.log(1e-4)
    return x.numpy()




# def plot_channel_distributions(data, channel_names, max_clip=None):
#     """
#     Plots the distribution of values for each channel in a large air pollution dataset.
#     """

#     n_channels = data.shape[1]
#     fig, axes = plt.subplots(nrows=1, ncols=n_channels, figsize=(12, 6))
#     for i, channel in enumerate(data.T):
#     # Calculate optimal bin width using Freedman-Diaconis rule
#         q1, q3 = np.percentile(channel, [25, 75])
#         iqr = q3 - q1
#         bin_width = 2 * iqr / np.sqrt(len(channel))

#         # Clip data if necessary
#         if max_clip is not None:
#             channel = np.clip(channel, 0, max_clip)

#         # Count occurrences in bins
#         counts, bins = np.histogram(channel, bins=np.arange(min(channel), max(channel) + bin_width, bin_width))

#         # Handle potential overflow into the last bin due to clipping
#         if max_clip is not None and counts[-1] > 0:
#             counts[-1] += np.sum(channel > max_clip)  # Add overflow count to last bin
#             bins = bins[:-1]  # Remove the last bin edge as it's not used

#         # Plot the histogram
#         print(bins.shape, counts.shape, bin_width)
#         axes[i].bar(bins, counts, width=bin_width, edgecolor='black')
#         axes[i].set_title(channel_names[i])
#         axes[i].set_xlabel(channel_names[i] + ' value')
#         axes[i].set_ylabel('Frequency')

#         plt.tight_layout()
#         plt.savefig('channel_distributions.png')


def compute_norm(data):
    """

    This is a special function to compute the spatial norm as in the aurora paper. We compute the average of the sum of max values in each timestep (time step is the channel). 
    We then divide this by 2

    Data is of shape (timestep, 1, 32, 64)
    """

    mean = 0
    std = np.mean(np.max(data, axis=(2, 3)))/2
    print(f'mean: {mean}')
    print(f'std: {std}')
    return (data - mean) / std



def plot_single_distribution(data, channel_name, max_clip=None, transform=False):
    """
    Plots the distribution of values for a single channel in a large air pollution dataset.
    """

    if transform:
        # we want to apply the chemical transformations to the data
        # we can only apply it on the data that is normalized and not clipped.
        print('applying chemical transformations to the data')
        data = compute_norm(data)
        print(f'after normalization, data min: {data.min()}')
        print(f'after normalization, data max: {data.max()}')
        data = chem_transform(data)
        print(f'after transformation, data min: {data.min()}')
        print(f'after transformation, data max: {data.max()}')

    else:
        # for the raw data we only clip the values to remove negative values
        # typically negative values are not possible in the data, so we clip the data to 0
        data = np.clip(data, 0, max_clip)
    data = data.astype(np.float16)

    # plotting only the MENA region distribution
    # ddeg_out = 5.625
    # lat = np.arange(-90+ddeg_out/2, 90, ddeg_out)
    # lon = np.arange(0, 360, ddeg_out)
    # region_info  = data_utils.get_region_info('MENAreg', lat, lon, patch_size=2)
    # data = data[:, :, region_info['min_h']:region_info['max_h']+1, region_info['min_w']:region_info['max_w']+1]




    bin_edges = np.histogram_bin_edges(data, bins='auto')
    print(f'bin width: {bin_edges[1] - bin_edges[0]}')

    counts, bins = np.histogram(data, bins=bin_edges)
    print(f'number of bins: {len(bins)}')
    # bins = bins[:-1]  # Remove the last bin edge as it's not used


    # to make the visualization look good, if the max count and next max count are too far apart, we make that count as second max and change that label to '>second_max'
    max_count = counts.max()
    second_max_count = np.sort(counts)[-2]
    if max_count / second_max_count > 1e1:
        print('large gap between the max and second max values')
        print(f'max_count: {max_count}')
        print(f'second_max_count: {second_max_count}')
        # we only make the count index with max count as a value larger than second max but not too large
        second_max_index = np.where(counts == max_count)[0][0]
        counts[second_max_index] = second_max_count * 2
    else:
        max_count = None

        
    data = data.flatten()

    plt.hist(data, bins=bin_edges, edgecolor='black')
    if max_count is not None:
        plt.ylim(0, second_max_count * 2)
    plt.title(f'{channel_name} distribution for MENA region')
    plt.xlabel(channel_name + ' value')
    if max_count is not None:
        plt.ylabel('Frequency clipped')
    else:
        plt.ylabel('Frequency')

    # we also plot the y-axis in log scale
    plt.yscale('log')

    # plt.tight_layout()
    path = f'./single_channel_distributions/log_transformations_orig/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}{channel_name}.png')


    
    plt.clf()
    




if __name__ == '__main__':
    # data = np.random.rand(1000, 3)  
    folder_path = "./aircast_data/"


    for i, channel in enumerate(CHANNEL_NAMES):
        data = load_npz_files(folder_path, variable_name=channel)
        # max_clip = 1  hard coded max_clip value
        max_clip = np.percentile(data, 99)

        # plot_channel_distributions(data, channel, max_clip)
        print(f'--------------------- plotting single distribution for {channel} ---------------------')
        print(f'max_clip: {max_clip}')
        print(f'data shape: {data.shape}')
        print(f'data min: {data.min()}')
        print(f'data max: {data.max()}')
        plot_single_distribution(data, channel, max_clip, transform = True)
        print('------------------------------------------------------------------------------------')