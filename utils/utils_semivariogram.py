'''
Compute empirical semivariograms on real and generated data samples.

See documentation at
https://pyinterpolate.readthedocs.io/en/latest/usage/tutorials/Semivariogram%20Estimation%20%28Basic%29.html
'''
import os
from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyinterpolate import build_experimental_variogram
import random
import tqdm


def comp_semivariogram(
        url_hdf5_folder: str,
        key_data: str = None,
        N: int = 32,
        num_samples: int = 10000,
        shuffle: bool = True,
        step_dist: int = None,
        max_dist: int = None,
        method: str = 'triangular',
        url_save_csv: str = None
        ) -> dict:
    '''
    Computes the semivariogram on an ensemble of images saved in one or several
    HDF5 files, located in url_hdf5_folder.

    Inputs:
    ------
        url_hdf5_folder (str): path to HDF5 files

        key_data (str): data key in the HDF5 files

        N (int): image size
            (must be consistent with the data in the HDF5 files)

        num_samples (int): number of data samples to use to calculate
            the semivariogram. If smaller than the number of samples available,
            will not return an error but the effective number of samples
            used will be printed.

        shuffle (bool): whether to shuffle the HDF5 files in url_hdf5_folder

        step_dist (int): distance unit step to compute the semivariogram

        max_dist (int): maximum distance at which to compute the semivariogram

        method (str): semivariogram method (see build_experimental_variogram)

        url_save_csv (str): CSV file path to save the semivariogram to

    Output:
    ------
        A dictionary with the following keys:
        - 'lags': the distance lags
        - 'covariance_mean': the empirical covariance averaged across the samples
        - 'covariance_SD': a value such that the 95th confidence interval for the
            covariance is given by ['covariance_mean' +/- 'covariance_SD']
        - 'semi_variance_mean': the empirical semivariance averaged across the samples
        - 'semi_variance_SD': same as 'covariance_SD' for the semivariance
    '''
    if step_dist is None or max_dist is None:
        raise ValueError('You must indicate a valid step_dist and max_dist!')
    if url_save_csv is not None:
        url_save_folder = Path(url_save_csv).parent
        if not os.path.exists(url_save_folder):
            raise ValueError('Problem with url_save, please check!')
        if not url_save_csv.endswith('.csv'):
            raise ValueError('url_save_csv must be a .csv file!')

    l_hdf5_files = [f
                    for f in os.listdir(url_hdf5_folder)
                    if f.endswith('hdf5')]

    # Shuffle files
    if shuffle:
        random.shuffle(l_hdf5_files)

    # Create coordinate array from N
    x_coords_0, y_coords_0 = np.repeat(np.arange(N), N), np.tile(np.arange(N), N)

    # Loop on number of samples
    sample_cnt = 0
    idx_file = 0
    lags = None
    semi_variance, covariance = [], []
    while sample_cnt < num_samples or idx_file < len(l_hdf5_files):

        print(f'Processing file {idx_file}...')

        with h5py.File(f'{url_hdf5_folder}/{l_hdf5_files[idx_file]}', 'r') as file:

            # Shape is (num_examples, N, N)
            data = file[key_data][:]
            sample_cnt += data.shape[0]
            if sample_cnt > num_samples:
                data = data[:-(sample_cnt-num_samples), :, :]
            if data.shape[-2:] != (N, N):
                raise ValueError('Issue in the data shape.')

            # Loop on data examples
            for j in tqdm.tqdm(range(data.shape[0])):

                # NOTE rescale data to [-1, 1] here
                data[j, :, :] = ((data[j, :, :] - np.min(data[j, :, :])) /
                                 (np.max(data[j, :, :]) - np.min(data[j, :, :])))
                data[j, :, :] = 2 * (data[j, :, :] - 0.5)

                values = data[j, :, :].flatten()
                x_coords = x_coords_0[np.isnan(values) == False]
                y_coords = y_coords_0[np.isnan(values) == False]
                values = values[np.isnan(values) == False]
                # Empirical semivariogram
                var_temp = build_experimental_variogram(
                    input_array=np.column_stack([x_coords, y_coords, values]),
                    step_size=step_dist,
                    max_range=max_dist,
                    method=method)
                if lags is None:
                    lags = var_temp.lags
                semi_variance.append(var_temp.experimental_semivariances)
                covariance.append(var_temp.experimental_covariances)
        idx_file += 1

    print(f'Calculated semivariogram with {sample_cnt} samples.')

    # List to arrays
    semi_variance = np.stack(semi_variance, axis=1)
    covariance = np.stack(covariance, axis=1)

    semivariogram = {
        'lags': lags,
        'covariance_mean': np.mean(covariance, axis=1),
        'covariance_SD': (1.96*np.median(covariance, axis=1) /
                          np.sqrt(covariance.shape[1])),
        'semi_variance_mean': np.mean(semi_variance, axis=1),
        'semi_variance_SD': (1.96*np.median(semi_variance, axis=1) /
                             np.sqrt(semi_variance.shape[1]))
    }

    if url_save_csv is not None:
        pd.DataFrame(semivariogram).to_csv(url_save_csv)

    return semivariogram


def read_semivariogram(url_csv: str) -> dict:
    """
    Reads semivariogram from .csv file
    """
    return pd.read_csv(url_csv, sep=',').to_dict(orient='list')


def plot_semivariogram(
        semivars: List[dict],
        colors: List[str],
        labels: List[str] = None,
        plot_labels: bool = True,
        key: str = 'semi_variance'
        ) -> None:
    '''
    Plots one or multiple semivariograms (each with their own label).

    Inputs:
    ------
        semivars (List[dict]): list of semivariogram dictionaries

        colors (List[str]): list of colors for each semivariogram

        labels (List[str]): list of labels for each semivariogram

        plot_labels (bool): whether to add labels to the plot

        key (str): the data key to plot ('semi_variance' or 'covariance')
    '''
    if plot_labels:
        if labels is None:
            labels = [f'Variogram {i}' for i in range(len(semivars))]
        if len(semivars) != len(labels):
            raise ValueError('vars and labels must be of the same length!')
    if len(colors) != len(semivars):
        raise ValueError('vars and colors must be of the same length!')
    if key not in {'semi_variance', 'covariance'}:
        raise ValueError(f"'var' must be either 'semi_variance' or 'covariance', got '{key}'")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    for i, var in enumerate(semivars):
        ax.plot(var['lags'], var[f'{key}_mean'],
                color=colors[i], label=labels[i])
        ax.fill_between(var['lags'],
                        np.array(var[f'{key}_mean']) + 0.5 * np.array(var[f'{key}_SD']),
                        np.array(var[f'{key}_mean']) - 0.5 * np.array(var[f'{key}_SD']),
                        color=colors[i], alpha=0.3)
    plt.grid(ls='--', alpha=0.8)
    plt.xlabel('Distance', fontsize=16)
    if key == 'semi_variance':
        plt.ylabel('$\hat{\gamma}$', fontsize=16)
    elif key == 'covariance':
        plt.ylabel('$\hat{\sigma}$', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    if plot_labels:
        plt.legend(loc=2, fontsize=14)
    plt.show()


def compare_semivariograms(
        url_semivar_real_csv: str,
        url_semivar_gen_csv: str,
        url_save_folder: str,
        url_save_png: str,
        key: str = 'semi_variance'
        ) -> None:
    '''
    Plots semivariograms for real and generated data
    and saves plot to url_save_folder.
    '''
    if not url_save_png.endswith('.png'):
        raise ValueError('url_save_png must be a .png file!')
    if not os.path.exists(url_save_folder):
        raise ValueError(f'{url_save_folder} does not exist!')
    if not os.path.exists(url_semivar_real_csv):
        raise ValueError(f'{url_semivar_real_csv} does not exist!')
    if not os.path.exists(url_semivar_gen_csv):
        raise ValueError(f'{url_semivar_gen_csv} does not exist!')

    # Load semivariograms
    semivar_real = read_semivariogram(url_semivar_real_csv)
    semivar_gen = read_semivariogram(url_semivar_gen_csv)

    plot_semivariogram([semivar_real, semivar_gen],
                       ['blue', 'red'],
                       ['Real data', 'Generated data'],
                       plot_labels=True,
                       key=key)
