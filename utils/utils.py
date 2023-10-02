import os
from pathlib import Path
import sys
from typing import List

import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append('../PDM_radar_generation_3/')
from improved_diffusion.datasets_hdf5 import load_data_hdf5
from improved_diffusion.datasets_image import load_data


def read_model_metadata(
        url_json: str
        ) -> dict:
    """
    Reads model metadata in a .json file into a dictionary
    """
    with open(url_json, 'r') as json_model:
        metadata = json.load(json_model)

    return metadata


def load_args_dic(
        url_model_path: str
        ) -> dict:
    """
    Reads and returns model args dictionary

    Args:
        url_model_path (str): path to model file (*.pt)

    Returns:
        dict: dictionary of model arguments
    """
    if not os.path.exists(url_model_path):
        raise ValueError('The specified model path does not exist.')

    # Get metadata JSON file
    url_metadata = f'{Path(url_model_path).parent}/metadata.json'
    if not os.path.exists(url_metadata):
        raise ValueError('There is no metadata.json file associated with this model.')

    return read_model_metadata(url_metadata)


def create_dataloader(
        data_dir: str,
        data_type: str = 'hdf5',
        num_channels: int = 1,
        image_size: int = 128,
        batch_size: int = 32,
        class_cond: bool = False,
        num_class: int = None,
        list_keys_hdf5_original: List[str] = ['wrap', 'unwrap'],
        key_data: str = 'unwrap',
        key_other: str = None,
        crop: bool = False
        ) -> DataLoader:
    '''
    Creates a DataLoader given a data directory.

    Inputs:
    ------
        data_dir (str): path to the data folder

        data_type (str): either "image" or "hdf5",
            depending on the way the data is stored

        num_channels (int): number of channels of the input image.

        image_size (int): the size to which images will be loaded.
            The hdf5 should hence contain the appropriately sized
            images.

        batch_size (int): the batch size of each returned pair.

        class_cond (bool): if True, include a "y" key in returned dicts
            for class label. If classes are not available and this is
            true, an exception will be raised.

            Assume classes are the first part of the filename, before an
            underscore.

        num_class (int): number of data classes

        list_keys_hdf5_original: see Hdf5Dataset.

        key_data: see Hdf5Dataset.

        key_other: see Hdf5Dataset.
    '''
    if not os.path.exists(data_dir):
        raise ValueError("The specified data directory does not exist.")

    if data_type not in {'image', 'hdf5'}:
        raise ValueError('Incorrect data_type.')

    if data_type == 'hdf5':

        loader = load_data_hdf5(
            url_hdf5_folder=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_channels=num_channels,
            list_keys_hdf5_original=list_keys_hdf5_original,
            key_data=key_data,
            key_other=key_other)

    elif data_type == 'image':

        loader = load_data(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_channels=num_channels,
            class_cond=class_cond,
            num_class=num_class,
            crop=crop,
            droplast=False)

    return loader


def check_inputs(args_dic: dict) -> None:
    """
    Check the validity of model training inputs.

    Args:
        args_dic (dict): dictionary of train() arguments
    """
    if args_dic['compute_val'] and args_dic['data_val_dir'] is None:
        raise ValueError('A valid "data_val_dir" is required.')

    if args_dic['class_cond'] and args_dic['num_class'] is None:
        raise ValueError('Please indicate the number of classes.')

    if not os.path.exists(args_dic['data_train_dir']):
        raise ValueError('"data_train_dir" does not exist!')

    if not os.path.exists(args_dic['data_val_dir']):
        raise ValueError('"data_val_dir" does not exist!')

    if args_dic['schedule_sampler'] not in {"uniform", "loss-second-moment"}:
        raise ValueError('Unrecognised schedule sampler.')

    if args_dic['noise_schedule'] not in {'linear', 'cosine'}:
        raise ValueError('Unrecognised noise_schedule.')

    if args_dic['loss_name'] not in {'mse', 'rescaled_mse', 'kl', 'rescaled_kl'}:
        raise ValueError("loss_name should be one of 'mse', 'rescaled_mse', 'kl' or 'rescaled_kl'")

    if args_dic['output_type'] not in {'epsilon', 'x_start', 'x_previous'}:
        raise ValueError("output_type should be one of 'epsilon', 'x_start' or 'x_previous'")

    if args_dic['batch_size'] < args_dic['microbatch']:
        raise ValueError('"batch_size" should be larger than "microbatch".')

    if args_dic['type_dataset'] not in {'image', 'hdf5'}:
        raise ValueError('Only load data of image or hdf5 file types.')


def save_sample_to_png(
        url_npz: str,
        idx: int,
        url_save_folder: str,
        name_save_png: str,
        class_cond: bool = True,
        plot: bool = False
        ) -> None:
    '''
    Converts a sample in an .npz file (indexed by idx)
    to a .png file.

    Inputs:
    ------
        url_npz (str): path to .npz file

        idx (int): index of sample in .npz file to save

        url_folder (str): folder where to save the .png file

        name_save_png (str): file name

        class_cond (bool): if True, adds class label to the file name

        plot (bool): whether to plot each sample before saving
    '''
    if not url_npz.endswith('.npz'):
        raise ValueError('Please indicate a .npz file!')

    arr_npz = np.load(url_npz)
    arr = np.array(arr_npz['arr_0'])
    arr = arr[idx, :, :, :]

    if class_cond:
        if 'arr_1' not in [key for key in arr_npz.keys()]:
            raise ValueError('The field arr_1 for the label information not in the samples npz.')

        arr_label = np.array(arr_npz['arr_1'])
        label = arr_label[idx]
        url_save_png = f'{url_save_folder}/{label}_{name_save_png}.png'
    else:
        url_save_png = f'{url_save_folder}/{name_save_png}.png'

    if plot:
        plt.imshow(arr, vmin=-1, vmax=1)
        plt.show()

    # Normalise to [0, 255]
    arr = np.clip(arr, -1, 1)
    arr_norm = 255. * (arr + 1.) / 2.
    im = Image.fromarray((arr_norm[:, :, 0] * 1).astype(np.uint8)).convert('L')
    im.save(url_save_png)


def save_samples_to_hdf5(
        url_npz: str,
        url_folder: str,
        name_save_hdf5: str
        ) -> None:
    '''
    Saves the contents of a .npz file to a HDF5 file.

    Inputs:
    ------
        url_npz (str): path to .npz file

        url_folder (str): folder where to save the HDF5 file

        name_save_hdf5 (str): file name
    '''
    if not url_npz.endswith('.npz'):
        raise ValueError('Please indicate a .npz file!')
    if not name_save_hdf5.endswith('.hdf5'):
        raise ValueError('name_save_hdf5 must be a .hdf5 file!')
    if not os.path.exists(url_folder):
        os.makedirs(url_folder, exist_ok=True)

    arr_npz = np.load(url_npz)
    arr = np.array(arr_npz['arr_0'])
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[:, :, :, 0]

    # Write to HDF5
    with h5py.File(f'{url_folder}/{name_save_hdf5}', 'w') as f:
        f.create_dataset('data', shape=arr.shape)
        f['data'][:] = arr
