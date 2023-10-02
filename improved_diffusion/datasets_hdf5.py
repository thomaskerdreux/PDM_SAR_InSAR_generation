import os
import tqdm
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Hdf5Dataset(Dataset):
    '''
    Dataset for hdf5 files of Sentinel-1 data contained
    in a specific folder.
    '''
    def __init__(
            self,
            url_hdf5_folder: str,
            C: int = 1,
            N: int = 128,
            list_keys_hdf5_original: List[str] = ['dates', 'unwrap', 'wrap', 'wrap_count'],
            key_data: str = 'wrap',
            key_other: str = 'dates',
            class_key: str = None):
        '''
        Dataset with tweaked __getitem__() that loads the data
        via contiguous chunks in the same hdf5 file.

        Inputs:
        ------
            url_hdf5_folder (str): url of the folder where the data is stored
                as HDF5 files.

            C (int): number of data channels.

            N (int): size of the image.

            list_keys_hdf5_original (List[str]): the  key list of the HDF5
                containing the data.

            key_data (str): the field containing the data that needs to be
                generated.

            key_other (str): if not None, it is a key in the HDF5 that can
                be efficiently loaded to retrieve the number of examples
                in the HDF5 via the first dimension of the stored array.

            class_key (str): key to access class labels in HDF5 files
        '''

        # Check inputs
        self.C = C
        self.N = N
        self.list_keys_hdf5_original = list_keys_hdf5_original
        if key_data not in list_keys_hdf5_original:
            raise ValueError('the data key is not consistent with the hdf5 keys.')
        if (key_other is not None) and (key_other not in list_keys_hdf5_original):
            raise ValueError('the other key is not consistent with the hdf5 keys.')
        self.key_data = key_data
        self.class_key = class_key

        if not os.path.exists(url_hdf5_folder):
            raise ValueError('The folder that should contains the HDF5 file does not exist.')
        self.url_hdf5_folder = url_hdf5_folder

        l_hdf5 = [f for f in os.listdir(url_hdf5_folder) if f.endswith('.hdf5')]
        if not l_hdf5:
            raise ValueError('There are no HDF5 files in this folder.')

        # Get the length of all the chunks.
        l_size_hdf5 = []

        for url in l_hdf5:
            with h5py.File(url_hdf5_folder + url, 'r') as f:
                if not set(list(f.keys())).issubset(list_keys_hdf5_original):
                    raise ValueError('The keys of the hdf5 are not well matching.')
                # Get the num of examples in this hdf5
                if key_other is not None:
                    size_hdf5 = f[key_other][:].shape[0]
                else:
                    # Unfortunately load the heavy data.
                    size_hdf5 = f[key_data][:].shape[0]
            l_size_hdf5.append(size_hdf5)

        self.l_url_hdf5 = l_hdf5
        self.l_size_hdf5 = l_size_hdf5

    def __len__(self):
        return np.abs(np.sum(np.array(self.l_size_hdf5)))

    def __getitem__(
            self,
            key: tuple
            ) -> Tuple[torch.Tensor, dict]:
        '''
        key is supposely a tuple (index, Tensor_of_index) where Tensor_of_index
        is a Tensor of indices for the hdf5 file number index.

        The output array is of shape (B, C, H, H).
        '''
        # Check the integrity of the key
        if (not isinstance(key, tuple)) or (len(key) != 2):
            raise ValueError('This is a wrong key for getitem.')
        hdf5_idx, idx_in_hdf5 = key  # The hdf5 index and index for that file.

        # Check that the idx are in the range.
        if hdf5_idx not in range(len(self.l_url_hdf5)):
            raise ValueError("hdf5_idx is not correct with respect to the number of hdf5 files.")
        if torch.max(idx_in_hdf5) > self.l_size_hdf5[hdf5_idx] - 1:
            raise ValueError("There is an idx that is much bigger than possible for the current hdf5.")
        if torch.min(idx_in_hdf5) < 0:
            raise ValueError("There is a small ")

        # Check the type of the tuple key
        if (not isinstance(hdf5_idx, int)) or (not isinstance(idx_in_hdf5, torch.Tensor)):
            raise ValueError('Keys of getitem have the wrong type.')
        if idx_in_hdf5.dtype != torch.int64:
            raise ValueError('idx_in_hdf5 should be an integer.')

        # Let us retrieve the data for this key
        url_hdf5 = f'{self.url_hdf5_folder}/{self.l_url_hdf5[hdf5_idx]}'
        if not os.path.exists(url_hdf5):
            raise ValueError('The url hdf5 seems not to exist.')

        with h5py.File(url_hdf5, 'r') as f:

            # Get the sorted indices for efficiency in IO.
            l_idx = list(torch.sort(idx_in_hdf5)[0])

            if f[self.key_data].shape[-2:] != (self.N, self.N):
                raise ValueError('The stored hdf5 files do not have the right image size.')

            # Get the data
            # If single channel
            if len(f[self.key_data].shape) == 3:
                data_batch = np.expand_dims(f[self.key_data][l_idx, :, :], axis=1)
            # If multiple channels
            elif len(f[self.key_data].shape) == 4:
                data_batch = f[self.key_data][l_idx, :, :, :]
            else:
                raise ValueError('Wrong data shape (should be either 3 or 4 dimensions).')
            if self.class_key is not None:
                label_batch = f[self.class_key][l_idx]

        out_dict = {}
        if self.class_key is not None:
            out_dict["y"] = np.array(label_batch, dtype=np.int64)

        return torch.tensor(data_batch), out_dict

    def plot(
            self,
            data_single: np.ndarray,
            url_plot_dest: str = "./images/one_batch",
            id: int = 1):
        '''
        Plots an element ("data_single") of the dataset
        to the "url_plot_dest" folder.
        '''

        if data_single.shape != (self.C, self.N, self.N):
            raise ValueError('Batch data has the wrong shape.')

        if not os.path.exists(url_plot_dest):
            os.makedirs(url_plot_dest, exist_ok=True)

        fig, axs = plt.subplots(
            1, self.C + 1,
            figsize=(10, (self.C + 1)*10))

        for i in range(self.C):
            im = axs[i].imshow(data_single[i, :, :])
            axs[i].set_title(f't {i}')
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.savefig(f'{url_plot_dest}/image_{id}.png')
        plt.close()


class Sampler(torch.utils.data.Sampler):

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int
            ):

        self.dataset = dataset
        self.batch_size = batch_size
        if not hasattr(dataset, 'l_size_hdf5'):
            raise ValueError('The dataset should have the attribute l_size_hdf5')

    def __iter__(self):
        '''
        This sampler aims at selecting contiguous batches of indices.
        '''
        hdf5_indices = [torch.arange(n) for n in self.dataset.l_size_hdf5]
        hdf5_remaining = torch.tensor(self.dataset.l_size_hdf5)
        total_remaining = sum(hdf5_remaining)
        while total_remaining > 0:
            hdf5_idx = torch.multinomial(hdf5_remaining.float(), 1)
            hdf5_avail = hdf5_remaining[hdf5_idx].item()
            n_samples = min(self.batch_size, hdf5_avail)
            yield (hdf5_idx.item(), hdf5_indices[hdf5_idx][hdf5_avail - n_samples : hdf5_avail])
            hdf5_remaining[hdf5_idx] -= n_samples
            total_remaining -= n_samples


def our_collate_fn(samples_collection):
    # We do our own batching, so `samples_collection` is a list of one element
    # which is returned here.
    assert isinstance(samples_collection, (list, tuple)) and len(samples_collection) == 1
    return samples_collection[0]


###
# Monitoring functions
###


def get_range_data(
        url_hdf5_folder: str,
        image_size: int,
        list_keys_hdf5_original: List[str] = ['dates', 'unwrap', 'wrap', 'wrap_count'],
        key_data: str = 'unwrap',
        key_other: str = 'dates',
        num_channels: int = 1
        ):
    '''
    Get the range of the data in the dataset.

    Inputs:
    ------
        url_hdf5_folder (str):

        image_size (int):

        plot (bool): we plot the histograms of the range of data.

        list_keys_hdf5_original, key_data, key_other: see Hdf5Dataset

        num_channels (int): number of data channels
    '''

    # Get the dataloader
    data_loader = load_data_hdf5(
        url_hdf5_folder=url_hdf5_folder,
        batch_size=32,
        image_size=image_size,
        num_channels=num_channels,
        list_keys_hdf5_original=list_keys_hdf5_original,
        key_data=key_data,
        key_other=key_other)

    range_min = 0
    range_max = 0

    l_range_min = []
    l_range_max = []

    for _, res in tqdm.tqdm(enumerate(data_loader)):
        range_min_batch = torch.max(res[0])
        range_max_batch = torch.min(res[0])

        range_min = min(range_min_batch, range_min)
        range_max = max(range_max_batch, range_max)

        # Also store all the values of range for histograms
        l_range_min.append(torch.min(res[0].view(res[0].shape[0], -1), dim=0)[0])
        l_range_max.append(torch.max(res[0].view(res[0].shape[0], -1), dim=0)[0])

    return range_min, range_max

###
# Loader for PDM generation
###


def load_data_hdf5(
        *,
        url_hdf5_folder: str,
        batch_size: int,
        image_size: int,
        num_channels: int,
        class_cond: bool = False,
        list_keys_hdf5_original: List[str] = ['dates', 'unwrap', 'wrap', 'wrap_count'],
        key_data: str = 'wrap',
        key_other: str = 'dates',
        class_key: str = None):
    """
    Creates a generator over (data, kwargs) pairs given a dataset,
    where the data are of shape (B, C, H, H)

    Inputs:
    -------
        url_hdf5_folder (str): a folder containing the hdf5
            that are used for the dataset.

        batch_size (int): the batch size of each returned pair.

        image_size (int): the size to which images will be loaded.
            The hdf5 should hence contain the appropriately sized
            images.

        num_channels (int): number of channels of the input image.

        class_cond (bool): if True, include a "y" key in returned dicts
            for class label. If classes are not available and this is
            true, an exception will be raised. Assume classes are
            stored in one of the HDF5 fields accessed with "class_key".
            Class labels must be consecutive integers.

        list_keys_hdf5_original: see Hdf5Dataset.

        key_data: see Hdf5Dataset.

        key_other: see Hdf5Dataset.

        class_key (str): key to access class labels in HDF5 files
    """
    # Check inputs
    if class_cond and class_key is None:
        raise ValueError('A valid "class_key" is required if class_cond is True')

    dataset = Hdf5Dataset(
        url_hdf5_folder,
        C=num_channels,
        N=image_size,
        list_keys_hdf5_original=list_keys_hdf5_original,
        key_data=key_data,
        key_other=key_other,
        class_key=class_key)

    print(f'Dataset of size {len(dataset)}')

    print('Done with creating the dataset')

    sampler = Sampler(dataset, batch_size)

    return DataLoader(
        dataset, sampler=sampler,
        num_workers=0,
        collate_fn=our_collate_fn,
        shuffle=False,
        drop_last=False)