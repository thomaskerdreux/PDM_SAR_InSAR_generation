import os
from typing import List, Tuple

import blobfile as bf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def list_image_files_recursively(
        data_dir: str):
    """
    List image files in a data directory.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results


def load_data_lowres(
        *,
        data_dir: str,
        batch_size: int,
        image_size_lr: int,
        image_size_hr: int,
        num_channels: int,
        class_cond: bool = False,
        num_class: int = None,
        deterministic: bool = False,
        crop: bool = False,
        droplast: bool = True
        ):
    """
    Creates a generator over (images, kwargs) pairs given a dataset.
    Reads image of size 'image_size_hr' in the dataset and returns
    a downscaled image of size 'image_size_lr'.

    Each image is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    Inputs:
    -------
        data_dir (str): a dataset directory.

        batch_size (int): the batch size of each returned pair.

        image_size_lr (int): the size to which images are resized.

        image_size_hr (int): the original image size

        num_channels (int): num of channels of the input image.

        class_cond (bool): if True, include a "y" key in returned dicts
            for class label. If classes are not available and this is
            true, an exception will be raised.

            Assume classes are the first part of the filename, before an
            underscore.

        num_class (int): number of data classes

        deterministic (bool): if True, yields results in a
            deterministic order.

        crop (bool): if True, randomly crops the image
            to the desired image_size

        droplast (bool): if True, drops the last few examples of the
            dataset to work with a whole number of batches.
    """
    # Check inputs
    if not data_dir:
        raise ValueError("Unspecified data directory!")
    if not os.path.exists(data_dir):
        raise ValueError("The specified data directory does not exist!")

    all_files = list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        if num_class != len(sorted_classes):
            print('Difference between the number of classes when reading the data and the input number of classes.')

    dataset = ImageResizeDataset(
        image_size_lr,
        image_size_hr,
        num_channels,
        all_files,
        classes=classes,
        crop=crop)

    return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=droplast,
            )
            if deterministic
            else DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=droplast,
            )
        )


class ImageResizeDataset(Dataset):

    def __init__(
            self,
            low_resolution: int,
            high_resolution: int,
            num_channels: int,
            image_paths: List[str],
            classes: List[str] = None,
            plot: bool = False,
            crop: bool = False):
        '''
        Inputs:
        -------
            low_resolution (int): desired size for the images, not
                necessarily the native resolution.

            high_resolution (int): size at which images are read
                from the dataset.

            num_channels (int): num of channels of the input
                image.

            images_paths (List[str]): list of all the urls
                associated with the images.

            classes (List[str]): if not None, it is the
                label associated to each image.

            crop (bool): if True, randomly crops the image
                to the desired image resolution.
        '''
        super().__init__()
        self.low_resolution = low_resolution
        self.high_resolution = high_resolution
        self.num_channels = num_channels
        self.local_images = image_paths
        self.local_classes = None if classes is None else classes
        self.plot = plot
        self.crop = crop

    def __len__(self):
        return len(self.local_images)

    def __getitem__(
            self,
            idx: int) -> Tuple[np.ndarray, dict]:
        """
        It outputs the image in format (C, H, W) along with a
        dictionary for the label.
        """

        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        if self.crop:
            width, height = pil_image.size
            if width < self.high_resolution or height < self.high_resolution:
                raise ValueError('The data cannot be cropped to the desired resolution!')
            # Randomly crop a portion of the image
            left, bottom = np.random.randint(0, width - self.high_resolution),\
                np.random.randint(0, height - self.high_resolution)
            pil_image = pil_image.crop((left, bottom,
                                        left + self.high_resolution,
                                        bottom + self.high_resolution))

        else:
            while min(*pil_image.size) >= 2 * self.high_resolution:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size),
                    resample=Image.BOX
                )

            scale = self.high_resolution / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size),
                resample=Image.BICUBIC
            )

        # This step converts the image to an array in [0, 255]
        if self.num_channels == 3:
            arr = np.array(pil_image.convert("RGB"))
        elif self.num_channels == 1:
            # Convert to grayscale
            arr = np.expand_dims(np.array(pil_image.convert('L')), axis=2)
        else:
            raise ValueError('We require either 1 or 3 channels.')

        if arr.shape != (self.high_resolution, self.high_resolution, self.num_channels):
            raise ValueError('The current image is not of the right size.')

        # This step rescales the array to [-1, 1]
        arr = arr.astype(np.float32) / 127.5 - 1

        # This step reorders the array dimensions
        # from (N, N, num_channels) to (num_channels, N, N)
        arr = np.transpose(arr, [2, 0, 1])

        # This step interpolates the data using PyTorch
        arr = torch.tensor(arr)
        arr = F.interpolate(arr.unsqueeze(0), self.low_resolution, mode="area").squeeze(0)

        if self.plot:
            plt.imshow(arr)
            plt.colorbar()
            plt.show()

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return arr, out_dict
