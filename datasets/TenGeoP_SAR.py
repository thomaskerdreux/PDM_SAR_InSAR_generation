'''
Process the TenGeoP-SARwv dataset
'''
import os

import numpy as np
from PIL import Image
import rasterio
import shutil
import tqdm


def TenGeoP_SAR_to_ImageDataset(
        url_dataset: str,
        url_dataset_image: str
        ) -> None:
    """
    Restructures the downloaded TenGeoP-SARwv dataset
    to fit the script improved_diffusion.datasets_image.py.

    GeoTiffs are converted to .png and the geospatial reference
    is dropped. Each data scene is also normalised to [0, 255].
    """
    if not os.path.exists(url_dataset):
        raise ValueError('The url_dataset does not exist!')
    if not os.path.exists(url_dataset_image):
        os.makedirs(url_dataset_image, exist_ok=True)

    # List the different image classes
    l_classes = os.listdir(url_dataset)

    # Loop on classes
    for image_class in l_classes:

        print(f'Processing class {image_class}...')

        l_class_files = [f for f in os.listdir(f'{url_dataset}/{image_class}') if f.endswith('.tiff')]

        for file in tqdm.tqdm(l_class_files):

            # Read GeoTIFF
            with rasterio.open(f'{url_dataset}/{image_class}/{file}') as src:
                # There is only one band
                try:
                    data = src.read(1)
                except:
                    print(f'An image in class {image_class} could not be read.')
                    continue
            # Normalise to [0, 255]
            data_norm = 255. * (data - np.min(data)) / (np.max(data) - np.min(data))

            # Save as .png
            im = Image.fromarray(data_norm)
            im = im.convert('L')
            image_root = file.split('.tif')[0]
            if not os.path.exists(f'{url_dataset_image}/{image_class}/{image_root}.png'):
                im.save(f'{url_dataset_image}/{image_class}_{image_root}.png')


def main(
        url_folder_tif: str,
        url_folder_png: str,
        frac_train: float = 0.8,
        frac_val: float = 0.05
        ) -> None:
    """
    Process TenGeoP-SARwv data from tif to png and split between
    train and test.

    Inputs:
    ------
        url_folder_tif (str): path to original TenGeoP-SARwv data
            (in GeoTiff format)

        url_folder_png (str): path to destination folder for .png
            files

        frac_train (float, optional): fraction of data to retain for
            training

        frac_train (float, optional): fraction of data to retain for
            validation
    """
    if frac_train + frac_val >= 1:
        raise ValueError("No data left for testing.")

    TenGeoP_SAR_to_ImageDataset(
        url_folder_tif,
        url_folder_png)

    # Split train/val/test
    os.makedirs(f'{url_folder_png}/train/', exist_ok=True)
    os.makedirs(f'{url_folder_png}/test/', exist_ok=True)
    os.makedirs(f'{url_folder_png}/val/', exist_ok=True)

    for image_class in ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']:

        l_files = [f for f in os.listdir(url_folder_png)
                   if f.startswith(f'{image_class}_')]

        # Move one-fifth to test and the rest to train
        N = int(len(l_files) * frac_train)
        M = int(len(l_files) * frac_val)
        for file in l_files[:N]:
            shutil.move(f'{url_folder_png}/{file}',
                        f'{url_folder_png}/train/{file}')
        for file in l_files[N:(N+M)]:
            shutil.move(f'{url_folder_png}/{file}',
                        f'{url_folder_png}/val/{file}')
        for file in l_files[(N+M):]:
            shutil.move(f'{url_folder_png}/{file}',
                        f'{url_folder_png}/test/{file}')


if __name__ == "__main__":

    main(
        "./data/GeoTIFF/",
        "./data/sar_tengeop/"
    )
