"""
Calculate the Frechet Inception Distance (FID) to evaluate image
generation models.

Code adapted from https://github.com/mseitzer/pytorch-fid/tree/master
licensed under the Apache License, Version 2.0.

"""
import os
import sys
from typing import Tuple

import h5py
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../../PDM_radar_generation_3/')
from improved_diffusion.CONSTANTS import DEVICE_ID


def get_frechet_distance(
        mu_real: np.ndarray,
        sigma_real: np.ndarray,
        mu_gen: np.ndarray,
        sigma_gen: np.ndarray,
        eps=1e-6
        ) -> float:
    """
    Calculate the Frechet Distance between two Gaussian distributions,
    characterised by their means and covariance matrices.

    The Frechet distance between two multivariate Gaussians
        X_1 ~ N(mu_1, C_1)
    and
        X_2 ~ N(mu_2, C_2)
    is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Inputs:
    ------
        mu_real (np.ndarray): array containing the mean of model activations
                              for real data samples.

        mu_gen (np.ndarray): array containing the mean of model activations
                             for generated data samples.

        sigma_real (np.ndarray): The covariance matrix over activations
                                 for real samples.

        sigma_gen (np.ndarray): The covariance matrix over activations
                                for generated samples.
    """
    mu_real = np.atleast_1d(mu_real)
    mu_gen = np.atleast_1d(mu_gen)

    sigma_real = np.atleast_2d(sigma_real)
    sigma_gen = np.atleast_2d(sigma_gen)

    if mu_real.shape != mu_gen.shape:
        raise ValueError('Train and test mean vectors have different lengths')
    if sigma_real.shape != sigma_gen.shape:
        raise ValueError('Train and test covariance have different dimensions')

    diff = mu_real - mu_gen

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               f'adding {eps} to diagonal of cov estimates')
        print(msg)
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma_real)
            + np.trace(sigma_gen) - 2 * tr_covmean)


def get_mu_sigma(
        dataloader: DataLoader,
        num_channels: int,
        model,
        num_samples: int
        ) -> Tuple[np.ndarray]:
    """
    Calculate the mean and covariance required as input to the FID.

    Inputs:
    ------
        dataloader (DataLoader): dataloader for the
            dataset of interest.

        num_channels (int): the number of channels for the input data

        model: Instance of classifier model

        num_samples (int): the number of real/generated samples
            to compute the scores

    Outputs:
    -------
        mu: The mean over samples of the activations of the pool_3 layer of
                the inception model.

        sigma: The covariance matrix of the activations of the pool_3 layer of
                the inception model.
    """
    # Obtain activation output from the model
    model.eval()

    pred_arr = []

    for batch, _ in tqdm(dataloader):

        batch = batch.to(DEVICE_ID)

        with torch.no_grad():

            # If only 1 channel, repeat to fit VGG16 input shape
            if num_channels == 1:
                batch = torch.repeat_interleave(batch, repeats=3, dim=1)

            # Apply cropped model
            pred = model(batch).squeeze().cpu().numpy()

        # In case there is more than 2 dimensions
        pred = np.sum(pred, axis=tuple([i for i in range(2, pred.ndim)]))

        # Update result
        pred_arr.append(pred)

    # Convert list to np.ndarray of shape (num_images, dims)
    # Each element of pred_arr is an array of shape (batch_size, dims)
    pred_arr = np.concatenate(pred_arr, axis=0)[:num_samples]

    # Average across examples
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)

    return mu, sigma


def get_fid(
        dataloader_real: DataLoader,
        dataloader_gen: DataLoader,
        num_channels: int,
        model: nn.Module,
        url_save_real_feature_fid: str,
        num_samples: int
        ) -> float:
    """
    Calculates the FID given two dataloaders sampling
    from the real and generated datasets.

    Input:
    ------
        dataloader_real (DataLoader): dataloader for the
            real samples

        dataloader_gen (DataLoader): dataloader for the
            generated samples

        num_channels (int): number of channel of the real
            and generated samples

        model (nn.Module): either a truncated classifier or
            an encoder that directly output a meaningful
            lower-dimensional representation of the data

        url_save_real_feature_fid (str): url to save the feature
            of the real dataset (no need to compute it several
            times)

        num_samples (int): the number of real/generated samples
            to compute the scores
    """

    if not url_save_real_feature_fid.endswith('.hdf5'):
        raise ValueError('If stored, the real data feature should be stored in an HDF5 file.')

    # Get mu, sigma for each image set
    if not os.path.exists(url_save_real_feature_fid):
        print('Extracting features from real images...')
        mu_real, sigma_real = get_mu_sigma(
            dataloader_real,
            num_channels,
            model,
            num_samples)

        # Save them into an hdf5 file
        with h5py.File(url_save_real_feature_fid, 'w') as f:
            f.create_dataset(
                'mu',
                shape=mu_real.shape,
                maxshape=mu_real.shape)
            f.create_dataset(
                'sigma',
                shape=sigma_real.shape,
                maxshape=sigma_real.shape)
            f['mu'][:] = mu_real
            f['sigma'][:] = sigma_real
    else:
        # Load the feature vector
        with h5py.File(url_save_real_feature_fid, 'r') as f:
            mu_real = f['mu'][:]
            sigma_real = f['sigma'][:]

    mu_gen, sigma_gen = get_mu_sigma(
        dataloader_gen,
        num_channels,
        model,
        num_samples)

    return get_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
