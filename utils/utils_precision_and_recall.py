"""
Slightly adapted from
https://github.com/kynkaat/improved-precision-and-recall-metric
https://github.com/youngjung/improved-precision-and-recall-metric-pytorch
"""
import os
import sys
from typing import Tuple, List

import h5py
import numpy as np
from time import time
import torch
import torch.nn as nn
import tqdm

sys.path.append('../../PDM_radar_generation_3/')
from improved_diffusion.CONSTANTS import DEVICE_ID


def get_features(
        dataloader: torch.utils.data.DataLoader,
        num_channels: int,
        num_samples: int,
        model
        ) -> torch.Tensor:
    '''
    Extracts classifier features.

    Inputs:
    ------
        dataloader (torch.utils.data.DataLoader): dataloader for the
            dataset of interest.

        num_channels (int): the number of channels of the input data

        num_samples (int): number of data samples to use

        model: Instance of classifier model (e.g., InceptionV3)
    '''
    model.eval()

    pred_tensor = []
    for batch, _ in tqdm.tqdm(dataloader):

        batch = batch.to(DEVICE_ID)

        with torch.no_grad():

            # If only 1 channel, repeat to fit VGG16 or InceptionV3 input shape
            # (3 channels)
            if num_channels == 1:
                batch = torch.repeat_interleave(batch, repeats=3, dim=1)

            # Apply cropped model
            pred = model(batch)

            if pred.ndim > 2:
                pred = pred.view(pred.shape[0], pred.shape[1], -1)
                pred = torch.sum(pred, dim=2)

        # Update result
        pred_tensor.append(pred)

    # Convert list to torch.Tensor of shape (num_images, dims)
    # Each element of pred_tensor is an array of shape (batch_size, dims)
    return torch.cat(pred_tensor, dim=0)[:num_samples]


def batch_pairwise_distances(
        U: torch.Tensor,
        V: torch.Tensor
        ) -> torch.Tensor:
    """
    Computes pairwise distances between two batches of feature vectors.
    """
    # Squared norms of each row in U and V.
    norm_u = torch.sum(U**2, 1)
    norm_v = torch.sum(V**2, 1)

    # norm_u as a column and norm_v as a row vectors.
    norm_u = torch.reshape(norm_u, (-1, 1))
    norm_v = torch.reshape(norm_v, (1, -1))

    # Pairwise squared Euclidean distances.
    return torch.maximum(norm_u - 2*torch.matmul(U, torch.t(V)) + norm_v,
                         torch.tensor([0.0]).to(DEVICE_ID))


class ManifoldEstimator():
    """
    Class to estimate the manifold of given feature vectors.
    """
    def __init__(
            self,
            features: torch.Tensor,
            row_batch_size: int = 25000,
            col_batch_size: int = 50000,
            nhood_sizes: list = [3],
            clamp_to_percentile: float = None,
            eps: float = 1e-5
            ):
        """
        Inputs:
        ------
            features (torch.Tensor): tensor of feature vectors from which
                to estimate the manifold.

            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter is trade-off between memory usage and performance).

            col_batch_size (int): Column batch size to compute pairwise
                distances.

            nhood_sizes (list): Number of neighbours used to estimate the
                manifold.

            clamp_to_percentile (float): Prune hyperspheres that have radius
                larger than the given percentile.

            eps (float): Small number for numerical stability.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances
        # to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_images],
                                  dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1-begin1, begin2:end2] =\
                    batch_pairwise_distances(
                        row_batch, col_batch).detach().cpu().numpy()

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(
                distance_batch[0:end1-begin1, :],
                seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(
            self,
            eval_features: torch.Tensor
            ) -> np.ndarray:
        """
        Evaluate if new feature vectors are in the manifold.
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros(
            [self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros(
            [num_eval_images, self.num_nhoods], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1-begin1, begin2:end2] =\
                    batch_pairwise_distances(feature_batch, ref_batch).detach().cpu().numpy()

            # From the minibatch of new feature vectors, determine if they are
            # in the estimated manifold. If a feature vector is inside a
            # hypersphere of some reference sample,
            # then the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from
            # distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(
                samples_in_manifold, axis=1).astype(np.int32)

        return batch_predictions


def knn_precision_recall_features(
        ref_features: torch.Tensor,
        eval_features: torch.Tensor,
        nhood_sizes: List[int] = [3],
        row_batch_size: int = 10000,
        col_batch_size: int = 50000
        ):
    """
    Calculates k-NN precision and recall for two sets of feature vectors.

    Inputs:
    ------
        ref_features (torch.Tensor): Feature vectors of reference images.

        eval_features (torch.Tensor): Feature vectors of generated images.

        nhood_sizes (list): Number of neighbors used to estimate the manifold.

        row_batch_size (int): Row batch size to compute pairwise distances
            (parameter to trade-off between memory usage and performance).

        col_batch_size (int): Column batch size to compute pairwise distances.

    Output:
    -------
        State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """

    num_images = ref_features.shape[0]

    # Initialise ManifoldEstimators.
    ref_manifold = ManifoldEstimator(
        ref_features,
        row_batch_size,
        col_batch_size,
        nhood_sizes)
    eval_manifold = ManifoldEstimator(
        eval_features,
        row_batch_size,
        col_batch_size,
        nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print(f'Evaluating k-NN precision and recall with {num_images} samples...')
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state = {'precision': precision.mean(axis=0)}

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0)

    print(f'Evaluated k-NN precision and recall in: {time() - start}s')

    return state


def compute_precision_and_recall(
        dataloader_real: torch.utils.data.DataLoader,
        dataloader_gen: torch.utils.data.DataLoader,
        num_channels: int,
        model: nn.Module,
        url_save_real_feature: str,
        nhood_sizes: List[int],
        num_samples: int
        ) -> Tuple[float, float]:
    """
    Inputs:
    ------
        dataloader_real (DataLoader): dataloader for the real data

        dataloader_gen (DataLoader): dataloader for the generated data

        num_channels (int): number of data channels

        model (nn.Module): neural network that outputs a feature
            representation of the data

        url_save_real_feature (str): url to save the feature
            of the real dataset (no need to compute it several
            times)

        nhood_sizes (List[int]): Nbr of k-nearest neighbors used
            to compute the hypersphere radius around each point

        num_samples (int): the number of real/generated samples
            to compute the scores
    """
    # Check inputs
    if not url_save_real_feature.endswith('.hdf5'):
        raise ValueError('The real data features should be stored in an HDF5 file.')

    it_start = time()

    # Calculate model features for real images
    if not os.path.exists(url_save_real_feature):
        print('Extracting features from real images...')
        features_real = get_features(
            dataloader_real,
            num_channels,
            num_samples,
            model)

        features_real = features_real.squeeze()  # remove useless dimensions

        # Save them into an HDF5 file
        with h5py.File(url_save_real_feature, 'w') as f:
            f.create_dataset(
                'features',
                shape=features_real.shape,
                maxshape=features_real.shape)
            f['features'][:] = features_real.detach().cpu().numpy()
    else:
        # Load the feature vector
        with h5py.File(url_save_real_feature, 'r') as f:
            features_real = f['features'][:]

        # Into a tensor
        features_real = torch.Tensor(features_real).to(DEVICE_ID)

    # Calculate model features for generated images
    print('Extracting features from generated images...')
    features_gen = get_features(
        dataloader_gen,
        num_channels,
        num_samples,
        model)

    features_gen = features_gen.squeeze()

    # Compute k-NN precision and recall
    state = knn_precision_recall_features(
        features_real,
        features_gen,
        nhood_sizes=nhood_sizes)

    # Store results
    precision = state['precision'][0]
    recall = state['recall'][0]

    # Print results
    print(f'Precision: {np.round(state["precision"][0], 3)}')
    print(f'Recall: {state["recall"][0]}')
    print(f'Iteration time: {time() - it_start}s\n')

    return precision, recall
