'''
Defines the ScheduleSampler class which allows
to sample diffusion timesteps to calculate model loss.
'''
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch


def create_named_schedule_sampler(
        name: str,
        diffusion):
    """
    Create a ScheduleSampler either a uniform sampler over the
    time steps of the diffusion process (UniformSampler) or a sampler
    accounting for the loss associated to each time steps
    (LossSecondMomentResampler).

    Inputs:
    -------
        name: the name of the sampler.
        diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalised, but must be positive.
        """

    def sample(
            self,
            batch_size: int,
            device: torch.device
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Importance-sampling of diffusion timesteps for a batch.

        Inputs:
        -------
            batch_size: the number of timesteps.
            device: the torch device to save to.

        Outputs:
        --------
        A tuple (timesteps, weights):
            - timesteps: a tensor of diffusion timestep indices.
            - weights: a tensor of weights to scale the resulting losses when averaging.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self) -> np.ndarray:
        return self._weights


class LossSecondMomentResampler(ScheduleSampler):

    def __init__(
            self,
            diffusion,
            history_per_term=10,
            uniform_prob=0.001):
        '''
        This provides a weighting schemes for each time step of the diffusion
        models according to past values of the losses.

        As long as there are not enough backward-forward iteration, the sampler
        first output weights of 1 for each time step of the diffusion model.
        Once warmed (see, self._warmed_up()).

        Inputs:
        -------
            diffusion (SpacedDiffusion): diffusion model defined in resample.py

            history_per_term (int): number of iterations or forward-backward for
                which the losses at each time steps of the diffusion model is
                stored.

            uniform_prob (float): it is the threshold value to ensure that
                even when the loss at a given time steps is zero, this time
                step still has the probability to be sampled (see,
                self.weights)

        '''
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int64)

    def weights(self) -> np.ndarray:
        '''
        If not warmed up, it outputs weights of 1 for every time steps of the diffusion,
        otherwise, the weight at time t is defined as

            w_t := 1/\sqrt{len(L_t)} |L_t|_2/C (1-p) + p/num_diff_steps,

        where L_t is the vector of loss associated to time step t and C is defined as

            C := sum_t^{num_diff_steps} 1/\sqrt{len(L_t)} |L_t|_2.
        '''
        if self._loss_history.shape != (self.diffusion.num_timesteps, self.history_per_term):
            raise ValueError('The shape of loss history tensor is mismatched.')
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.nanmean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(
            self,
            ts: torch.Tensor,
            losses: torch.Tensor):
        """

        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
