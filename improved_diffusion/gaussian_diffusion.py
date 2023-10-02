"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import enum
import math
import os
from tqdm.auto import tqdm
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(
        schedule_name: str,
        num_diffusion_timesteps: int
        ):
    """
    Get a pre-defined beta schedule based on schedule_name. Used to define
    the scheduler (beta_t) in the diffusion model.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.

    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(
        num_diffusion_timesteps: int,
        alpha_bar,
        max_beta: float = 0.999):
    """
    [Only used in "get_named_beta_schedule" when schedule_name == "cosine"]

    Creates a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    Inputs:
    ------
        num_diffusion_timesteps (int): the number of betas to produce.

        alpha_bar (function): a lambda that takes an argument t from 0 to 1 and
            produces the cumulative product of (1-beta) up to that
            part of the diffusion process.

        max_beta (float): the maximum beta to use; use values lower than 1 to
            prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    '''
    The loss functions measure the likelihood of the generated samples
    at the end of the reverse process given the original data distribution.
    '''
    # Use raw MSE loss (and KL when learning variances)
    MSE = enum.auto()
    # Use raw MSE loss (with RESCALED_KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )
    # Use the variational lower-bound
    KL = enum.auto()
    # Like KL, but rescale to estimate the full VLB
    RESCALED_KL = enum.auto()

    def is_vb(self):
        '''
        Check if self is using a variational lower bound
        '''
        return self in [LossType.KL, LossType.RESCALED_KL]


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    """

    def __init__(
            self,
            *,
            betas: np.ndarray,
            model_mean_type: enum.Enum,
            model_var_type: enum.Enum,
            loss_type: enum.Enum,
            rescale_timesteps: bool = False,
            ):
        '''
        Inputs:
        ------
            betas (np.ndarray): a 1-D numpy array of beta coefficients for each diffusion
                timestep, starting at T and going to 1. Obtained from "noise_schedule".

            model_mean_type (enum.Enum): a ModelMeanType determining what the model outputs,
                i.e., either x_{t-1}, x_0 or epsilon. See definition of epsilon in
                "Improved Denoised Diffusion Models".

            model_var_type (enum.Enum): a ModelVarType determining how variance is handled.

            loss_type (enum.Enum): a LossType determining the loss function to use.

            rescale_timesteps (bool): if True, pass floating point timesteps into the
                model so that they are always scaled like in the original paper (0 to 1000).
        '''
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        if len(betas.shape) != 1:
            raise ValueError("Betas must be 1-D!")
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError('The beta coefficients should be in (0, 1]')

        self.num_timesteps = int(betas.shape[0])

        # Define alpha coefficients (alpha=1-beta)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor
            ) -> Tuple[torch.Tensor]:
        """
        Get the moments of distribution q(x_t | x_0), where x_0 is the
        original image (x_start). It corresponds to the noising process where

            q(x_t|x_0) = N(x_t; \sqrt(\alpha_bar_t)x_0, (1-\alpha_bar_t) I),

        as defined in "Improved Denoising Diffusion Probabilistic Model".

        Inputs:
        ------
            x_start (torch.Tensor): the [N x C x ...] tensor of noiseless inputs

            t (torch.Tensor): the number of diffusion steps (minus 1).
                Here, 0 means one step. Of shape (1, )

        Output:
        ------
            A tuple (mean, variance, log_variance), all of x_start's shape
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor = None
            ) -> torch.Tensor:
        """
        Diffuses the data for a given number of diffusion steps.\n
        In other words, sample from q(x_t | x_0), see formula in
        q_mean_variance.

        Inputs:
        ------
            x_start (torch.Tensor): the initial data batch.

            t (torch.Tensor): the number of diffusion steps (minus 1).
                Here, 0 means one step.

            noise (torch.Tensor): if specified, the split-out normal noise.

        Output:
        ------
            A noisy version of x_start.
        """
        if noise is None:
            # Standard normal distribution
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError('THe noise and x_0 should be of the same shape')
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(
            self,
            x_start: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Computes the mean, variance and log-variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0),

        see equation (9)-(11) in
            'Improved Denoising Diffusion Probabilistic Model'.

        """
        if x_start.shape != x_t.shape:
            raise ValueError('x0 and xt should be of same shape.')

        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    ###
    # AUXILIARY FUNCTIONS FOR SAMPLING AND LOSS CALCULATION
    ###

    def p_mean_variance(
            self,
            model,
            x: torch.Tensor,
            t: torch.Tensor,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None
            ) -> dict:
        """
        Applies the model to get the moments of p(x_{t-1} | x_t), as well as
        a prediction of x_0 for which there are different approaches
        according to ModelMeanType, i.e.,
            - For ModelMeanType.START_X, the model is directly trained to
                output x0.
            - For ModelMeanType.EPSILON, the model is trained to predict
                epsilon and x0 is obtained via equation (12) in
                "Improved Denoising Diffusion Probabilistic Models".
            - For ModelMeanType.PREVIOUS_X, the model is trained to output
                the time t-1 and we obtain x_0 simply with equation (3) in
                "Improved Denoising Diffusion Probabilistic Models". It is
                never used.

        Inputs:
        ------
            model (_WrappedModel): the model, which takes a signal and
                a batch of timesteps as input.

            x (torch.Tensor): the [N x C x ...] tensor at time t.

            t (torch.Tensor): a 1-D Tensor of randomly sampled diffusion timesteps.

            clip_denoised (bool): if True, clip the denoised signal into
                [-max_clip_val, max_clip_val].

            denoised_fn: if not None, a function which applies to the
                x_start prediction before it is used to sample. Applies before
                clip_denoised.

            model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.

        Output:
        ------
            a dict with the following keys:\n
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        if t.shape != (B,):
            raise ValueError('The 1-D time steps should have same number of time steps as x_t')

        # x = torch.normal(0, 1, (16, 1, 32, 32)).to(device)
        # t = torch.arange(23, 39).to(device)
        # res = model(x, t)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # Get the variance and log variance according to whether learn_sigma is True or False
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # The variance is learned along with the mean
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            # The variance is fixed.
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(
                x: torch.Tensor) -> torch.Tensor:
            if denoised_fn is not None:
                x = denoised_fn(x)
            return x.clamp(-max_clip_val, max_clip_val) if clip_denoised else x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:  # self.model_mean_type == ModelMeanType.EPSILON
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            # The following corresponds to equation (10)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            eps: torch.Tensor
            ):
        '''
        In the case of ModelMeanType.EPSILON, the prediction of x0
        corresponds to equation (12) in 
            "Improved Denoising Diffusion Probabilistic Models".
        '''
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            xprev: torch.Tensor
            ):
        '''
        In the case ModelMeanType.PREVIOUS_X, the model is trained to output
        the time t-1 so that we have access to xprev.
        We obtain x_0 simply with equation (3) in
            "Improved Denoising Diffusion Probabilistic Models".
        '''
        assert x_t.shape == xprev.shape
        # (xprev - coef2*x_t) / coef1
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            pred_xstart: torch.Tensor
            ):
        '''
        Equation (12) in "Improved Denoising Diffusion Probabilistic Models".
        '''
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(
            self,
            t: torch.Tensor):
        '''
        If rescale_timesteps, it rescale the time steps between 
        (0, 1000).
        '''
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    ###
    # SAMPLING FUNCTIONS as in DDPMs (lengthy)
    ###

    def p_sample(
            self,
            model,
            x: torch.Tensor,
            t: torch.Tensor,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None
            ):
        """
        Sample x_{t-1} from the model at the given timestep.

        Inputs:
        ------
            model: the model to sample from.

            x (torch.Tensor): the current tensor at x_{t-1}.

            t (torch.Tensor): the value of the current diffusion timestep,
                starting at 0 for the first diffusion step.

            clip_denoised (bool): if True, clip the x_start prediction to
                [-max_clip_val, max_clip_val].

            denoised_fn: if not None, a function which applies to the
                x_start prediction before it is used to sample.

            model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.

        Output:
        ------
            a dict containing the following keys:\n
                - 'sample': a random sample from the model.\n
                - 'pred_xstart': a prediction of x_0.\n
        """

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            max_clip_val=max_clip_val,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape: tuple,
            noise: torch.Tensor = None,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress: bool = False,
            show: bool = False,
            save: list = None
            ) -> torch.Tensor:
        """
        Generate samples from the model by looping over p_sample.

        Inputs:
        ------
            model: the model module.

            shape (tuple): the shape of the samples, (N, C, H, W).
                -> (batch_size, num_channels, H, W).

            noise (torch.Tensor): if specified, the noise from the encoder to sample.
                Should be of the same shape as "shape".

            clip_denoised: if True, clip x_start predictions to [-1, 1].

            denoised_fn: if not None, a function which applies to the
                x_start prediction before it is used to sample.

            model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.

            device: if specified, the device to create the samples on.
                If not specified, use a model parameter's device.

            progress (bool): if True, show a tqdm progress bar.

            save (list): if not None, then saves the intermediate samples
                at specified time steps.
        Output:
        ------
            a non-differentiable batch of samples.
        """
        final = None

        if show:
            folder_img = f'images/every_iteration_with_clip_{max_clip_val}/'
            if not os.path.exists(folder_img):
                os.mkdir(folder_img)

        for idx, out in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            max_clip_val=max_clip_val,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        )):
            # Dictionary that contains 'sample' and 'pred_xstart'
            final = out
            if show:
                plt.imshow(final['pred_xstart'].detach().cpu().numpy()[0, 0, :, :])
                plt.colorbar()
                plt.savefig(f'{folder_img}/pred_xstart_{idx}.png')
                plt.close()
            if save:
                if idx in save:
                    x = final['pred_xstart'].detach().cpu().numpy()[0, 0, :, :]
                    with h5py.File(f'./images/sample_reverse_{idx}.hdf5', 'w') as file:
                        file.create_dataset('sample', x.shape, maxshape=x.shape)
                        file['sample'][:] = x

        return final["sample"]

    def p_sample_loop_progressive(
            self,
            model,
            shape: tuple,
            noise: torch.Tensor = None,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress: bool = False,
            ):  # -> Generator
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None\
            else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    max_clip_val=max_clip_val,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    ##
    # DENOISING DIFFUSION IMPLICIT MODELS (DDIMs) (Fast)
    ##

    def ddim_sample(
            self,
            model,
            x: torch.Tensor,
            t: torch.Tensor,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None,
            eta: float = 0.0,
            ):
        """
        Sample x_{t-1} from the model using DDIM, see paper
            "Denoising Diffusion Implicit models".

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            max_clip_val=max_clip_val,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
            )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x: torch.Tensor,
            t: int,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None,
            eta: float = 0.0,
            ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            max_clip_val=max_clip_val,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
            )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape: tuple,
            noise: torch.Tensor = None,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress: bool = False,
            eta: float = 0.0,
            ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            max_clip_val=max_clip_val,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            model,
            shape: tuple,
            noise: torch.Tensor = None,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress: bool = False,
            eta: float = 0.0,
            ):
        """
        Use (Denoising Diffusion Implicit Models) DDIM to sample from the
        model and yield intermediate samples from each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None\
            else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    max_clip_val=max_clip_val,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    ###
    # TRAINING LOSS FUNCTIONS
    ###

    def _vb_terms_bpd(
            self,
            model,
            x_start: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            model_kwargs=None
            ):
        """
        Calculate the loss function L_{t-1}.

        If the diffusion time step is 0, return the decoder NLL,
        otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        The resulting units are bits per dims (bpd) to allows for comparison
        to other papers.

        Inputs:
        ------
            model ():

            x_start (torch.Tensor): tensor of shape (batch_size, num_channels, H, W)
                (or with batch_size -> microbatch)

            x_t (torch.Tensor): tensor of images with noise (diffusion steps t),
                of same shape as x_start

            t (torch.Tensor): tensor of diffusion time steps
                (of shape [batch_size] or [microbatch])

            model_kwargs (dict): if not None, a dictionary with class labels
                for conditional training

        Output:
        ------
            A dict with the following keys:\n
                - 'output': a shape [N] tensor of Negative Log-Likelihoods or KLs.\n
                - 'pred_xstart': the x_0 predictions.
        """
        # Computes mean and variance of q(x_{t-1}|x_t, x_start)
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start,
            x_t=x_t,
            t=t
        )
        # Applies neural network to predict p(x_{t-1}|x_t)
        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            max_clip_val=max_clip_val,
            model_kwargs=model_kwargs
        )
        # L_{t-1}: KL divergence between q(x_{t-1}|x_t, x_start) and p(x_{t-1}|x_t)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        # Average for each batch element
        kl = mean_flat(kl) / np.log(2.0)

        # Negative log-likelihood
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)

        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
            self,
            model,
            x_start: torch.Tensor,
            t: torch.Tensor,
            model_kwargs: dict = None,
            noise: torch.Tensor = None
            ):
        """
        Computes training losses for a single timestep.

        Inputs:
        ------
            model (_WrappedModel): the model to evaluate the loss on.

            x_start (torch.Tensor): the [N x C x ...] tensor of inputs.

            t (torch.Tensor): a batch of timestep indices.

            model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model which correspond to class labels.

            noise: if specified, the specific Gaussian noise to try to remove.

        Output:
        ------
            a dict with the key "loss" containing a tensor of shape [N].
                Some mean or variance settings may also have other keys.
        """

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        # Adds noise to x_start for t diffusion time steps
        # according to forward (noising) process q.
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        # Compute loss
        # If loss is KL divergence
        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps

        # If loss is MSE
        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:

            # Apply model to x_t
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            # Learn the variance with the VB
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # Define the target based on ModelMeanType
            target = {
                # PREVIOUS_X: predict x_{t-1}
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                # START_X: predict x_0
                ModelMeanType.START_X: x_start,
                # EPSILON: predict the noise
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["loss"] = terms["mse"] + terms["vb"] if "vb" in terms\
                else terms["mse"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start: torch.Tensor) -> torch.Tensor:
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        Input:
        ------
            x_start (torch.Tensor): the [N x C x ...] tensor of inputs.

        Output:
        ------
            a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(
            self,
            model,
            x_start: torch.Tensor,
            clip_denoised: bool = True,
            max_clip_val: float = 1,
            subset_timesteps: int = None,
            model_kwargs=None
            ):
        """
        Computes the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        Inputs:
        ------
            model: the model to evaluate the loss on.

            x_start (torch.Tensor): the [N x C x ...] tensor of inputs.
                shape (batch_size, num_channels, ...)

            clip_denoised (bool): if True, clip denoised samples in
                [-max_clip_val, max_clip_val].

            subset_timesteps (int): if not None, keeps every subset_timesteps
                timesteps to calculate the loss.

            model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning
                (with data labels).

        Output:
        ------
            a dict containing the following keys:\n
                - 'total_bpd': the total variational lower-bound, per batch element.\n
                - 'prior_bpd': the prior term in the lower-bound.\n
                - 'vb': an [N x T] tensor of terms in the lower-bound.\n
                - 'xstart_mse': an [N x T] tensor of x_0 MSEs for each timestep.\n
                - 'mse': an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []

        sampled_timesteps = list(range(self.num_timesteps))[::-1]
        if subset_timesteps is not None:
            # Select last 'subset_timesteps' diffusion time steps
            sampled_timesteps = sampled_timesteps[-subset_timesteps:]

        # Loop on all diffusion time steps
        for t in sampled_timesteps:

            # Tensor of timesteps, all equal to t
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)

            # Sample from the forward noising process
            # after t diffusion steps: q(x_t | x_0)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            # Calculate VLB term at t (= L_t)
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    max_clip_val=max_clip_val,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            # MSE loss
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            # MSE loss on noise variance
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        # to Tensors of shape (batch_size, num_diffusion_steps)
        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        # Add L0 term
        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd

        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(
        arr: np.ndarray,
        timesteps: torch.Tensor,
        broadcast_shape: tuple
        ) -> torch.Tensor:
    """
    Extracts values from a 1-D numpy array for a batch of indices.

    Inputs:
    ------
        arr (np.ndarray): the 1-D numpy array.

        timesteps (torch.Tensor): a 1-D tensor of indices into the array to 
            extract of lenght the (micro)batch size.

        broadcast_shape: a larger shape of K dimensions with the batch
            dimension equal to the length of timesteps.

    Output:
    ------
        a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # Check the inputs
    if arr.ndim != 1 or timesteps.ndim != 1:
        raise ValueError('arr should be a 1-D array')
    if timesteps.shape[0] != broadcast_shape[0]:
        raise ValueError('The broadcast shape should have the same shape as the idx tensor.')
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
