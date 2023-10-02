from typing import Union, List

import numpy as np
import torch

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(
        num_timesteps: int,
        section_counts: Union[List[int], str]) -> set:
    """
    Creates a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there are 300 timesteps and the section counts are [10, 15, 20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    Inputs:
    ------
        num_timesteps (int): the number of diffusion steps in the original
            process to divide up.

        section_counts (Union[List[int], str])): either a list of numbers,
            or a string containing comma-separated numbers, indicating
            the step count per section. As a special case, use "ddimN"
            where N is a number of steps to use the striding from the
            DDIM paper.

    Output:
    ------
        a set of diffusion steps from the original process to use.
    """

    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        # Case where the string is of format, e.g., "2,34,5"
        section_counts = [int(x) for x in section_counts.split(",")]

    # size_per is the section lenght
    size_per, extra = divmod(num_timesteps, len(section_counts))
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"a section of {size} steps is to small to be splitted into {section_count} steps"
            )
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size

    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip time steps from a base equally time spaced 
    gaussian diffusion process, where the resampling of time steps is operated
    by the function 'space_timesteps'

    Inputs:
    ------
        use_timesteps: a collection (sequence or set) of timesteps from the
            original diffusion process to retain.

        kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(
            self,
            use_timesteps: Union[list, set],
            **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
            self,
            model,
            *args,
            **kwargs):
        # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
            self,
            model,
            *args,
            **kwargs):
        '''
        model (_WrappedModel)
        '''
        # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:

    def __init__(
            self,
            model,
            timestep_map: List[int],
            rescale_timesteps: bool,
            original_num_steps: int):
        '''

        Inputs:
        -------
            model

            timestep_map (List[int]): I do not really understand.

            rescale_timesteps (bool):

            original_num_steps (int): the original number of
                time steps of the diffusion process.

        '''
        self.model = model
        self.timestep_map = timestep_map
        # Whether to rescale diffusion timesteps to 1-1000
        self.rescale_timesteps = rescale_timesteps
        # Original (non-rescaled) diffusion timesteps
        self.original_num_steps = original_num_steps

    def __call__(
            self,
            x: torch.Tensor,
            ts: torch.Tensor, **kwargs):
        '''
        '''
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        if torch.max(map_tensor) > len(map_tensor):
            raise ValueError('There is an incompatiblity between the timestep map and tensor of time steps.')
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
