"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import tqdm

from improved_diffusion.CONSTANTS import DEVICE_ID
from improved_diffusion.script_util import create_model_and_diffusion
from utils.utils import read_model_metadata


def main_sample(
        model_path: str = None,
        clip_denoised: bool = True,
        max_clip_val: float = 1,
        diffusion_steps: int = 100,
        num_samples: int = 10000,
        sample_class: int = None,
        batch_size: int = 16,
        use_ddim: bool = False,
        url_save_path: str = 'images/',
        to_0_255: bool = False,
        plot: bool = True):
    '''
    Main sample function to generate images from a model file.

    Inputs:
    ------
        model_path (str): path to saved model

        clip_denoised (bool): clip the xstartpred of the model between
            [-max_clip_val, max_clip_val]. Note that it does not control
            directly the range of the samples. Indeed, it clips the prediction
            of the model from which are extracted (via q_posterior_mean_variance)
            the mean and variance parameters which control the distribution
            of the sample.

            This option is useful to constrain the model toward the right range and simply
            needs to be a loose upperbound of the dataset range.

        max_clip_val (float): a non-negative value that clips the
            images to [-mcv, mcv] when clip_denoised is True.

        diffusion_steps (int): number of diffusion steps to use

        num_samples (int): number of samples to generate

        sample_class (int): if not None, we only sample from this class.

        batch_size (int): batch size for sample generation

        use_ddim (bool): if True, uses Denoiser Diffusion Implicit Models (DDIM)
            to sample images from the model, otherwise uses p_sample_loop.

        url_save_path (str): folder where to save the generated images

        to_0_255 (bool): if True, then it bins the pixel data to integer
            between 0 and 255

        plot (bool): whether to plot the generated images. The plot option
            is not compatible with any number of channels in the data.

    References:
    ----------
    For DDIMs, see
        https://arxiv.org/pdf/2010.02502.pdf
    For original Probabilistic Diffusion Models, see
        https://arxiv.org/abs/2006.11239
    For improved Diffusion Probabilistic Models, see
        https://arxiv.org/abs/2102.09672
    '''
    # Check inputs
    if not os.path.exists(model_path):
        raise ValueError('The specified model does not exist!')
    if max_clip_val <= 0:
        raise ValueError('The max_clip_val is negative.')
    if not os.path.exists(url_save_path):
        os.makedirs(url_save_path, exist_ok=True)

    # Get model metadata from .json file
    url_model_folder = Path(model_path).parent
    url_metadata = [f for f in os.listdir(url_model_folder) if f.endswith('.json')]
    if len(url_metadata) == 1:
        model_dic = read_model_metadata(f'{url_model_folder}/{url_metadata[0]}')
    else:
        raise ValueError('There is no metadata file associated with this model.')

    # Create dictionary of arguments to pass for model initialisation
    args_dic = {
        'model_path': model_path,
        'clip_denoised': clip_denoised,
        'max_clip_val': max_clip_val,
        'num_samples': num_samples,
        'batch_size': batch_size,
        'use_ddim': use_ddim,
        'image_size': model_dic['image_size'],
        'num_input_channels': model_dic['num_input_channels'],
        'num_model_channels': model_dic['num_model_channels'],
        'num_res_blocks': model_dic['num_res_blocks'],
        'num_heads': model_dic['num_heads'],
        'num_heads_upsample': model_dic['num_heads_upsample'],
        'attention_resolutions': model_dic['attention_resolutions'],
        'dropout': model_dic['dropout'],
        'learn_sigma': model_dic['learn_sigma'],
        'sigma_small': model_dic['sigma_small'],
        'class_cond': model_dic['class_cond'],
        'num_class': model_dic['num_class'],
        'diffusion_steps': diffusion_steps,
        'noise_schedule': model_dic['noise_schedule'],
        'timestep_respacing': model_dic['timestep_respacing'],
        'loss_name': model_dic['loss_name'],
        'output_type': model_dic['output_type'],
        'rescale_timesteps': model_dic['rescale_timesteps'],
        'use_checkpoint': model_dic['use_checkpoint'],
        'use_scale_shift_norm': model_dic['use_scale_shift_norm']}

    # Create model
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args_dic)
    # Update model with saved weights
    params = torch.load(model_path, map_location="cpu")
    model.load_state_dict(params)
    model.to(DEVICE_ID)
    model.eval()

    if not args_dic['class_cond'] and sample_class is not None:
        print('Model is not class-conditional, "sample_class" argument will be ignored.')

    print("Sampling...")
    all_images = []
    all_labels = []
    num_batch = num_samples // batch_size + 1 * (num_samples % batch_size > 0)

    for batch_cnt in tqdm.tqdm(np.arange(num_batch)):

        # If model was trained on conditional, set classes to sample from.
        model_kwargs = {}
        if args_dic['class_cond']:
            if sample_class is None:
                classes = torch.randint(
                    low=0, high=args_dic['num_class'], size=(batch_size,), device=DEVICE_ID
                )
            else:
                classes = torch.tensor(
                    [sample_class] * batch_size, device=DEVICE_ID
                )
            model_kwargs["y"] = classes

        if model_dic['class_cond']:
            all_labels.append(classes.cpu().numpy())

        # Sampling function
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (batch_size, model_dic['num_input_channels'],
                model_dic['image_size'], model_dic['image_size']),
            clip_denoised=clip_denoised,
            max_clip_val=max_clip_val,
            model_kwargs=model_kwargs,
        )

        if to_0_255:
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        # From (B, C, H, H) to (B, H, H, C)
        sample = sample.permute(0, 2, 3, 1)
        # Optimize the memory footprint of sample
        sample = sample.contiguous()

        # all_images is a list of batches of images, each of shape (B, H, W, C)
        all_images.append(sample.cpu().numpy())

        if plot:
            for i in range(all_images[-1].shape[0]):
                plt.figure(figsize=(10, 10))
                ax = plt.gca()
                im = plt.imshow(all_images[-1][i])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.colorbar(im, cax=cax)
                if model_dic['class_cond']:
                    plt.title(f'class {all_labels[-1][i]}')
                plt.savefig(f'{url_save_path}/sample{batch_cnt*batch_size+i}.png')
                plt.close()

        print(f"Created {len(all_images) * batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]

    if model_dic['class_cond']:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: num_samples]

    # Returns the rank of the current process in the default group
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = f"{url_save_path}/samples_{shape_str}.npz"
    print(f"Saving to {out_path}")
    if model_dic['class_cond']:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    # Synchronizes all processes
    print("Sampling complete!")
