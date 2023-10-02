import os
from pathlib import Path
from typing import List

import blobfile as bf
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from improved_diffusion.CONSTANTS import DEVICE_ID
from improved_diffusion.script_util import create_sr_model_and_diffusion
from utils.utils import read_model_metadata


def load_generated_data_for_worker(
        url_lr_data: str = None,
        batch_size: int = 32,
        class_cond: bool = False
        ) -> DataLoader:
    """
    Returns a data loader on the low-resolution generated
    samples (base_samples)

    Inputs:
    ------
        url_lr_data (str): .npz file with low-resolution data samples.
            Defaults to None.
        batch_size (int): Batch size for the data loader.
            Defaults to 32.
        class_cond (bool): whether to return the class labels.
            Defaults to False.
    """
    # Read low-resolution samples
    with bf.BlobFile(url_lr_data, "rb") as f:
        obj = np.load(f)
        image_lr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    buffer_lr = []
    label_buffer = []
    while True:
        for i in np.arange(image_lr.shape[0]):
            buffer_lr.append(image_lr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer_lr) == batch_size:
                batch_lr = torch.from_numpy(np.stack(buffer_lr)).float()
                # If RGB image, rescale to [-1, 1]
                if batch_lr.shape[1] == 3:
                    batch_lr = batch_lr / 127.5 - 1.0
                res = dict(low_res=batch_lr)
                if class_cond:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer_lr, label_buffer = [], []


def main_sample_superres(
        model_path: str = None,
        url_lr_data: str = None,
        clip_denoised: bool = True,
        max_clip_val: float = 1,
        diffusion_steps: int = 100,
        batch_size: int = 16,
        use_ddim: bool = False,
        plot: bool = False,
        url_save_path: str = 'images/',
        to_0_255: bool = False
        ) -> None:
    '''
    Inputs:
    ------
        model_path (str): path to saved superresolution model

        url_lr_data (str): path to low-resolution data .npz file

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

        batch_size (int): batch size for sample generation

        use_ddim (bool): if True, uses Denoiser Diffusion Implicit Models (DDIM)
            to sample images from the model, otherwise uses p_sample_loop.

        url_save_path (str): folder where to save the generated images

        to_0_255 (bool): if True, then it bins the pixel data to integer
            between 0 and 255
    '''
    if not os.path.exists(model_path):
        raise ValueError('The specified model does not exist!')

    if max_clip_val <= 0:
        raise ValueError('The max_clip_val is negative.')

    if not os.path.exists(url_save_path):
        os.makedirs(url_save_path, exist_ok=True)

    if not os.path.exists(url_lr_data) or not url_lr_data.endswith('.npz'):
        raise ValueError("url_lr_data is incorrect.")

    # Get model metadata from .json file
    url_model_folder = Path(model_path).parent
    url_metadata = [f for f in os.listdir(url_model_folder)
                    if f.endswith('.json')]
    if len(url_metadata) == 1:
        model_dic = read_model_metadata(f'{url_model_folder}/{url_metadata[0]}')
    else:
        raise ValueError('There is no metadata file associated with this model.')

    # Read number of low-resolution samples
    with bf.BlobFile(url_lr_data, "rb") as f:
        obj = np.load(f)
        num_samples = obj["arr_0"].shape[0]
        del obj

    # Create dictionary of arguments to pass for model initialisation
    args_dic = {
        'model_path': model_path,
        'clip_denoised': clip_denoised,
        'max_clip_val': max_clip_val,
        'num_samples': num_samples,
        'batch_size': batch_size,
        'use_ddim': use_ddim,
        'image_size': model_dic['image_size'],
        'image_size_lr': model_dic['image_size_lr'],
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
        'use_scale_shift_norm': model_dic['use_scale_shift_norm'],
        'crop': model_dic['crop']}

    if not args_dic['class_cond']:
        raise ValueError('This script only works for conditional models!')

    print("Creating dataloader")
    loader_lr = load_generated_data_for_worker(
        url_lr_data,
        batch_size,
        args_dic['class_cond'])

    # Create model
    print("Creating model and diffusion...")
    model, diffusion = create_sr_model_and_diffusion(args_dic)
    # Update model with saved weights
    params = torch.load(model_path, map_location="cpu")
    model.load_state_dict(params)
    model.to(DEVICE_ID)
    model.eval()

    print("Sampling...")
    all_images = []
    all_labels = []
    num_batch = num_samples // batch_size + 1 * (num_samples % batch_size > 0)

    for batch_cnt in tqdm.tqdm(np.arange(num_batch)):

        # Obtain a batch of downsampled images
        model_kwargs = next(loader_lr)
        if model_dic['class_cond']:
            all_labels.append(model_kwargs['y'])
        model_kwargs['low_res'] = torch.moveaxis(model_kwargs['low_res'], -1, 1)
        batch_lr = model_kwargs['low_res']
        model_kwargs = {k: v.to(DEVICE_ID) for k, v in model_kwargs.items()}
        sample_fn = (
            diffusion.p_sample_loop
            if not use_ddim
            else diffusion.ddim_sample_loop
        )
        # Get high-resolution sample
        sample = sample_fn(
            model,
            (batch_size, model_dic['num_input_channels'],
             model_dic['image_size'], model_dic['image_size']),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )

        if to_0_255:
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        # From (B, C, H, H) to (B, H, H, C)
        sample = sample.permute(0, 2, 3, 1)
        batch_lr = batch_lr.permute(0, 2, 3, 1)

        # Optimize the memory footprint of sample
        sample = sample.contiguous()

        # Move to CPU
        sample = sample.cpu()

        # Remove sample bias
        mean_lr = torch.mean(batch_lr, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mean_hr = torch.mean(sample, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sample = sample - mean_hr + mean_lr

        # all_images is a list of batches of images of shape (B, H, W, C)
        all_images.append(sample.numpy())

        if plot:
            for i in range(all_images[-1].shape[0]):
                m, M = torch.min(batch_lr[i]).item(), torch.max(batch_lr[i]).item()
                fig, axs = plt.subplots(1, 2, figsize=(20, 10))
                im = axs[0].imshow(batch_lr[i], vmin=m, vmax=M)
                axs[0].set_title('Original LR image')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(all_images[-1][i], vmin=m, vmax=M)
                axs[1].set_title('Generated HR image')
                fig.colorbar(im, ax=axs[1])
                plt.savefig(f'{url_save_path}/sample{batch_cnt*batch_size+i}.png')
                plt.close()

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
