'''
Evaluate model performance by calculating its log-likelihood
on a set of images.
'''
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from improved_diffusion.CONSTANTS import DEVICE_ID
from improved_diffusion.respace import SpacedDiffusion
from improved_diffusion.script_util import create_model_and_diffusion
from improved_diffusion.unet import UNetModel
from sample import read_model_metadata
from utils.utils import create_dataloader


def run_evaluation_loop(
        model: UNetModel,
        diffusion: SpacedDiffusion,
        loader_data: DataLoader,
        clip_denoised: bool,
        subset_timesteps: int
        ) -> dict:
    """
    Calculate model loss terms given a dataloader.
    """
    # Initialise lists of metrics
    all_bpd, all_xstart_mse = [], []

    for (batch, model_kwargs) in tqdm.tqdm(loader_data):

        batch = batch.to(DEVICE_ID)
        model_kwargs = {k: v.to(DEVICE_ID) for k, v in model_kwargs.items()}

        # Calculate loss metrics on this batch
        # We are interested in 'total_bpd' and 'xstart_mse'
        minibatch_metrics = diffusion.calc_bpd_loop(
            model,
            batch,
            clip_denoised=clip_denoised,
            subset_timesteps=subset_timesteps,
            model_kwargs=model_kwargs
        )

        # Average metrics across the batch
        # VB
        total_bpd = minibatch_metrics["total_bpd"].mean()
        all_bpd.append(total_bpd.item())
        # xstart_MSE
        xstart_mse = minibatch_metrics["xstart_mse"].mean()
        all_xstart_mse.append(xstart_mse.item())

    return {
        'vb_loss': np.mean(np.array(all_bpd)),
        'mse_xstart_loss': np.mean(np.array(all_xstart_mse))
    }


def calc_model_loss(
        model_path: str,
        url_dataset: str,
        data_type: str = "image",
        list_keys_hdf5_original: List[str] = [],
        key_data: str = None,
        key_other: str = None,
        diffusion_steps: int = 100,
        batch_size: int = 32,
        clip_denoised: bool = True,
        subset_timesteps: int = None
        ) -> dict:
    '''
    Inputs:
    ------
        model_path (str): path to .pt model file

        url_dataset (str): path to dataset on which to evaluate the model

        data_type (str):

        diffusion_steps (int):

        batch_size (int):

        clip_denoise (bool):

        subset_timesteps (int): if not None, calculates the loss on every
            'subset_timesteps' diffusion time steps.
    '''
    # Check inputs
    if not os.path.exists(model_path):
        raise ValueError('The specified model does not exist!')

    # Get model metadata from .json file
    url_model_folder = Path(model_path).parent
    url_metadata = [f for f in os.listdir(url_model_folder) if f.endswith('.json')]
    if len(url_metadata) == 1:
        model_dic = read_model_metadata(f'{url_model_folder}/{url_metadata[0]}')
    else:
        raise ValueError('There is no metadata file associated with this model!')

    # Create dictionary of arguments to pass for model initialisation
    args_dic = {
        'model_path': model_path,
        'clip_denoised': clip_denoised,
        'batch_size': batch_size,
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
        'use_scale_shift_norm': model_dic['use_scale_shift_norm'],
        'crop': model_dic['crop']}

    # Create dataloader from url_dataset
    loader_data = create_dataloader(
        data_dir=url_dataset,
        data_type=data_type,
        num_channels=model_dic['num_input_channels'],
        image_size=model_dic['image_size'],
        batch_size=batch_size,
        class_cond=model_dic['class_cond'],
        num_class=model_dic['num_class'],
        list_keys_hdf5_original=list_keys_hdf5_original,
        key_data=key_data,
        key_other=key_other,
        crop=model_dic['crop'])

    # Create model
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args_dic)
    # Update model with saved weights
    params = torch.load(model_path, map_location="cpu")
    model.load_state_dict(params)
    model.to(DEVICE_ID)
    # Set to evaluation mode
    model.eval()

    return run_evaluation_loop(
        model,
        diffusion,
        loader_data,
        args_dic['clip_denoised'],
        subset_timesteps)


if __name__ == "__main__":

    vb_loss_all_steps = np.zeros(21)
    mse_loss_all_steps = np.zeros(21)
    vb_loss_sub_steps = np.zeros(21)
    mse_loss_sub_steps = np.zeros(21)

    for j in range(21):

        # No subsetting
        res = calc_model_loss(
            model_path=f'./models_data/mnist_cond_try/model_epoch={j}.pt',
            url_dataset='./data/mnist_val/',
            data_type='image',
            diffusion_steps=100,
            batch_size=32,
            clip_denoised=True,
            subset_timesteps=None
            )
        vb_loss_all_steps[j] = res['vb_loss']
        mse_loss_all_steps[j] = res['mse_xstart_loss']

        # With subsetting
        res = calc_model_loss(
            model_path=f'./models_data/mnist_cond_try/model_epoch={j}.pt',
            url_dataset='./data/mnist_val/',
            data_type='image',
            diffusion_steps=100,
            batch_size=32,
            clip_denoised=True,
            subset_timesteps=15
            )
        vb_loss_sub_steps[j] = res['vb_loss']
        mse_loss_sub_steps[j] = res['mse_xstart_loss']

    vb_loss_all_steps = np.array([2.22722752, 1.90901954, 1.74354593,
                                  1.67318312, 1.67589817, 1.71404836,
                                  1.70361207, 1.64420795, 1.71087053,
                                  1.68999469, 1.66051892, 1.67765769,
                                  1.64319855, 1.69199453, 1.67175644,
                                  1.69851503, 1.69167430, 1.69029692,
                                  1.66496420, 1.65350213, 1.63705818])
    vb_loss_sub_steps = np.array([1.98941159, 1.74569662, 1.60022275,
                                  1.54136888, 1.55760296, 1.59754450,
                                  1.59204621, 1.52541065, 1.59341836,
                                  1.57673632, 1.55018573, 1.56672116,
                                  1.53241180, 1.57986455, 1.56403541,
                                  1.59199979, 1.58652473, 1.58486247,
                                  1.55526416, 1.55146280, 1.53004509])
