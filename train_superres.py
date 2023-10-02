"""
Train a diffusion model to generate high-resolution (e.g., 256x256)
SAR images conditioned on low-resolution (e.g., 128x128) images.
"""
import os
from pathlib import Path

import json
from torch.utils.tensorboard import SummaryWriter

from improved_diffusion.CONSTANTS import DEVICE_ID
from improved_diffusion.datasets_image_superres import load_data_superres
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import create_sr_model_and_diffusion
from improved_diffusion.train_util import TrainLoop
from utils.utils import check_inputs, load_args_dic


def main_superres(
        data_train_dir: str = None,
        data_val_dir: str = None,
        num_epochs: int = 100,
        schedule_sampler: str = "uniform",
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        batch_size: int = 4,
        microbatch: int = -1,
        ema_rate="0.9999",
        save_interval: int = None,
        resume_checkpoint: str = None,
        use_fp16: bool = False,
        fp16_scale_growth: float = 1e-3,
        image_size: int = 256,
        image_size_lr: int = 128,
        num_input_channels: int = 3,
        num_model_channels: int = 128,
        num_res_blocks: int = 2,
        num_heads: int = 4,
        num_heads_upsample: int = -1,
        attention_resolutions: str = "16,8",
        dropout: float = 0.0,
        learn_sigma: bool = False,
        sigma_small: bool = False,
        class_cond: bool = False,
        num_class: int = None,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        timestep_respacing: str = "",
        loss_name: str = "mse",
        output_type: str = "epsilon",
        rescale_timesteps: bool = True,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
        url_folder_experiment: str = './models_data/first_try_superres/',
        compute_val: bool = False,
        subset_timesteps: int = None,
        crop: bool = False
        ) -> None:
    '''
    Inputs:
    ------
        data_train_dir (str): directory where training data is located

        data_val_dir (str): directory where validation data is located

        num_epochs (int): number of training epochs

        schedule_sampler (str): an object of class ScheduleSampler
            defined in resample.py, used to sample diffusion time steps
            when calculating the loss. It is either "uniform" or
            "loss-second-moment". If "uniform", timesteps are sampled
            uniformly. If "loss-second-moment", timesteps are sampled
            with importance sampling according to weights that correspond
            to past values of model losses.

        lr (float): model learning rate

        weight_decay (float): weight decay coefficient in AdamW method

        lr_anneal_steps (int): step decay for learning rate annealing

        batch_size (int): training batch size

        microbatch (int): -1 disables microbatches. If > 0 then equals
            the microbatch size. Must be < batch_size.

        ema_rate (float/str): either a float or a comma-separated list of
                rates to compute smoothed version(s) of model using an
                Exponential Moving Average (EMA).
                (e.g., https://pytorch.org/ignite/generated/ignite.handlers.ema_handler.EMAHandler.html)

        save_interval (int): if not None, number of steps at which the model is
            the model to a .pt file.

        resume_checkpoint (str): if not None, the model path from which
            to resume the training. It should be of the format
            path/to/modelNNNNNN.pt

        use_fp16 (bool): whether to convert model weights to 16 bytes
            during training

        fp16_scale_growth (float): a step to increase the lg_loss_scale
            when use_fp16 is True (see train_util.py)

        image_size (int): size of high-resolution output image.

        image_size_lr (int): size of low-resolution input image.

        num_input_channels (int): number of channels of input images.

        num_model_channels (int): base channel count for the model.

        num_res_blocks (int): number of residual blocks per downsample.

        num_heads (int): the number of attention heads in each attention layer.

        num_heads_upsample (int): deprecated

        attention_resolutions (str): a collection of downsample rates
            at which attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling,
            attention will be used.

        dropout (float): the dropout probability.

        learn_sigma (bool): if True, also learns the variance, and output will
            have twice as many channels as input.

        sigma_small (bool): if learn_sigma is False, determines the variance
            type in the diffusion model:
            FIXED_SMALL if sigma_small is True, else FIXED_LARGE

        class_cond (bool): if True, include a "y" key in returned dicts
            for class label. If classes are not available
            and this is True, an exception will be raised.

        num_class (int): number of data classes

        diffusion_steps (int): number of diffusion steps

        noise_schedule (str): a string that defines the variance scheduler
            for the beta coefficients in the diffusion model.
            Can be either "linear" or "cosine".

        timestep_respacing (str): if not None, then will be used to rescale the
            timesteps from the original number of timesteps.
            See respace.space_timesteps() for more details.

        loss_name (str): model loss type. One of 'mse', 'rescaled_mse',
            'kl' or 'rescaled_kl'

        output_type (str): determines what the model predicts.
            If 'epsilon', the model predicts the noise.
            If 'x_start', the model predicts x_0
            If 'x_previous', the model predicts x_{t-1}
            NOTE only useful for MSE loss

        rescale_timesteps (bool): whether to rescale timesteps,
            see rescale.WrappedModel for more details. if True, pass
            floating point timesteps into the model so that they are
            always scaled like in the paper "Denoising Diffusion Probabilistic Models"
            in (0 to 1000).

        use_checkpoint (bool): use gradient checkpointing to reduce memory usage.

        use_scale_shift_norm (bool): If True, mimics batch normalisation in UNet model.

        url_folder_experiment (str): path to model folder

        compute_val (bool): if False, the validation loss is not
                computed at each epoch (can be very long).

        subset_timesteps (int): if compute_val is True and subset_timesteps
            is not None, calculate the validation loss on the last
            'subset_timesteps' diffusion time steps
            (i.e. {0, ..., subset_timesteps-1})

        crop (bool): whether to crop (not resize) images to
            the desired image_size
    '''
    # Create dictionary with model arguments
    if resume_checkpoint is None:
        args_dic = {
            'type_dataset': 'image',
            'data_train_dir': data_train_dir,
            'data_val_dir': data_val_dir,
            'num_epochs': num_epochs,
            'schedule_sampler': schedule_sampler,
            'lr': lr,
            'weight_decay': weight_decay,
            'lr_anneal_steps': lr_anneal_steps,
            'batch_size': batch_size,
            'microbatch': microbatch,
            'ema_rate': ema_rate,
            'save_interval': save_interval,
            'resume_checkpoint': resume_checkpoint,
            'use_fp16': use_fp16,
            'fp16_scale_growth': fp16_scale_growth,
            'image_size': image_size,
            'image_size_lr': image_size_lr,
            'num_input_channels': num_input_channels,
            'class_cond': class_cond,
            'num_class': num_class,
            'learn_sigma': learn_sigma,
            'sigma_small': sigma_small,
            'num_model_channels': num_model_channels,
            'num_res_blocks': num_res_blocks,
            'num_heads': num_heads,
            'num_heads_upsample': num_heads_upsample,
            'attention_resolutions': attention_resolutions,
            'dropout': dropout,
            'diffusion_steps': diffusion_steps,
            'noise_schedule': noise_schedule,
            'timestep_respacing': timestep_respacing,
            'output_type': output_type,
            'rescale_timesteps': rescale_timesteps,
            'loss_name': loss_name,
            'use_checkpoint': use_checkpoint,
            'use_scale_shift_norm': use_scale_shift_norm,
            'compute_val': compute_val,
            'crop': crop
        }
    else:
        # resume_checkpoint should be of the form path/to/modelNNNNNN.pt
        cond_1 = resume_checkpoint.endswith('.pt')
        cond_2 = (len(resume_checkpoint.split('model')[-1]) == 9)
        if not (cond_1 and cond_2):
            raise ValueError('The resume_checkpoint is not properly provided.')

        args_dic = load_args_dic(resume_checkpoint)
        args_dic['resume_checkpoint'] = resume_checkpoint

        # Overwrite arguments if needed
        args_dic['compute_val'] = compute_val
        args_dic['lr'] = lr
        args_dic['num_epochs'] = num_epochs

        # Get url_folder_experiment from resume_checkpoint
        url_folder_experiment = str(Path(resume_checkpoint).parent)

    # Check inputs
    check_inputs(args_dic)

    # Create model log folder
    if not os.path.exists(url_folder_experiment):
        os.makedirs(url_folder_experiment, exist_ok=True)

    # Dump model arguments to .json
    with open(f"{url_folder_experiment}/metadata.json", "w") as fp:
        json.dump(args_dic, fp)

    print("Creating model and diffusion...")
    model, diffusion = create_sr_model_and_diffusion(args_dic)
    model.to(DEVICE_ID)
    schedule_sampler = create_named_schedule_sampler(
        args_dic['schedule_sampler'],
        diffusion)

    print("Creating data loaders...")
    data_train = load_data_superres(
        data_dir=args_dic['data_train_dir'],
        batch_size=args_dic['batch_size'],
        image_size_hr=args_dic['image_size'],
        image_size_lr=args_dic['image_size_lr'],
        num_channels=args_dic['num_input_channels'],
        class_cond=args_dic['class_cond'],
        num_class=args_dic['num_class'],
        crop=args_dic['crop'])
    data_val = load_data_superres(
        data_dir=args_dic['data_val_dir'],
        batch_size=args_dic['batch_size'],
        image_size_hr=args_dic['image_size'],
        image_size_lr=args_dic['image_size_lr'],
        num_channels=args_dic['num_input_channels'],
        class_cond=args_dic['class_cond'],
        num_class=args_dic['num_class'],
        crop=args_dic['crop'])

    # Initialise writer
    writer = SummaryWriter(url_folder_experiment)

    print("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data_train=data_train,
        data_val=data_val,
        num_epochs=args_dic['num_epochs'],
        batch_size=args_dic['batch_size'],
        microbatch=args_dic['microbatch'],
        lr=args_dic['lr'],
        ema_rate=args_dic['ema_rate'],
        save_interval=args_dic['save_interval'],
        resume_checkpoint=args_dic['resume_checkpoint'],
        logdir=url_folder_experiment,
        writer=writer,
        use_fp16=args_dic['use_fp16'],
        fp16_scale_growth=args_dic['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args_dic['weight_decay'],
        lr_anneal_steps=args_dic['lr_anneal_steps'],
        compute_val=compute_val,
        subset_timesteps=subset_timesteps
    ).run_loop()
