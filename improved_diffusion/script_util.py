from typing import Union, List

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel, SuperResModel


def create_model_and_diffusion(
        args_dic: dict
        ) -> None:
    '''
    Initialises the model and Gaussian diffusion process.
    '''
    model = create_model(
        args_dic['image_size'],
        args_dic['num_input_channels'],
        args_dic['num_model_channels'],
        args_dic['num_res_blocks'],
        learn_sigma=args_dic['learn_sigma'],
        class_cond=args_dic['class_cond'],
        num_class=args_dic['num_class'],
        use_checkpoint=args_dic['use_checkpoint'],
        attention_resolutions=args_dic['attention_resolutions'],
        num_heads=args_dic['num_heads'],
        num_heads_upsample=args_dic['num_heads_upsample'],
        use_scale_shift_norm=args_dic['use_scale_shift_norm'],
        dropout=args_dic['dropout'],
    )
    diffusion = create_gaussian_diffusion(
        steps=args_dic['diffusion_steps'],
        learn_sigma=args_dic['learn_sigma'],
        sigma_small=args_dic['sigma_small'],
        noise_schedule=args_dic['noise_schedule'],
        loss_name=args_dic['loss_name'],
        output_type=args_dic['output_type'],
        rescale_timesteps=args_dic['rescale_timesteps'],
        timestep_respacing=args_dic['timestep_respacing'],
    )
    return model, diffusion


def create_sr_model_and_diffusion(
        args_dic: dict
        ) -> None:
    '''
    Initialises the super-resolution model and
    Gaussian diffusion process
    '''
    model = create_sr_model(
        args_dic['image_size'],
        args_dic['image_size_lr'],
        args_dic['num_input_channels'],
        args_dic['num_model_channels'],
        args_dic['num_res_blocks'],
        learn_sigma=args_dic['learn_sigma'],
        class_cond=args_dic['class_cond'],
        num_class=args_dic['num_class'],
        use_checkpoint=args_dic['use_checkpoint'],
        attention_resolutions=args_dic['attention_resolutions'],
        num_heads=args_dic['num_heads'],
        num_heads_upsample=args_dic['num_heads_upsample'],
        use_scale_shift_norm=args_dic['use_scale_shift_norm'],
        dropout=args_dic['dropout'],
    )
    diffusion = create_gaussian_diffusion(
        steps=args_dic['diffusion_steps'],
        learn_sigma=args_dic['learn_sigma'],
        sigma_small=args_dic['sigma_small'],
        noise_schedule=args_dic['noise_schedule'],
        loss_name=args_dic['loss_name'],
        output_type=args_dic['output_type'],
        rescale_timesteps=args_dic['rescale_timesteps'],
        timestep_respacing=args_dic['timestep_respacing'],
    )
    return model, diffusion


def create_model(
        image_size: int,
        num_input_channels: int,
        num_model_channels: int,
        num_res_blocks: int,
        learn_sigma: bool,
        class_cond: bool,
        num_class: int,
        use_checkpoint: bool,
        attention_resolutions: str,
        num_heads: int,
        num_heads_upsample: int,
        use_scale_shift_norm: bool,
        dropout: float,
        ):
    '''
    Create Unet model.

    Inputs:
    ------
        image_size (int): size of input image.

        num_input_channels (int): number of channels in input image
            (e.g., 3 for RGB, 1 for InSAR data)

        num_model_channels (int): base channel count for the model.

        num_res_blocks (int): number of residual blocks per downsample.

        learn_sigma (bool): if True, also learns the variance, and output will
            have twice as many channels as input.

        class_cond (bool): if True, include a "y" key in returned dicts
            for class label. If classes are not available
            and this is True, an exception will be raised.

        num_class (int): number of data classes

        use_checkpoint (bool): use gradient checkpointing to reduce
            memory usage.

        attention_resolutions (str): a collection of downsample rates
            at which attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling,
            attention will be used.

        num_heads (int): the number of attention heads in each attention layer.

        dropout (float): the dropout probability.
    '''
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"Unsupported image size: {image_size}."
                         "Should be one of 32, 64, 128, 256 or 512.")

    attention_ds = [
        image_size // int(res) for res in attention_resolutions.split(",")
    ]

    unet = UNetModel(
        in_channels=num_input_channels,
        num_model_channels=num_model_channels,
        out_channels=(num_input_channels if not learn_sigma
                      else 2*num_input_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_class if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )
    unet.image_size = image_size
    return unet


def create_sr_model(
        image_size: int,
        image_size_lr: int,
        num_input_channels: int,
        num_model_channels: int,
        num_res_blocks: int,
        learn_sigma: bool,
        class_cond: bool,
        num_class: int,
        use_checkpoint: bool,
        attention_resolutions: str,
        num_heads: int,
        num_heads_upsample: int,
        use_scale_shift_norm: bool,
        dropout: float,
        ):
    '''
    Create Unet model.

    Inputs:
    ------
        image_size (int): size of high-resolution output image.

        image_size_lr (int): size of low-resolution input image.

        num_input_channels (int): number of channels in input image
            (e.g., 3 for RGB, 1 for InSAR data)

        num_model_channels (int): base channel count for the model.

        num_res_blocks (int): number of residual blocks per downsample.

        learn_sigma (bool): if True, also learns the variance, and output will
            have twice as many channels as input.

        class_cond (bool): if True, include a "y" key in returned dicts
            for class label. If classes are not available
            and this is True, an exception will be raised.

        num_class (int): number of data classes

        use_checkpoint (bool): use gradient checkpointing to reduce memory
            usage.

        attention_resolutions (str): a collection of downsample rates
            at which attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling,
            attention will be used.

        num_heads (int): the number of attention heads in each attention layer.

        dropout (float): the dropout probability.
    '''
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"Unsupported image size: {image_size}."
                         "Should be one of 32, 64, 128, 256 or 512.")

    _ = image_size_lr  # hack to prevent unused variable

    attention_ds = [
        image_size // int(res) for res in attention_resolutions.split(",")
    ]

    return SuperResModel(
        in_channels=num_input_channels,
        num_model_channels=num_model_channels,
        out_channels=(num_input_channels if not learn_sigma
                      else 2*num_input_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_class if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
        *,
        steps: int = 1000,
        learn_sigma: bool = False,
        sigma_small: bool = False,
        noise_schedule: str = "linear",
        loss_name: str = "mse",
        output_type: str = "epsilon",
        rescale_timesteps: bool = False,
        timestep_respacing: Union[List[int], str] = None
        ):
    '''
    Inputs:
    ------
        steps (int): number of diffusion steps.

        learn_sigma (bool): whether to also learn the variance.

        sigma_small (bool): if True, FIXED_SMALL will be used
            as the model's output variance. Else FIXED_LARGE.

        noise_schedule (str): It is either 'linear' or 'cosine'.

        loss_name (str): determines the model loss type
            (of class ModelLossType)

        output_type (str): determines what the model predicts.
            If 'epsilon', the model predicts the noise.
            If 'x_start', the model predicts x_0
            If 'x_previous', the model predicts x_{t-1}
            NOTE only useful for MSE loss

        rescale_timesteps (bool): whether to rescale the model timesteps.

        timestep_respacing (Union[List[int], str])): it corresponds to the
            input section_counts in the function space_timesteps in script
            respace.py
    '''

    if noise_schedule not in {'linear', 'cosine'}:
        raise ValueError('Unrecognised noise_schedule.')

    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if loss_name == 'kl':
        loss_type = gd.LossType.KL
    elif loss_name == 'rescaled_kl':
        loss_type = gd.LossType.RESCALED_KL
    elif loss_name == 'rescaled_mse':
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if output_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON
    elif output_type == 'x_start':
        model_mean_type = gd.ModelMeanType.START_X
    elif output_type == 'x_previous':
        model_mean_type = gd.ModelMeanType.PREVIOUS_X

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
