import json
import os
from pathlib import Path
import time
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn as nn
import tqdm

from feature_extraction.MNIST_classifier_Inception import load_best_model_url_inception_mnist
from feature_extraction.MNIST_classifier_VGG import load_best_model_url_vgg_mnist
from feature_extraction.utils_feature_extraction import (
    load_and_crop_VGG,
    load_and_crop_Inception)
from improved_diffusion.CONSTANTS import DEVICE_ID
from improved_diffusion.datasets_image import load_data
from improved_diffusion.gaussian_diffusion import get_named_beta_schedule
from improved_diffusion.script_util import create_model_and_diffusion
from sample_time_comparison import (
    plot_FID_results,
    plot_images_mosaic,
    plot_PrecisionRecall_results)
from utils.utils import (
    save_sample_to_png,
    read_model_metadata
)
from utils.utils_FID import get_fid
from utils.utils_sample_DPM import (
    DPM_Solver,
    model_wrapper,
    NoiseScheduleVP
)
from utils.utils_precision_and_recall import compute_precision_and_recall


##
# Perform the sampling with ODE techniques
##


def sample_from_ODE(
        model_path: str = None,
        clip_denoised: bool = True,
        max_clip_val: float = 1,
        sample_steps: int = 10,
        num_samples: int = 16,
        batch_size: int = 1,
        sampled_class: int = None,
        url_save_path: str = 'images/',
        algorithm_type: str = 'dpmsolver',
        method_type: str = 'multistep',
        skip_type: str = 'time-uniform',
        order: int = 3,
        plot: bool = True
        ) -> None:
    '''
    Code is adapted from here
        https://github.com/LuChengTHU/dpm-solver

    The input `model` has the following format:
        model(x, t_input, **model_kwargs) -> noise | x_start | v | score

    The shape of `x` is `(batch_size, **shape)`,
    and the shape of `t_continuous` is `(batch_size,)`.

    Inputs:
    -------

        sample_steps (int): number of sample time steps for the
            DPM-solver.

        num_samples (int): number of different samples to perform.

        sampled_class (int): if not None it is the class from which to
            sample.

        url_save_path (str): url of folder where to save the generated
            samples.

        algorithm_type (str): either dpmsolver or dpmsolver++; according to
            the original paper.

        method_type (str): either 'singlestep', 'multistep', 'singlestep_fixed'
            or 'adaptive'.

        order (int): either 1, 2 or 3.

        skip_type (str): Either "time-uniform", "time-quadratic" or "logSNR",
            where "logSNR" is recommended for low-resolution images, and 
            "time-uniform" is recommended for high-resolutional.
    '''

    if skip_type not in {'time-uniform', 'logSNR', 'time-quadratic'}:
        raise ValueError(f'the skip_type {skip_type} is unknown')

    if algorithm_type not in {'dpmsolver', 'dpmsolver++'}:
        raise ValueError(f'The sampling algorithm {algorithm_type} is unknown')

    if method_type not in {'singlestep', 'multistep', 'singlestep_fixed', 'adaptive'}:
        raise ValueError(f'The method type {method_type} is unknown')

    if order not in {1, 2, 3}:
        raise ValueError(f'the order {order} cannot be used.')

    # First read the model metadata from .json file
    url_model_folder = Path(model_path).parent
    url_metadata = [f for f in os.listdir(url_model_folder)
                    if f.endswith('.json')]
    if len(url_metadata) == 1:
        model_dic = read_model_metadata(f'{url_model_folder}/{url_metadata[0]}')
    else:
        raise ValueError('There is no or multiple metadata files associated with this model!')

    if model_dic['rescale_timesteps']:
        diffusion_steps = 1000
    else:
        diffusion_steps = model_dic['diffusion_steps']

    if model_dic['image_size'] <= 64:
        if skip_type != 'logSNR':
            print('For such low-resolution images, it is recommend to use skip_type equal to logSNR')
    else:
        if skip_type != 'time-uniform':
            print('For high-resolution images, it is recommend to use skip_type equal to time-uniform')

    # betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
    # And we need to set the betas or the alpha_cumprods but not the two.
    name_beta = 'cosine'
    if name_beta not in {'linear', 'cosine'}:
        raise ValueError('The beta scheme is not known.')
    betas = get_named_beta_schedule(name_beta, diffusion_steps)
    betas = torch.Tensor(betas)
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    # If singlestep, requires to set beta_1
    # noise_schedule.beta_1 ?

    # Check some of the inputs
    if not os.path.exists(model_path):
        raise ValueError('The specified model does not exist!')

    if max_clip_val <= 0:
        raise ValueError('The max_clip_val is negative.')

    if not os.path.exists(url_save_path):
        os.makedirs(url_save_path, exist_ok=True)

    if sampled_class is not None:

        if not model_dic['class_cond']:
            raise ValueError('Do not specify a class sample for unconditional models.')

        if sampled_class >= model_dic['num_class']:
            raise ValueError('This class is unavailable!')

    # Create dictionary of arguments to pass for model initialisation
    args_dic = {
        'model_path': model_path,
        'clip_denoised': clip_denoised,
        'max_clip_val': max_clip_val,
        'batch_size': batch_size,
        'use_ddim': False,
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

    def our_model_wrapper(
            model,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: bool = False,
            sample_class: int = None,
            max_clip_val=max_clip_val):
        '''
        The model should be transform to that appraoch:
            model(x, t_input, **model_kwargs) -> noise | x_start | v | score

        The `t_input` is the discrete-time labels (i.e. 0 to 999) of the model

        Inputs:
        -------

            cond (bool): if True the model is conditional and should
                be loaded accordingly (e.g. with a non-void model_kwargs
                dictionary).

            sample_class (int): the class from which we hope to sample. If
                None, the sample are performed from a random array of
                classes.

        '''

        if sample_class is not None:
            if sample_class >= args_dic['num_class']:
                raise ValueError(f'Not enough class to sample class {sample_class}')

        model_kwargs = {}

        if cond:
            if sample_class is None:
                classes = torch.randint(
                        low=0, high=args_dic['num_class'],
                        size=(batch_size,),
                        device=DEVICE_ID)
            else:
                classes = torch.tensor(
                    [sample_class] * batch_size,
                    device=DEVICE_ID)

            model_kwargs["y"] = classes

        dic = diffusion.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=True,
            max_clip_val=max_clip_val,
            denoised_fn=None,
            model_kwargs=model_kwargs
            )

        pred_xstart = dic['pred_xstart']
        noise = diffusion._predict_eps_from_xstart(
            x, t, pred_xstart)

        return noise

    if model_dic['class_cond']:
        def model_wrap(
                x: torch.Tensor,
                t: torch.Tensor):
            return our_model_wrapper(
                model, x, t,
                cond=True,
                sample_class=sampled_class,
                max_clip_val=max_clip_val)
    else:
        def model_wrap(
                x: torch.Tensor,
                t: torch.Tensor):
            return our_model_wrapper(
                model, x, t,
                cond=False,
                sample_class=None,
                max_clip_val=max_clip_val)

    model_fn = model_wrapper(
        model_wrap,
        noise_schedule,
        model_type="noise",
        model_kwargs={})

    # 3. Define dpm-solver and sample by singlestep DPM-Solver.
    # (We recommend singlestep DPM-Solver for unconditional sampling)
    # Adjust the `steps` to balance the computation
    # costs and the sample quality.
    dpm_solver = DPM_Solver(
        model_fn,
        noise_schedule,
        algorithm_type=algorithm_type)

    all_images = []
    num_batch = num_samples // batch_size + 1 * (num_samples % batch_size > 0)

    for batch_cnt in tqdm.tqdm(np.arange(num_batch)):

        x_T = torch.randn((
            batch_size,
            model_dic['num_input_channels'],
            model_dic['image_size'],
            model_dic['image_size']),
            device=DEVICE_ID)

        x_sample = dpm_solver.sample(
            x_T,
            steps=sample_steps,
            order=order,
            skip_type="time-uniform",
            method=method_type,
        )
        # From (B, C, H, H) to (B, H, H, C)
        x_sample = torch.moveaxis(x_sample, 1, -1)
        # Optimize the memory footprint of sample
        x_sample = x_sample.contiguous()

        all_images.append(x_sample.cpu().numpy())

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
                plt.savefig(f'{url_save_path}/sample{batch_cnt*batch_size+i}_numsteps_{sample_steps}.png')
                plt.close()

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]

    if model_dic['class_cond']:
        if sampled_class is not None:
            label_arr = np.array([sampled_class] * num_samples)
        else:
            print('Here, we give false labels to the sampled images (TO SOLVE)')
            label_arr = np.random.randint(0, args_dic['num_class'],
                                          num_samples)

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = f"{url_save_path}/samples_{shape_str}.npz"

    print(f"Saving to {out_path}")
    if model_dic['class_cond']:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)


##
# Comparison and evaluation of the various parameters
##


def make_time_comparison_ODE(
        url_PDM_model: str,
        name_id_save: str,
        dataloader_real: torch.utils.data.DataLoader,
        url_folder_save_img: str,
        l_type_scores: List[str] = ['FID', 'Precision', 'Recall'],
        nhood_size: int = 3,
        image_size=32,
        nbr_channels=1,
        url_model_feature: nn.Module = None,
        type_model_feature='VGG',
        layer_size: int = None,
        l_nbr_diff_steps: List[int] = [10, 50, 100, 200],
        l_method_sampling: List[dict] = [{}],
        nbr_samples=1000,
        sampled_class: int = None,
        max_clip_val=1) -> None:
    """
    Similar to make_time_comparison in sample_time_comparison.py
    only that it evaluates the various parameters of the ODE sampling.
    Here, the description of the inputs has been lighten, please refer
    to the function make_time_comparison.

    Inputs:
    ------
        url_PDM_model (str): url of the PDM model trained with or without
            learning the variance. It should be in a folder that contains
            a single json file to load metadata.

        name_id_save (str): str use as identifier.

        dataloader_real (torch.utils.data.DataLoader): Dataloader for the
            training dataset.

        url_folder_save_img (str): url of the folder where to save the images.

        l_type_scores (List[str]): the list of the various scores we seek to
            compute and store in a json file of format:
                dic_<type_score>.json

        nhood_size (int): List of int corresponding to the k in the
            k-nearest neighbor that will be used to compute the Manifold for
            computed the improved Precision and Recall.

        image_size (int): size of the real and generated images.

        nbr_channels (int): number of channel of the real and generated
            images.

        url_model_feature (str): if not None, it is the url of trained model
            from which a representation of the training/generated will be
            performed to compute the various generation scores.

        type_model_feature (str): Here, either 'VGG' or 'Inception'.

        layer_size (int): if not None, it is the size of the layer to extract
            from the model.

        l_nbr_diff_steps (List[int]): list of the various diffusion steps we
            try to compare.

        l_method_sampling (List[dict]): a list od dict that list all the various
            parameters of sampling we would like to test.

        nbr_samples (int): nbr of different sample we generate for each pair
            of sampling method and number of diffusion steps.

        sampled_class (int): if not None, the generated samples are sample
            only from one class.

        max_clip_val (float): non-negative float use during the sampling
            to cropped the learnt parameters in order to better constraint
            the sampling range.
    """

    if nhood_size >= nbr_samples:
        raise ValueError('the k in kNN must be smaller than the number of sample')

    if url_model_feature is not None:
        if not os.path.exists(url_model_feature):
            raise ValueError('The url of the model from which to extract a data representation does not exist.')

    if not set(l_type_scores).issubset(['FID', 'Precision', 'Recall']):
        raise ValueError('Some of the scores to computes are unknown or not implemented.')

    if 'Precision' in l_type_scores or 'Recall' in l_type_scores:
        if not {'Precision', 'Recall'}.issubset(set(l_type_scores)):
            raise ValueError('We expect')

    if type_model_feature not in ['VGG', 'Inception']:
        raise ValueError('We do not use this type of model for feature extraction.')

    if not os.path.exists(url_PDM_model):
        raise ValueError('The train PDM model does not exist')

    if not os.path.exists(url_folder_save_img):
        os.makedirs(url_folder_save_img, exist_ok=True)

    # Get metadata for both models
    url_model_folder = Path(url_PDM_model).parent
    url_metadata = [f for f in os.listdir(url_model_folder) if f.endswith('.json')]

    if len(url_metadata) == 1:
        model_dic = read_model_metadata(f'{url_model_folder}/{url_metadata[0]}')
    elif not url_metadata:
        raise ValueError('There is no metadata file associated with this model!')
    else:
        raise ValueError('Too many json, do not know which one is associated to the model.')

    # Let us first create all the samples
    dic_time = {}
    url_json_time = f'{url_folder_save_img}/dic_time.json'

    list_keys = ['solver', 'method_type', 'skip_type', 'order']
    for _, dic_method in enumerate(l_method_sampling):

        if set(list(dic_method.keys())) != set(list_keys) :
            raise ValueError('The dictionnary providing the ODE sampler values is inconsistent')

        if os.path.exists(url_json_time):
            break

        for _, nbr_diff in enumerate(l_nbr_diff_steps):

            key = (f"{dic_method['solver']}_{dic_method['method_type']}_"
                f"{dic_method['skip_type']}_order_{dic_method['order']}_diffsteps_{nbr_diff}")

            # Check that the folder of sample does not already exist
            url_save_sample_folder = f'{url_folder_save_img}/{name_id_save}_{key}_nbrsample_{nbr_samples}/'
            if os.path.exists(url_save_sample_folder):
                print(f'For {key}, the folder already exists')
                continue
            else:
                os.makedirs(url_save_sample_folder, exist_ok=True)

            # Sample for the given method
            t0 = time.time()

            sample_from_ODE(
                url_PDM_model,
                clip_denoised=True,
                max_clip_val=max_clip_val,
                sample_steps=nbr_diff,
                num_samples=nbr_samples,
                batch_size=16,
                sampled_class=sampled_class,
                url_save_path=url_save_sample_folder,
                algorithm_type=dic_method['solver'],
                method_type=dic_method['method_type'],
                skip_type=dic_method['skip_type'],
                order=dic_method['order'],
                plot=False)

            t1 = time.time()

            dic_time[key] = t1 - t0

            url_npz = [url for url in os.listdir(url_save_sample_folder) if url.endswith('.npz')]
            if len(url_npz) != 1:
                raise ValueError('Issue in saving the .npz during saving')
            # Change .png and .pz to 
            url_npz = f'{url_save_sample_folder}/{url_npz[0]}'

            # We need to read each image a .png file for using the
            # dataloader later on.
            for idx in range(nbr_samples):
                name_save_png = f'{idx}.png'
                save_sample_to_png(
                    url_npz,
                    idx,
                    url_save_sample_folder,
                    name_save_png,
                    class_cond=model_dic['class_cond'])

    # Save time dictionary for sampling with various nbr of diffusion steps and
    # sampling methods
    if not os.path.exists(url_json_time):
        with open(url_json_time, "w") as f:
            json.dump(dic_time, f)

    # Initialise the scores dictionaries
    if "FID" in l_type_scores:
        # Then we compute the FID for each of these at different times.
        dic_FID = {}
        url_json_fid = f'{url_folder_save_img}/dic_fid_{type_model_feature}.json'
    if "Precision" in l_type_scores:
        dic_precision, dic_recall = {}, {}
        url_json_precision = f'{url_folder_save_img}/dic_precision_{type_model_feature}.json'
        url_json_recall = f'{url_folder_save_img}/dic_recall_{type_model_feature}.json'

    # Cut the model feature to extract the feature layer of interest.
    if type_model_feature == 'VGG':
        model_for_extraction = load_and_crop_VGG(
            url_model_feature,
            last_layer=True)
    elif type_model_feature == 'Inception':
        model_for_extraction = load_and_crop_Inception(
            url_model_feature,
            layer_size=layer_size,
            in_01=False)

    # Url where to save an hdf5 file to contain the feature representation
    # of the data.
    url_save_real_feature = f'{url_model_folder}/{name_id_save}_{type_model_feature}_{layer_size}.hdf5'
    # It is a different file for the FID computation be we store different elements.
    url_save_real_feature_fid = f'{url_model_folder}/FID_{name_id_save}_{type_model_feature}_{layer_size}.hdf5'

    for _, dic_method in enumerate(l_method_sampling):
        for _, nbr_diff in enumerate(l_nbr_diff_steps):
            key = (f"{dic_method['solver']}_{dic_method['method_type']}_"
                f"{dic_method['skip_type']}_order_{dic_method['order']}_diffsteps_{nbr_diff}")

            # Identify the folder where lies the samples data.
            url_save_sample_folder = f'{url_folder_save_img}/{name_id_save}_{key}_nbrsample_{nbr_samples}/'

            if not os.path.exists(url_save_sample_folder):
                raise ValueError(f'The folder {url_save_sample_folder} that should contain the samples does not exist!')

            dataloader_gen = load_data(
                data_dir=url_save_sample_folder,
                batch_size=16,
                image_size=image_size,
                num_channels=nbr_channels,
                class_cond=True,
                num_class=10,
                deterministic=False,
                crop=False,
                droplast=False)

            if 'FID' in l_type_scores:

                if os.path.exists(url_json_fid):
                    print(f'FID for experiment {key} already computed')
                else:
                    fid = get_fid(
                        dataloader_real,
                        dataloader_gen,
                        nbr_channels,
                        model_for_extraction,
                        url_save_real_feature_fid,
                        nbr_samples
                        )

                    dic_FID[key] = fid

            if 'Precision' in l_type_scores:

                if os.path.exists(url_json_precision):
                    print(f'P&R for experiment {key} already computed')
                else:
                    metric_results = compute_precision_and_recall(
                        dataloader_real,
                        dataloader_gen,
                        nbr_channels,
                        model_for_extraction,
                        url_save_real_feature,
                        [nhood_size],
                        nbr_samples)

                    # Extract Precision and Recall from the metric_results.
                    precision, recall = metric_results

                    dic_precision[key] = precision
                    dic_recall[key] = recall

    # Save the scores in a json file.
    if 'FID' in l_type_scores:
        if not os.path.exists(url_json_fid):
            with open(url_json_fid, "w") as f:
                json.dump(dic_FID, f)

    if 'Precision' in l_type_scores:
        if not os.path.exists(url_json_precision):
            with open(url_json_precision, "w") as f:
                json.dump(dic_precision, f)
            with open(url_json_recall, "w") as f:
                json.dump(dic_recall, f)

    return


def create_dict_input_ODE_method(
        l_solver: List[str] = ['dpmsolver'],
        l_method_type: List[str] = ['singlestep'],
        l_skip_type: List[str] = ['logSNR'],
        l_order: List[int] = [1, 2, 3]) -> List[dict]:
    '''
    Create a list of dictionnaries that contains the four following
    keys 'solver', 'method_type', 'skip_type' and 'order'. This
    list specifies various different methods.

    Recall that  for dataset with low-resolution images, it is recommended
    to use skip_type equal to 'logSNR' and for high-resolution datasets, it is
    recommended to use skip_type equal to 'time-uniform'.

    '''
    l_all_solver = ['dpmsolver', 'dpmsolver++']
    l_all_method = ['singlestep', 'multistep', 'singlestep_fixed', 'adaptive']
    l_all_skip_type = ['time-uniform', 'logSNR', 'time-quadratic']
    l_all_order = [1, 2, 3]

    # Check the input values here.
    if not set(l_solver).issubset(set(l_all_solver)):
        raise ValueError('The list of different solvers is not well specified.')

    if not set(l_method_type).issubset(l_all_method):
        raise ValueError('The list of different methods is not well specified.')

    if not set(l_skip_type).issubset(l_all_skip_type):
        raise ValueError('The list of all skip type is not well specified')

    if not set(l_order).issubset(l_all_order):
        raise ValueError('The list of all orders is not well specified')

    l_dict = []
    dic = {}
    for solver_val in l_solver:
        for method_type_val in l_method_type:
            for skip_type in l_skip_type:
                for order_val in l_order:
                    dic['solver'] = solver_val
                    dic['method_type'] = method_type_val
                    dic['skip_type'] = skip_type
                    dic['order'] = order_val
                    if dic['method_type'] == 'adaptive' and dic['order'] == 1:
                        print("For adaptive step size solver, order must be 2 or 3.")
                        dic = {}
                        continue
                    l_dict.append(dic)
                    dic = {}

    return l_dict


def compare_approaches_MNIST(
        type_model_feature: str = 'VGG',
        url_folder_models_feature: str = None,
        l_nbr_diff_steps: List[int] = [5, 10, 20, 30],
        nbr_samples: int = 1000,
        l_method_sampling: List[dict] = [{}],
        url_folder: str = None,
        layer_size: int = None,
        sampled_class: int = None,
        url_PDM_model_cond: str = None,
        url_PDM_model_uncond: str = None) -> None:
    '''
    Comparing the various ODE sampling parameters in the case
    of the MNIST dataset. The comparison is made according to
    three criteria: FID, Improved Precision&Recall and normal
    sampling.

    Inputs:
    ------
        type_model_feature (str): Either 'VGG' or 'Inception'
            and corresponds to the feature extractor we use for
            computing the method. Note that for 'VGG', there is
            no need to choose a layer_size; while for 'Inception'
            it is not implemented yet.

        url_folder_models_feature (str): folder where to find the
            various fine-tuned feature extractor for MNIST (VGG
            or Inception).

        l_nbr_diff_steps (List[int]): the various number of
            sampling time steps at which to perform the comparison.

        nbr_samples (int): The number of samples to sample for each
            set of parameters.

        l_method_sampling (List[dict]): list of dictionnaries containing
            the various parameters at which to perform the ODE parameters.

        url_folder (str): url where to store the PDM samples according
            to various parameters.

        layer_size (int): If not None, it corresponds to the layer at
            which feature are extracted in the Inception network.

        sampled_class (int): if not None and cond is True, then the
            samples belong to this class.

        cond (bool): if True, we consider a conditional PDM model and
            otherwise an unconditional model.
    '''

    if not os.path.exists(url_folder):
        os.makedirs(url_folder, exist_ok=True)

    if type_model_feature == 'Inception':
        if layer_size not in [64, 192, 768, 2048]:
            raise ValueError('For the Inception feature extractor, the layer-size has specific values.')

    if type_model_feature not in {'VGG', 'Inception'}:
        raise ValueError('Unknown type_model_feature.')

    if url_PDM_model_cond is None and url_PDM_model_uncond is None:
        raise ValueError('You need to specify at least one model.')

    if url_PDM_model_cond is not None and url_PDM_model_uncond is not None:
        raise ValueError('You need to specify only one model.')

    # The conditional and unconditional PDM models for MNIST
    if url_PDM_model_cond is not None:
        # url_model = f"./models_data/ODE_experiments/mnist_diffsteps_1000/ema_0.9999_35_vb=0.9274_mse=0.0013.pt"
        url_model = url_PDM_model_cond
        if not os.path.exists(url_PDM_model_cond):
            raise ValueError('You need to setup a trained PDM model for MNIST.')
        name = 'MNIST_conditional'
    else:
        # url_model = f"./models_data/ODE_experiments/mnist_uncond_diffsteps_1000/ema_0.9999_23_vb=0.9521_mse=0.0013.pt"
        url_model = url_PDM_model_uncond
        if not os.path.exists(url_PDM_model_uncond):
            raise ValueError('You need to setup a trained PDM model for MNIST.')
        name = 'MNIST_unconditional'

    if not os.path.exists(url_model):
        raise ValueError('You need to setup a trained PDM model for MNIST.')

    name_id_save = 'mnist'
    data_dir_val = 'data/mnist_test/'

    val_loader = load_data(
            data_dir=data_dir_val,
            batch_size=16,
            image_size=32,
            num_channels=1,
            class_cond=True,
            num_class=10,
            deterministic=False,
            crop=False,
            droplast=False)

    if not os.path.exists(url_folder_models_feature):
        raise ValueError('The url of the model from which to extract a data representation does not exist.')

    if type_model_feature == 'VGG':
        url_folder_classifier = load_best_model_url_vgg_mnist(
             url_folder_models_feature)
    else:
        url_folder_classifier = load_best_model_url_inception_mnist(
             url_folder_models_feature)

    make_time_comparison_ODE(
        url_model,
        name,
        val_loader,
        url_folder,
        l_type_scores=['FID', 'Precision', 'Recall'],
        image_size=32,
        nhood_size=3,
        layer_size=layer_size,
        url_model_feature=url_folder_classifier,
        type_model_feature=type_model_feature,
        l_nbr_diff_steps=l_nbr_diff_steps,
        l_method_sampling=l_method_sampling,
        nbr_samples=nbr_samples,
        sampled_class=sampled_class,
        max_clip_val=1)

    # Make a Plot of FID results as a function of time
    plot_FID_results(
        url_folder,
        type_model_feature=type_model_feature,
        ODE_plot=True,
        FID_max=200)

    # Plot a mosaic of examples
    plot_images_mosaic(
        url_folder,
        ODE_plot=True)

    # Plot the precision and recall scores
    plot_PrecisionRecall_results(
        url_folder,
        name_id_save,
        type_model_feature=type_model_feature,
        ODE_plot=True)