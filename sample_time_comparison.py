'''
Here, we compare the sample time in terms of the FID obtained or other metrics.
'''
import json
import os
import pdb
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn

from feature_extraction.utils_feature_extraction import (
    InceptionV3,
    load_and_crop_VGG
)
from feature_extraction.MNIST_classifier_VGG import load_best_model_url_vgg_mnist
from improved_diffusion.CONSTANTS import (
    DEVICE_ID,
    DIC_LINE,
    ODE_METHODS_SC,
    ODE_SKIP_SC)
from improved_diffusion.datasets_image import load_data
from sample import read_model_metadata, main_sample
from utils.utils import save_sample_to_png
from utils.utils_FID import get_fid
from utils.utils_precision_and_recall import compute_precision_and_recall


def make_time_comparison(
        url_model: str,
        url_model_var: str,
        name_id_save: str,
        dataloader_real: torch.utils.data.DataLoader,
        url_folder_save_img: str,
        l_type_scores: List[str] = ['FID', 'Precision', 'Recall'],
        nhood_size: int = 3,
        image_size=32,
        num_channels=1,
        in_01: (bool) = False,
        url_model_feature: nn.Module = None,
        type_model_feature='VGG',
        layer_size: int = None,
        l_num_diff_steps: List[int] = [10, 50, 100, 200],
        l_method_sampling: List[str] = ['DDIM', 'variance'],
        num_samples=1000,
        label_samples: int = None,
        max_clip_val=1) -> None:
    """
    Here, given a trained PDM, we sample (according to various methods)
    the model with various numbers of diffusion steps.
    The samples are then saved in various folders to later
    compute the generative scores.

    Inputs:
    ------
        url_model (str): url of the model that is trained without learning
            the variance of the reverse diffusion process. It should be in a
            folder that contains a single json file to load metadata. Note
            that the model should be conditional to compute some metrics.

        url_model_var (str): url of the model that is trained to learn
            the variance of the reverse diffusion process. It should be in a
            folder that contains a single json file to load metadata. Note that
            the model should be conditional to compute some metrics.

        name_id_save (str): str use as identifier.

        dataloader_real (torch.utils.data.DataLoader): Dataloader for the
            training dataset.

        url_folder_save_img (str): url of the folder where to save the images.

        l_type_scores (List[str]): the list of the various scores we seek to
            compute and store in a json file of format:
                dic_<type_score>.json
            "FID" correspond to the Frechet Inception Distance, and "Precision"
            and "Recall" goes together and corresponds to the improved
            Precision and Recall metric.
            In each case, the latent representation of the training set or
            generated samples is extracted from url_model_feature. This model
            can corresponds to a fine-tuned classical classifier (vgg or Inception).

        nhood_size (int): List of int corresponding to the k in the
            k-nearest neighbor that will be used to compute the Manifold for
            computed the improved Precision and Recall.

        image_size (int): size of the real and generated images.

        num_channels (int): number of channel of the real and generated
            images.

        in_01 (bool): if True, then the data is naturally in the [0, 1]
            range, and otherwise it expects the data to be in the range
            [-1, 1] (which is the case of the SAR dataset). This only matters
            for the Inception network to extract latent representation
            in order to compute generation scores.

        url_model_feature (str): if not None, it is the url of trained model
            from which a representation of the training/generated will be
            performed to compute the various generation scores.

        type_model_feature (str): Either 'VGG', 'Inception'.
            If 'VGG' or 'Inception', then url_model_feature is a fine-tuned
            version of the VGG on the (labeled) training database.

        layer_size (int): if not None, it is the size of the layer to extract
            from the model.

        l_num_diff_steps (List[int]): list of the various diffusion steps we
            try to compare.

        l_method_sampling (List[str]): list of the name of the various methods,
            i.e., 'original' when not learning the variance of the reverse
            diffusion process, 'variance' when learning the variance of the
            reverse diffusion process, 'DDIM' when Denoised Diffusion Implicit
            Model which corresponds to learning a faster reverse process that
            fit the marginal of the reverse diffusion proces, 'DDIM-var'
            when combining 'variance' and 'DDIM' and finally 'ODE' when
            leveraging that approximate sample is equivalent to
            approximately solving an ODE.

        num_samples (int): number of different samples we generate for each pair
            of sampling method and number of diffusion steps.

        label_samples (int): if not None, the generated samples are sample
            only from one class.

        max_clip_val (float): non-negative float use during the sampling
            to cropped the learnt parameters in order to better constraint
            the sampling range.
    """

    if nhood_size >= num_samples:
        raise ValueError('the k in kNN must be smaller than the number of sample')

    if url_model_feature is not None:
        if not os.path.exists(url_model_feature):
            raise ValueError('The url of the model from which to extract a data representation does not exist.')

    if not set(l_method_sampling).issubset(['original', 'DDIM', 'variance', 'DDIM-var', 'ODE']):
        raise ValueError('Some of the fast sampling methods are unknown or not implemented.')

    if not set(l_type_scores).issubset(['FID', 'Precision', 'Recall']):
        raise ValueError('Some of the scores to computes are unknown or not implemented.')

    if 'Precision' in l_type_scores or 'Recall' in l_type_scores:
        if not {'Precision', 'Recall'}.issubset(set(l_type_scores)):
            raise ValueError('We expect')

    if type_model_feature not in ['VGG', 'Inception']:
        raise ValueError('We do not use this type of model for feature extraction.')

    if type_model_feature == 'Inception' and layer_size is None:
        raise ValueError('If working with Inception, you need to specify layer_size and not let it to None')

    if type_model_feature == 'Inception' and layer_size is not None:
        if layer_size not in [64, 192, 768, 2048]:
            raise ValueError(f'Layer size of {layer_size} do not work for Inception.')

    if not os.path.exists(url_model):
        raise ValueError('The train PDM without variance does not exist')

    if not os.path.exists(url_model_var):
        raise ValueError('The train PDM with variance does not exist')

    if not os.path.exists(url_folder_save_img):
        os.makedirs(url_folder_save_img, exist_ok=True)

    # Get metadata for both models
    url_model_folder = Path(url_model).parent
    url_metadata = [f for f in os.listdir(url_model_folder) if f.endswith('.json')]

    if len(url_metadata) == 1:
        model_dic = read_model_metadata(f'{url_model_folder}/{url_metadata[0]}')
    elif not url_metadata:
        raise ValueError('There is no metadata file associated with this model!')
    else:
        raise ValueError('Too many json, do not know which one is associated to the model.')

    url_model_folder = Path(url_model_var).parent
    url_metadata = [f for f in os.listdir(url_model_folder) if f.endswith('.json')]
    if len(url_metadata) == 1:
        model_dic_var = read_model_metadata(f'{url_model_folder}/{url_metadata[0]}')
    elif not url_metadata:
        raise ValueError('There is no metadata file associated with this model!')
    else:
        raise ValueError('Too many json, do not know which one is associated to the model.')

    if not (model_dic['class_cond'] and model_dic_var['class_cond']):
        raise ValueError('Both PDMS models should be conditional for computing these metrics.')

    # Let us first create all the samples
    dic_time = {}
    url_json_time = f'{url_folder_save_img}/dic_time.json'
    for _, method in enumerate(l_method_sampling):

        if os.path.exists(url_json_time):
            break

        # Determine the parameter associated to each
        if method == 'DDIM':
            url_model_use = url_model
            use_ddim = True
        elif method == 'variance':
            url_model_use = url_model_var
            use_ddim = False
        elif method == 'original':
            url_model_use = url_model
            use_ddim = False
        elif method == 'DDIM-var':
            url_model_use = url_model_var
            use_ddim = True
        elif method == 'ODE':
            url_model_use = url_model
            use_ddim = False
            raise ValueError('Not implemented yet')
        else:
            raise ValueError(f'We do not know method {method}')

        for _, num_diff in enumerate(l_num_diff_steps):

            key = f'{method}_{num_diff}'

            # Check that the folder of sample does not already exist
            url_save_sample_folder = f'{url_folder_save_img}/{method}_numdiff_{num_diff}_numsample_{num_samples}/'
            if os.path.exists(url_save_sample_folder):
                print(f'For {method} with {num_diff} diff steps, the folder already exists')
                continue
            else:
                os.makedirs(url_save_sample_folder, exist_ok=True)

            # Sample for the given method
            t0 = time.time()

            main_sample(
                model_path=url_model_use,
                clip_denoised=True,
                max_clip_val=max_clip_val,
                num_samples=num_samples,
                sample_class=label_samples,
                batch_size=16,
                use_ddim=use_ddim,
                diffusion_steps=num_diff,
                url_save_path=url_save_sample_folder,
                plot=False)
            t1 = time.time()

            dic_time[key] = t1 - t0

            url_npz = [url for url in os.listdir(url_save_sample_folder) if url.endswith('.npz')]
            if len(url_npz) != 1:
                raise ValueError('Issue in saving the .npz during saving')
            # Change .png and .pz to 
            url_npz = f'{url_save_sample_folder}/{url_npz[0]}'
            for idx in range(num_samples):
                name_save_png = f'{name_id_save}_{method}_{num_diff}_{idx}.png'
                save_sample_to_png(
                    url_npz,
                    idx,
                    url_save_sample_folder,
                    name_save_png,
                    class_cond=False)

    # Save time dictionary for sampling with various numbers of diffusion steps and
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
        # In the following we can either 64, 192, 768 or 2048
        if layer_size is not None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[layer_size]
        else:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model_for_extraction = InceptionV3(
            [block_idx],
            normalise_input=in_01,
            )
        model_for_extraction = model_for_extraction.to(DEVICE_ID)
    else:
        raise ValueError('No other option for feature extraction')

    # Url where to save an hdf5 file to contain the feature representation
    # of the data.
    url_save_real_feature = f'{url_model_folder}/{name_id_save}_{type_model_feature}_{layer_size}.hdf5'
    # It is a different file for the FID computation be we store different elements.
    url_save_real_feature_fid = f'{url_model_folder}/FID_{name_id_save}_{type_model_feature}_{layer_size}.hdf5'

    for _, method in enumerate(l_method_sampling):
        for _, num_diff in enumerate(l_num_diff_steps):
            key = f'{method}_{num_diff}'
            print(key)

            # Identify the folder where lies the samples data.
            url_save_sample_folder = f'{url_folder_save_img}/{method}_numdiff_{num_diff}_numsample_{num_samples}/'
            dataloader_gen = load_data(
                data_dir=url_save_sample_folder,
                batch_size=16,
                image_size=image_size,
                num_channels=num_channels,
                class_cond=True,
                num_class=10,
                crop=False,
                droplast=False)

            if 'FID' in l_type_scores:

                if os.path.exists(url_json_fid):
                    print(f'FID for {method} and {num_diff} diffusion steps already computed')
                else:
                    fid = get_fid(
                        dataloader_real,
                        dataloader_gen,
                        num_channels,
                        model_for_extraction,
                        url_save_real_feature_fid,
                        num_samples
                        )

                    dic_FID[key] = fid

            if 'Precision' in l_type_scores:

                if os.path.exists(url_json_precision):
                    print(f'P&R for {method} and {num_diff} diffusion steps already computed')
                else:
                    metric_results = compute_precision_and_recall(
                        dataloader_real,
                        dataloader_gen,
                        num_channels,
                        model_for_extraction,
                        url_save_real_feature,
                        [nhood_size],
                        num_samples)

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

###
# Plots for the Precision and Recall scores.
###


def plot_PrecisionRecall_results(
        url_folder: str,
        name_id: str,
        type_model_feature='VGG',
        ODE_plot=False) -> None:
    '''
    In the url_folder, there should be at least three json files
        dic_precision_<type_model_feature>.json,
        dic_recall_<type_model_feature>.json,
        dic_time.json
    that contains the necessary data to generate time plots.
    Each contains key of the form
        <method>_<num_diffusion>.json

    We provide three differents plots:
        - Precision against time
        - Recall against time
        - Precision against recall with label of num
            of diffusion steps associated to each
            steps.

    Inputs:
    -------
        url_folder (str): see method's description.

        name_id (str): string to save plots with different names.

        type_model_feature (str): Either 'VGG', 'Inception'.

        ODE_plot (bool): If True, then we have a different
            syntax for the methods. 
    '''

    if type_model_feature not in ['VGG', 'Inception']:
        raise ValueError('type_model_feature is not known')

    if not os.path.exists(url_folder):
        raise ValueError(f'The folder {url_folder} does not exist.')

    if not (os.path.exists(f'{url_folder}/dic_time.json')):
        raise ValueError('The time dictionary does not exists')

    if not (os.path.exists(f'{url_folder}/dic_precision_{type_model_feature}.json')):
        raise ValueError('The precision dictionary does not exists')

    if not (os.path.exists(f'{url_folder}/dic_recall_{type_model_feature}.json')):
        raise ValueError('The recall dictionary does not exists')

    # Opening JSON file
    f = open(f'{url_folder}/dic_precision_{type_model_feature}.json')
    dic_precision = json.load(f)

    f = open(f'{url_folder}/dic_recall_{type_model_feature}.json')
    dic_recall = json.load(f)

    f = open(f'{url_folder}/dic_time.json')
    dic_time = json.load(f)

    dic_keys = [key for key in dic_precision.keys()]
    if set([key for key in dic_time.keys()]) != set(dic_keys):
        raise ValueError('time dictionary and precision do not agree')
    if set([key for key in dic_recall.keys()]) != set(dic_keys):
        raise ValueError('recall dictionary and precision do not agree')

    if not ODE_plot:
        l_method = list(set([key.split('_')[0] for key in dic_keys]))
    else:
        # In the case of ODE, the name is of the format
        # "dpmsolver_adaptive_logSNR_order_3_diffsteps_10"
        l_method = list(set([key.split('_diffsteps_')[0] for key in dic_keys]))

    # Plot the precision/recall against the time
    for type_score in ['precision', 'recall']:
        plt.figure(figsize=(20, 20))
        plt.title(f'{type_score} from {type_model_feature} VS time',
                  fontsize=40)
        for method in l_method:

            # Get the data
            list_points = []
            list_label_points = []
            for key in dic_keys:
                # Check is the key is good
                if not ODE_plot:
                    if key.split('_')[0] != method:
                        continue

                if type_score == 'precision':
                    list_points.append((dic_time[key], dic_precision[key]))
                elif type_score == 'recall':
                    list_points.append((dic_time[key], dic_recall[key]))

                if not ODE_plot:
                    list_label_points.append(key.split('_')[1])
                else:
                    list_label_points.append(key.split('_diffsteps_')[1])

            # Plot them
            arr_points = np.array(list_points)

            if ODE_plot:
                # file of format
                # <naem_id>_dpmsolver_adaptive_logSNR_order_2_diffsteps_40_nbrsample_100
                # Need shortcut (SC)

                name_method = method.split('dpmsolver')[1].split('_')[1]
                name_skiptime = method.split('dpmsolver')[1].split('_')[2]
                name_order = method.split('order_')[1].split('_')[0]

                if name_method not in [key for key in ODE_METHODS_SC.keys()]:
                    raise ValueError(f'Issue in the formatting of {name_method}')

                if name_skiptime not in [key for key in ODE_SKIP_SC.keys()]:
                    raise ValueError(f'Issue in the formatting of {name_skiptime}')

                method_SC = (f'{ODE_METHODS_SC[name_method]}-'
                                f'{ODE_SKIP_SC[name_skiptime]}-{name_order}')
            else:
                method_SC = method

            if not ODE_plot:
                plt.plot(
                    arr_points[:, 0],
                    arr_points[:, 1],
                    DIC_LINE[method],
                    markersize=20,
                    label=method_SC)
            else:
                plt.plot(
                    arr_points[:, 0],
                    arr_points[:, 1],
                    'p-',
                    markersize=20,
                    label=method_SC)

            for i in range(arr_points.shape[0]):
                plt.annotate(
                    list_label_points[i],
                    (arr_points[i, 0], arr_points[i, 1]),
                    fontsize=25)

        plt.ylabel(f'{type_score}', fontsize=40)
        plt.xlabel('time (s)', fontsize=40)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(loc='upper left', fontsize=40)
        plt.grid()
        plt.savefig(f'images/{name_id}_{type_model_feature}_{type_score}_time_comparison.png')
        plt.show()

    # Plot Precision Against Recall
    plt.figure(figsize=(20, 20))
    plt.title(f'Precision VS Recall from {type_model_feature}', fontsize=40)
    for method in l_method:
        # Get the data
        list_points = []
        list_label_points = []
        for key in dic_keys:

            if not ODE_plot:
                # Check if the key is good
                if key.split('_')[0] != method:
                    continue

            list_points.append((dic_recall[key], dic_precision[key]))

            if not ODE_plot:
                list_label_points.append(key.split('_')[1])
            else:
                list_label_points.append(key.split('_diffsteps_')[1])

        # Plot them
        arr_points = np.array(list_points)

        if ODE_plot:
            # file of format
            # <naem_id>_dpmsolver_adaptive_logSNR_order_2_diffsteps_40_nbrsample_100
            # Need shortcut (SC)

            name_method = method.split('dpmsolver')[1].split('_')[1]
            name_skiptime = method.split('dpmsolver')[1].split('_')[2]
            name_order = method.split('order_')[1].split('_')[0]

            if name_method not in [key for key in ODE_METHODS_SC.keys()]:
                raise ValueError(f'Issue in the formatting of {name_method}')

            if name_skiptime not in [key for key in ODE_SKIP_SC.keys()]:
                raise ValueError(f'Issue in the formatting of {name_skiptime}')

            method_SC = (f'{ODE_METHODS_SC[name_method]}-'
                            f'{ODE_SKIP_SC[name_skiptime]}-{name_order}')
        else:
            method_SC = method

        if not ODE_plot:
            plt.plot(
                arr_points[:, 0], arr_points[:, 1],
                DIC_LINE[method],
                markersize=20,
                label=method_SC)
        else:
            plt.plot(
                arr_points[:, 0], arr_points[:, 1],
                'p-',
                markersize=20,
                label=method_SC)

        for i in range(arr_points.shape[0]):
            plt.annotate(
                list_label_points[i],
                (arr_points[i, 0], arr_points[i, 1]),
                fontsize=25)

    plt.ylabel('precision', fontsize=40)
    plt.xlabel('recall', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.legend(loc='upper right', fontsize=40)
    plt.savefig(f'images/{name_id}_{type_model_feature}_precision_recall_comparison.png')
    plt.show()


###
# Plots for the FID score
###


def plot_FID_results(
        url_folder: str,
        type_model_feature='Inception',
        ODE_plot=False,
        FID_max=None):
    '''
    In the url_folder, there should be two json files
        dic_fid.json and dic_time.json
    that contains the necessary data to generate time plots.
    Each contains key of the form
        <method>_<num_diffusion>.json

    Inputs:
    -------
        ODE_plot (bool): If True, then we have a different
                syntax for the methods.

        FID_max (float): if not None, this corresponds 

    '''
    # Check inputs
    if type_model_feature not in {'VGG', 'Inception'}:
        raise ValueError('type_model_feature is not known')

    if not os.path.exists(url_folder):
        raise ValueError(f'The folder {url_folder} does not exist.')

    if not (os.path.exists(f'{url_folder}/dic_time.json')):
        raise ValueError('The time dictionary does not exists')

    if not (os.path.exists(f'{url_folder}/dic_fid_{type_model_feature}.json')):
        raise ValueError('The FID dictionary does not exists')

    # Opening JSON file
    f = open(f'{url_folder}/dic_fid_{type_model_feature}.json')
    dic_fid = json.load(f)

    f = open(f'{url_folder}/dic_time.json')
    dic_time = json.load(f)

    dic_keys = [key for key in dic_fid.keys()]
    if set([key for key in dic_time.keys()]) != set(dic_keys):
        raise ValueError('Incompatibility in the keys of the time dict and others')

    if not ODE_plot:
        l_method = list(set([key.split('_')[0] for key in dic_keys]))
    else:
        # In the case of ODE, the name is of the format
        # "dpmsolver_adaptive_logSNR_order_3_diffsteps_10"
        l_method = list(set([key.split('_diffsteps_')[0] for key in dic_keys]))

    plt.figure(figsize=(20, 20))
    plt.title('FID VS time', fontsize=40)
    for method in l_method:

        # Get the data
        list_points = []
        list_label_points = []
        for key in dic_keys:
            # Check is the key is good
            if not ODE_plot:
                if key.split('_')[0] != method:
                    continue

            list_points.append((dic_time[key], dic_fid[key]))
            if not ODE_plot:
                list_label_points.append(key.split('_')[1])
            else:
                list_label_points.append(key.split('_diffsteps_')[1])
        # Plot them
        arr_points = np.array(list_points)

        if FID_max is not None:
            ax = plt.gca()
            ax.set_ylim([0, FID_max])

        if ODE_plot:
            # file of format
            # <naem_id>_dpmsolver_adaptive_logSNR_order_2_diffsteps_40_nbrsample_100
            # Need shortcut (SC)

            name_method = method.split('dpmsolver')[1].split('_')[1]
            name_skiptime = method.split('dpmsolver')[1].split('_')[2]
            name_order = method.split('order_')[1].split('_')[0]

            if name_method not in [key for key in ODE_METHODS_SC.keys()]:
                raise ValueError(f'Issue in the formatting of {name_method}')

            if name_skiptime not in [key for key in ODE_SKIP_SC.keys()]:
                raise ValueError(f'Issue in the formatting of {name_skiptime}')

            method_SC = (f'{ODE_METHODS_SC[name_method]}-'
                            f'{ODE_SKIP_SC[name_skiptime]}-{name_order}')
        else:
            method_SC = method

        if not ODE_plot:
            plt.plot(
                arr_points[:, 0], arr_points[:, 1],
                DIC_LINE[method],
                markersize=20,
                label=method_SC)
        else:
            plt.plot(
                arr_points[:, 0], arr_points[:, 1],
                'p-',
                markersize=20,
                label=method_SC)

        for i in range(arr_points.shape[0]):
            plt.annotate(
                list_label_points[i],
                (arr_points[i, 0], arr_points[i, 1]),
                fontsize=25)

    plt.ylabel('FID', fontsize=40)
    plt.xlabel('time (s)', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.legend(loc='upper right', fontsize=40)
    plt.savefig(f'images/fid_{type_model_feature}_time_comparison.png')
    plt.show()

    return


def plot_images_mosaic(
        url_folder: str,
        label: int = None,
        ODE_plot=False):
    '''
    url_folder should contain folder with name like
    <name_method>_numdiff_<num of diff steps>_numsample_<num of generated samples>
    when ODE_plot is false.

    Inputs:
    -------
        url_folder (str): url of the folder containing the many samples resulting 
            from different experiments.

        label (int): If None, then we do not expect that images to be stored in the
            format 'label_{idx}.png'.

        ODE_plot (bool): different formatting if it is an experiment for evaluating
            ODE sampling or not.
    '''

    if not os.path.exists(url_folder):
        raise ValueError('The folder containing the various samples folder does not exist. ')

    list_dir = [url for url in os.listdir(url_folder) if os.path.isdir(f"{url_folder}/{url}")]

    # Extract list methods
    if not ODE_plot:
        list_methods = list(set([url.split('_')[0] for url in list_dir]))
        l_num_diff = list(set([url.split('_')[2] for url in list_dir]))
    else:
        # In the ODE case the files have the following format:
        # MNIST_unconditional_dpmsolver_adaptive_logSNR_order_3_diffsteps_40_nbrsample_500
        list_methods = list(set([url.split('_diffsteps_')[0] for url in list_dir]))
        list_methods.sort()
        l_num_diff = list(set([url.split('_diffsteps_')[1].split('_')[0] for url in list_dir]))

    # Order l_num_diff
    l_num_diff = np.array([int(num) for num in l_num_diff])
    l_num_diff = list(np.sort(l_num_diff))

    num_samples = list_dir[0].split('_')[-1]
    if not np.all(np.array([url.split('_')[-1] for url in list_dir]) == num_samples):
        raise ValueError('The folder contain different numbers of samples.')

    num_methods, num_diff_steps = len(list_methods), len(l_num_diff)
    _, axs = plt.subplots(
        num_methods,
        num_diff_steps,
        figsize=(10*num_diff_steps, 10*num_methods),
        squeeze=False)

    for i, method in enumerate(list_methods):
        for j, num_steps in enumerate(l_num_diff):

            # Get the folder url
            if not ODE_plot:
                url_sample = f'{url_folder}/{method}_numdiff_{num_steps}_numsample_{num_samples}/'
            else:
                url_sample = f'{url_folder}/{method}_diffsteps_{num_steps}_nbrsample_{num_samples}/'
                # url_sample = f'{url_folder}/{list_dir[i]}/'

            # Take the first image in that folder with the correct label
            if label is not None:
                list_img_url = [url for url in os.listdir(url_sample)
                                if (url.endswith('.png') and url.startswith(f'{label}_'))]
            else:
                list_img_url = [url for url in os.listdir(url_sample)
                    if url.endswith('.png')]

            if len(list_img_url) == 0:
                raise ValueError('There are no image with the')

            img_url = f'{url_sample}/{list_img_url[0]}'

            img = np.asarray(Image.open(img_url))

            axs[i, j].imshow(img)

            if ODE_plot:
                # file of format
                # MNIST_unconditional_dpmsolver_adaptive_logSNR_order_2_diffsteps_40_nbrsample_100
                # Need shortcut (SC)

                name_method = method.split('dpmsolver')[1].split('_')[1]
                name_skiptime = method.split('dpmsolver')[1].split('_')[2]
                name_order = method.split('order_')[1].split('_')[0]

                if name_method not in [key for key in ODE_METHODS_SC.keys()]:
                    raise ValueError(f'Issue in the formatting of {name_method}')

                if name_skiptime not in [key for key in ODE_SKIP_SC.keys()]:
                    raise ValueError(f'Issue in the formatting of {name_skiptime}')

                method_SC = (f'{ODE_METHODS_SC[name_method]}-'
                             f'{ODE_SKIP_SC[name_skiptime]}-{name_order}')
            else:
                method_SC = method
            if i == 0:
                axs[i, j].set_title(f'{num_steps}', fontsize=45)
            if j == 0:
                axs[i, j].set_ylabel(f'{method_SC}', fontsize=45)

            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.savefig('images/samples_at_diff_steps.png')
    plt.show()

    return