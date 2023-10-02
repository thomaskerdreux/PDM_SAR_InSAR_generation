import os

import torch.nn as nn
from torchvision.models import (
    inception_v3,
    Inception_V3_Weights,
    vgg16,
    VGG16_Weights)

from feature_extraction.MNIST_classifier_Inception import (
    fine_tune_classifier_inception_for_MNIST)
from feature_extraction.MNIST_classifier_VGG import (
    load_best_model_vgg_mnist,
    fine_tune_vgg_for_MNIST)
from feature_extraction.SAR_classifier_Inception import (
    fine_tune_classifier_inception_for_SAR)
from feature_extraction.SAR_classifier_VGG import (
    fine_tune_vgg_for_SAR)

from improved_diffusion.datasets_image import load_data


if __name__ == '__main__':

    # Choose the type of fine-tuning to perform
    type_feature_extractor = 'VGG'
    type_dataset = 'SAR'

    if type_feature_extractor not in ['Inception', 'VGG']:
        raise ValueError('Only Inception and VGG are valid feature extractors.')

    if type_dataset not in ['MNIST', 'SAR']:
        raise ValueError('Only two types of datasets are valid.')

    # Get the dataloaders for your dataset
    if type_dataset == 'MNIST':
        # Image parameters
        num_channels = 1
        image_size = 32

        # Training batch size
        batch_size = 28
        nbr_classes = 10

        # Path to training/validation data
        data_dir_train = 'data/mnist_train/'
        data_dir_val = 'data/mnist_val/'

        if not os.path.exists(data_dir_train):
            raise ValueError('You need to get the MNIST dataset (see ReadMe).')

        loader_train = load_data(
            data_dir=data_dir_train,
            batch_size=batch_size,
            image_size=image_size,
            num_channels=num_channels,
            class_cond=True,
            num_class=10,
            crop=False,
            droplast=False)

        loader_val = load_data(
            data_dir=data_dir_val,
            batch_size=batch_size,
            image_size=image_size,
            num_channels=num_channels,
            class_cond=True,
            num_class=10,
            crop=False,
            droplast=False)

    if type_dataset == 'SAR':

        # Image number of channels
        image_size = 128
        num_channels = 1

        # Trainign batch size
        batch_size = 64
        nbr_classes = 10

        # Path to training/validation data
        data_dir_train = 'data/sar_tengeop/train/'
        data_dir_val = 'data/sar_tengeop/val/'

        if not os.path.exists(data_dir_train):
            raise ValueError('You need to get the SAR dataset or put the right url.')

        loader_train = load_data(
            data_dir=data_dir_train,
            batch_size=batch_size,
            image_size=image_size,
            num_channels=num_channels,
            class_cond=True,
            num_class=10,
            crop=True,
            droplast=False)

        loader_val = load_data(
            data_dir=data_dir_val,
            batch_size=batch_size,
            image_size=image_size,
            num_channels=num_channels,
            class_cond=True,
            num_class=10,
            crop=True,
            droplast=False)

    if type_feature_extractor == 'VGG':
        # Define the model
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Change the number of output classes
        IN_FEATURES = model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, nbr_classes)
        model.classifier[-1] = final_fc

        if image_size < 32 or image_size > 224:
            raise ValueError('VGG expects an input image size between 32 and 224.')

        url_folder = f'./models_data/{type_dataset}_classifier_vgg_128/'

        if not os.path.exists(url_folder):
            os.makedirs(url_folder, exist_ok=True)

        l_models = [url for url in os.listdir(url_folder) if url.endswith('.tar')]

        if len(l_models) > 0:
            model = load_best_model_vgg_mnist(
                loader_val,
                url_folder)

        # Fine-tune the model
        start_lr = 1e-3
        url_folder_experiment = f'./models_data/{type_dataset}_classifier_vgg_{image_size}/'

        if type_dataset == 'MNIST':
            fine_tune_vgg_for_MNIST(
                model,
                loader_train,
                loader_val,
                url_folder_experiment,
                num_batch_plot_train=5,
                num_epoch=100,
                start_lr=start_lr)
        elif type_dataset == 'SAR':
            fine_tune_vgg_for_SAR(
                model,
                loader_train,
                loader_val,
                url_folder_experiment,
                nbr_batch_plot_train=5,
                nbr_epoch=100,
                start_lr=start_lr)

    if type_feature_extractor == 'Inception':
        # Prepare the model for fine-tuning.
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False

        for parameter in model.parameters():
            parameter.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, nbr_classes)
        )

        # Fine-tune the model
        start_lr = 1e-3
        url_folder_experiment = f'./models_data/{type_dataset}_classifier_Inception_{image_size}/'

        if type_dataset == 'MNIST':
            fine_tune_classifier_inception_for_MNIST(
                model,
                loader_train,
                loader_val,
                url_folder_experiment,
                num_batch_plot_train=5,
                num_epoch=100,
                start_lr=start_lr,
                transform=True,
                resize=True)
        elif type_dataset == 'SAR':
            fine_tune_classifier_inception_for_SAR(
                model,
                loader_train,
                loader_val,
                url_folder_experiment,
                num_batch_plot_train=5,
                num_epoch=100,
                start_lr=start_lr,
                transform=True,
                resize=True)