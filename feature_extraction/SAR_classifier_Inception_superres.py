"""
Train a classifier on low-resolution (e.g., 128x128) SAR images
from the TenGeoP dataset, downscaled from high-resolution
(e.g., 256x256) images.
"""
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms

from improved_diffusion.datasets_image_lowres import load_data_lowres
from improved_diffusion.CONSTANTS import DEVICE_ID


def calculate_accuracy(
        y_pred: torch.Tensor,
        y: torch.Tensor
        ) -> float:
    '''
    Compute a model's accuracy with predicted (y_pred)
    and true (y) data labels.
    '''
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float() / y.shape[0]


def best_model_url(
        url_folder: str
        ) -> str:
    '''
    Finds the url of the best-performing model in url_folder, which
    contains a list of .tar files for each model with the following name:
        SAR_classifier_inception_epoch=0_lossval=0.91_accuracy=0.69.tar
    '''
    # Check input
    if not os.path.exists(url_folder):
        raise ValueError('The experiment folder does not exist.')

    list_tar = [url for url in os.listdir(url_folder) if url.endswith('.tar')]

    if len(list_tar) == 0:
        raise ValueError(f'There is no .tar file in {url_folder}')

    list_acc = [float(url.split('accuracy=')[-1].split('.t')[0])
                for url in list_tar]
    idx = np.argmax(list_acc)
    best_url = f'{url_folder}/{list_tar[idx]}'

    return best_url


def validation_step(
        model: nn.Module,
        loader_val: DataLoader,
        loss_fn,
        transform: bool = False,
        resize: bool = True
        ) -> Tuple[float, float]:
    """
    Compute loss and accuracy on validation dataset.

    Inputs:
    ------
        model (nn.Module): instance of classifier model

        loader_val (DataLoader): dataloader for the validation data

        loss_fn: loss function for model training

        transform (bool): if True, then the data is rescaled to
            [0, 1] and then normalised with transforms.Normalize()

        resize (bool): if True, resize the inputs to (3, 299, 299)
    """
    model.eval()

    val_loss = 0
    val_acc = 0

    with torch.no_grad():

        for idx_batch, (data, label_dic) in enumerate(loader_val):

            data = data.to(DEVICE_ID)
            # Artificially repeat the channels for Inception input.
            data = torch.repeat_interleave(data, 3, dim=1)

            if resize:
                data = F.interpolate(
                    data,
                    size=(299, 299),
                    mode='bilinear',
                    align_corners=False)

            if transform:
                # First set the SAR data into [0, 1] when it is
                # originally in [-1, 1]
                data = data/2 + 0.5
                if torch.min(data) < 0 or torch.min(data) > 1:
                    raise ValueError('The SAR data should now be in [0, 1].')

                # The normalization for data that is in [0, 1]
                train_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])

                data = train_normalization(data)

            y_pred = model(data)
            labels = label_dic['y'].to(DEVICE_ID)

            val_loss += loss_fn(y_pred, labels)
            val_acc += calculate_accuracy(y_pred, labels)

        # Computing the validation loss on an epoch
        avg_val_loss_epoch = val_loss / (idx_batch + 1)
        avg_val_acc_epoch = val_acc / (idx_batch + 1)

    return avg_val_loss_epoch, avg_val_acc_epoch


def train_classifier_inception_for_SAR(
        model: nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_val: torch.utils.data.DataLoader,
        url_folder_experiment: str,
        start_lr: float = 1e-3,
        num_batch_plot_train: int = 1,
        num_epoch: int = 100,
        factor: float = 0.5,
        patience: int = 5,
        transform: bool = False,
        resize: bool = True
        ) -> None:
    '''
    During the training, the data have to be resized to (299, 299).

    Inputs:
    ------
        model (nn.Module): instance of classifier model

        loader_train (DataLoader): dataloader for the training data

        loader_val (DataLoader): dataloader for the validation data

        url_folder_experiment (str): model folder where model files
            and tensorboard data will be written

        start_lr (float): initial learning rate

        num_batch_plot_train (int): step at which to print training
            error / accuracy and save to tensorboard

        num_epoch (int): number of epochs to run

        factor (float): factor by which to decrease the learning rate
            (see torch.optim.lr_scheduler.ReduceLROnPlateau)

        patience (int): number of epochs after which to decrease the
            learning rate if the model training error did not improve
            (see torch.optim.lr_scheduler.ReduceLROnPlateau)

        transform (bool): if True, then the data is rescaled to
            [0, 1] and then normalised with transforms.Normalize()

        resize (bool): if True, resize the inputs to (3, 299, 299)
    '''
    if not os.path.exists(url_folder_experiment):
        os.makedirs(url_folder_experiment, exist_ok=True)

    writer = SummaryWriter(url_folder_experiment)

    optimizer = optim.Adam(model.parameters(), lr=start_lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=factor,
        patience=patience,
        threshold=1e-3,
        min_lr=1e-9,
        verbose=True)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(DEVICE_ID)

    model.train()
    model.to(DEVICE_ID)

    counter_batch = 0

    for idx_epoch in range(num_epoch):

        running_loss = 0.
        running_avg_loss = 0.0
        running_acc = 0.
        running_avg_accuracy = 0.

        for idx_batch, (data, label_dic) in enumerate(loader_train):

            optimizer.zero_grad()

            data = data.to(DEVICE_ID)
            data = torch.repeat_interleave(data, 3, dim=1)

            if transform:
                # First set the SAR data into [0, 1] when it is
                # originally in [-1, 1]
                data = data/2 + 0.5
                if torch.min(data) < 0 or torch.min(data) > 1:
                    raise ValueError('The SAR data should now be in [0, 1].')

                # The normalization for data that is in [0, 1]
                train_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])

                data = train_normalization(data)

            if resize:
                data = F.interpolate(
                    data,
                    size=(299, 299),
                    mode='bilinear',
                    align_corners=False)

            y_pred = model(data)

            # prepare the labels
            labels = label_dic['y'].to(DEVICE_ID)

            loss = loss_fn(y_pred, labels)

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

            acc = calculate_accuracy(y_pred, labels)

            # Gather data and report
            running_loss += loss.item()
            running_avg_loss += loss.item()
            running_acc += acc.item()
            running_avg_accuracy += acc.item()

            if idx_batch % num_batch_plot_train == num_batch_plot_train-1:
                print(f'Epoch {idx_epoch} batch {idx_batch + 1} train loss: {running_loss / num_batch_plot_train } and accuracy: {running_acc / num_batch_plot_train } ')
                writer.add_scalar('Loss/train',
                                  running_loss / num_batch_plot_train,
                                  counter_batch)
                running_loss = 0.

                writer.add_scalar('Accuracy/train',
                                  running_acc / num_batch_plot_train,
                                  counter_batch)
                running_acc = 0.

            counter_batch += 1

        avg_train_loss_epoch = running_avg_loss / (idx_batch + 1)
        avg_train_accuracy_epoch = running_avg_accuracy / (idx_batch + 1)

        # Compute val loss and val accuracy.
        avg_val_loss_epoch, avg_val_accuracy_epoch = validation_step(
            model,
            loader_val,
            loss_fn,
            transform=transform,
            resize=resize)

        print(f"Accuracy on the validation set is {avg_val_accuracy_epoch}")

        # Decrease learning rate according to the validation loss on a epoch
        scheduler.step(avg_val_loss_epoch)

        # Update tensorboard writer
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_train_loss_epoch,
                            'Validation': avg_val_loss_epoch},
                           idx_epoch)
        writer.add_scalars('Training vs. Validation Accuracy',
                           {'Training': avg_train_accuracy_epoch,
                            'Validation': avg_val_accuracy_epoch},
                           idx_epoch)

        writer.flush()  # Write to disk immediately

        # Save model
        if idx_epoch == 0:
            best_score = avg_val_loss_epoch * 1.01

        if avg_val_loss_epoch.item() < best_score:
            best_score = avg_val_loss_epoch.item()
            print('Saving model...')
            torch.save(model.state_dict(),
                       f'{url_folder_experiment}/SAR_classifier_inception_epoch={idx_epoch}_lossval={np.round(avg_val_loss_epoch.item(), 2)}_accuracy={np.round(avg_val_accuracy_epoch.item(), 2)}.tar')


def fine_tune_classifier(
        url_folder_model: str,
        data_dir_train: str,
        data_dir_val: str,
        image_size: int = 128,
        image_size_hr: int = 256,
        num_channels: int = 1,
        num_class: int = 10,
        batch_size: int = 64,
        num_batch_plot_train: int = 20,
        start_lr: bool = 1e-3,
        num_epoch: int = 100,
        factor: float = 0.5,
        patience: int = 5,
        transform: bool = True
        ) -> None:
    """
    Fine-tunes the Inception model in two steps: first, by optimising
    only the classifier (model.fc)

    Inputs:
    ------
        url_folder_model (str): path to folder where to save model data

        data_dir_train (str): path to the training data

        data_dir_val (str): path to the validation data

        image_size (int): image size to downscale to

        image_size_hr (int): high-resolution image size to downscale from

        num_channels (int): number of data channels

        num_class (int): number of data classes

        batch_size (int): batch size for model training

        num_batch_plot_train (int): step at which to print training
            error / accuracy and save to tensorboard

        start_lr (float): initial learning rate

        num_epoch (int): number of epochs to run

        factor (float): factor by which to decrease the learning rate
            (see torch.optim.lr_scheduler.ReduceLROnPlateau)

        patience (int): number of epochs after which to decrease the
            learning rate if the model training error did not improve
            (see torch.optim.lr_scheduler.ReduceLROnPlateau)

        transform (bool): if True, then the data is rescaled to
            [0, 1] and then normalised with transforms.Normalize()
    """
    # Define dataloaders
    loader_train = load_data_lowres(
        data_dir=data_dir_train,
        batch_size=batch_size,
        image_size_lr=image_size,
        image_size_hr=image_size_hr,
        num_channels=num_channels,
        class_cond=True,
        num_class=num_class,
        crop=True,
        droplast=False)

    loader_val = load_data_lowres(
        data_dir=data_dir_val,
        batch_size=batch_size,
        image_size_lr=image_size,
        image_size_hr=image_size_hr,
        num_channels=num_channels,
        class_cond=True,
        num_class=num_class,
        crop=True,
        droplast=False)

    # Define the model
    # From https://github.com/Harry24k/Pytorch-Basic/blob/master/Week5/20.%20Transfer%20Learning%20with%20Inception%20v3.ipynb
    model = models.inception_v3(weights=True)
    model.aux_logits = False
    # First we freeze the feature extractor and only train the classifier
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_class)
    )

    if transform:
        url_folder_experiment = f'./models_data/SAR_classifier_inception_2step_{image_size}_lr_{start_lr}_transform/'
    else:
        url_folder_experiment = f'./models_data/SAR_classifier_inception_2step_{image_size}_lr_{start_lr}/'

    print('Training the classifier...')
    train_classifier_inception_for_SAR(
        model,
        loader_train,
        loader_val,
        url_folder_experiment,
        start_lr=start_lr,
        num_batch_plot_train=num_batch_plot_train,
        num_epoch=num_epoch,
        factor=factor,
        patience=patience,
        transform=transform,
        resize=True)
    print('Done with the classifier training.')

    # Then we optimise all models weights.
    model = models.inception_v3(weights=True)
    model.aux_logits = False
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 10)
    )
    model.train()

    # Load the fine-tuned classifier.
    url_best_model = best_model_url(
        url_folder_experiment)
    model.load_state_dict(torch.load(url_best_model, map_location=DEVICE_ID))

    if transform:
        url_folder_experiment_new = f'{url_folder_model}/SAR_all_layers_classifier_inception_2step_128_lr_{start_lr}_transform/'
    else:
        url_folder_experiment_new = f'{url_folder_model}/SAR_all_layers_classifier_inception_2step_128_lr_{start_lr}/'

    print('Now training all model layers...')
    train_classifier_inception_for_SAR(
        model,
        loader_train,
        loader_val,
        url_folder_experiment_new,
        start_lr=start_lr,
        num_batch_plot_train=num_batch_plot_train,
        num_epoch=num_epoch,
        factor=factor,
        patience=patience,
        transform=transform,
        resize=True)
    print('Done.')
