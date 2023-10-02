import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16, VGG16_Weights

from improved_diffusion.CONSTANTS import DEVICE_ID


"""
https://www.kaggle.com/code/darthmanav/explaining-vgg-model-fine-tuning-pca-t-sne
"""


def calculate_accuracy(
        y_pred: torch.Tensor,
        y: torch.Tensor):
    '''
    Compute the accuracy of the model.
    '''
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float() / y.shape[0]


def validation_step(
        model: nn.Module,
        loader_val: torch.utils.data.DataLoader,
        loss_fn
        ) -> Tuple[float, float]:
    """
    Compute loss and accuracy on validation dataset.
    """
    model.eval()

    val_loss = 0
    val_acc = 0

    with torch.no_grad():

        for idx_batch, (data, label_dic) in enumerate(loader_val):

            data = data.to(DEVICE_ID)
            data = torch.repeat_interleave(data, 3, dim=1)

            y_pred = model(data)
            labels = label_dic['y'].to(DEVICE_ID)

            val_loss += loss_fn(y_pred, labels)
            val_acc += calculate_accuracy(y_pred, labels)

        # Computing the validation loss on an epoch
        avg_val_loss_epoch = val_loss/(idx_batch + 1)
        avg_val_acc_epoch = val_acc/(idx_batch + 1)

    return avg_val_loss_epoch, avg_val_acc_epoch


def fine_tune_vgg_for_MNIST(
        model: nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_val: torch.utils.data.DataLoader,
        url_folder_experiment: str,
        start_lr=1e-3,
        num_batch_plot_train=1,
        num_epoch=100,
        factor=0.5,
        patience=5) -> None:
    '''
    First we freeze the feature extractor (i.e., model.features) 
    from VGG and only train the classifier (i.e., model.classifier)
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

    for parameter in model.features.parameters():
        parameter.requires_grad = False

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
            loss_fn)

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
                       f'{url_folder_experiment}/MNIST_classifier_vgg16_epoch={idx_epoch}_lossval={np.round(avg_val_loss_epoch.item(), 2)}_accuracy={np.round(avg_val_accuracy_epoch.item(), 2)}.tar')


def load_best_model_vgg_mnist(
        dataloader_val: torch.utils.data.DataLoader,
        url_folder: str):
    '''
    '''

    if not os.path.exists(url_folder):
        raise ValueError(f'The folder url {url_folder} does not exist.')

    list_model = [url for url in os.listdir(url_folder) if url.endswith('.tar')]

    l_accuracy = []
    for url in list_model:
        model_url = f'{url_folder}/{url}'

        model = vgg16(VGG16_Weights.IMAGENET1K_V1)

        # Change the number of output classes
        IN_FEATURES = model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, 10)
        model.classifier[-1] = final_fc

        model.load_state_dict(torch.load(model_url, map_location=DEVICE_ID))
        model.to(DEVICE_ID)

        loss_fn = nn.CrossEntropyLoss()

        _, accuracy = validation_step(
            model,
            dataloader_val,
            loss_fn
            )

        l_accuracy.append(accuracy.item())

    model = vgg16(VGG16_Weights.IMAGENET1K_V1)

    # Change the number of output classes
    IN_FEATURES = model.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, 10)
    model.classifier[-1] = final_fc

    model.load_state_dict(torch.load(model_url, map_location=DEVICE_ID))
    model.to(DEVICE_ID)

    return model


def load_best_model_url_vgg_mnist(
        url_folder: str) -> str:
    '''
    url_folder should contains .tar files on the following format
        MNIST_classifier_vgg16_epoch=4_lossval=0.06_accuracy=0.99.tar
    '''

    list_models = [url for url in os.listdir(url_folder) if url.endswith('.tar')]

    if len(list_models) == 0:
        raise ValueError('There are no .tar file here ..')

    list_acc = [float(url.split('_accuracy=')[-1].split('.tar')[0]) for url in list_models]

    idx_best = np.argmax(list_acc)

    url_best_model = f'{url_folder}/{list_models[idx_best]}'

    return url_best_model