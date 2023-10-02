import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16

sys.path.append('../../PDM_SAR_InSAR_generation/')
from improved_diffusion.CONSTANTS import DEVICE_ID


def load_and_crop_VGG(
        url_model_path: str,
        last_layer: bool = False,
        nbr_classes=10) -> nn.Module:
    '''
    Loads the classifier model given the model path (.tar file)
    and crop it at the first layer of the classifier.

    Inputs:
    ------
        url_model_path (str): path to classifier model (.tar file)

        last_layer (bool): if True, then we output the last layer
            (i.e. of dimension 10 in the case of an MNIST).

        nbr_classes (int): number of classes in the dataset on which
            the VGG model is fine-tuned.
    '''
    if not url_model_path.endswith('.tar'):
        raise ValueError('Please indicate a .tar file as model path.')

    # Define the model
    model = vgg16()
    # Change the number of output classes
    IN_FEATURES = model.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, nbr_classes)
    model.classifier[-1] = final_fc

    # Apply model weights
    model.load_state_dict(torch.load(url_model_path))
    model = model.to(DEVICE_ID)

    # Crop model to access embedded feature space
    # Here we simply crop to access feature space of dimension 4096
    # within the classifier
    if not last_layer:
        model.classifier = model.classifier[0]

    return model


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(
            self,
            output_blocks: list = [DEFAULT_BLOCK_INDEX],
            resize_input: bool = True,
            normalise_input: bool = True,
            requires_grad: bool = False):
        '''
        Inputs:
        -------
            output_blocks (list): list of InceptionV3 output blocks

            resize_input (bool): if True, the inputs are
                resized to (3, 299, 299). Note that we
                expect input with three channels.

            normalise_input (bool): if True, it normalises
                the input from a (0, 1) range to range (-1, 1).

            requires_grad (bool): if False, freezes model parameters
        '''

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalise_input = normalise_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = torchvision.models.inception_v3(weights=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(
            self,
            inp: torch.Tensor
            ) -> torch.Tensor:
        """
        Get Inception feature maps

        Input:
        -------
            inp : torch.autograd.Variable
                Input tensor of shape Bx3xHxW. Values are expected to be in
                range (0, 1) (if self.resize_input is True).

        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalise_input:
            if torch.min(inp) < 0 or torch.max(inp) > 1:
                raise ValueError('The data is not in the range (0, 1).')
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        # We only need the last feature representation
        if len(outp) != 1:
            raise ValueError('We expect a list of length 1 here.')
        return outp[0]


def load_and_crop_Inception(
        url_model_feature,
        layer_size=2048,
        in_01=True):
    '''

    Inputs:
    -------
        url_model_feature (str): url of the .tar arxiv of the
            fine-tuned Inception model.

        layer_size (int): Size at which to cut in the Inception
            model to obtain a feature representation of the data.

        in_01 (bool): If True, it expects the data to be already in
            [0,1] and otherwise, the normalization layer is embedded
            in the .
    '''
    if layer_size not in [64, 192, 768, 2048]:
        raise ValueError(f'Layer size of {layer_size} do not work for Inception.')

    if not url_model_feature.endswith('.tar'):
        raise ValueError('Please indicate a .tar file as model path.')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[layer_size]
    model_for_extraction = InceptionV3(
        [block_idx],
        normalise_input=in_01,
        )
    model_for_extraction = model_for_extraction.to(DEVICE_ID)

    return model_for_extraction