# Radar data generation with Probabilistic Diffusion Models (PDMs)
![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg?style=plastic)
![Pytorch 2.0](https://img.shields.io/badge/pytorch-2.0-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

This repository is the codebase for [Diffusion Models for Interferometric Satellite Aperture Radar](https://arxiv.org/abs/2308.16847), by Tuel, Kerdreux et al..

The code was adapted from [https://github.com/openai/improved-diffusion](https://github.com/openai/improved-diffusion) to provide a simple, plug-and-play option to apply and test Probabilistic Diffusion Models (PDMs) for image generation specifically in the context of Radar Satellite imagery (SAR or InSAR).

## Creators and Maintainers

This code has been created by [Thomas Kerdreux](mailto:thomaskerdreux@gmail.com) and [Alexandre Tuel](mailto:alexandret@geolabe.com) at Geolabe LLC.

## Usage

This directory provides a simple and versatile code to apply Probabilistic Diffusion Models (PDMs) to the challenge of satellite image generation. It is coded in [https://pytorch.org/](PyTorch).

We provide several Synthetic Aperture Radar (SAR)-based datasets and the corresponding trained PDMs for use in this codebase (in this [Zenodo repository](https://doi.org/10.5281/zenodo.8222778)).
You can also use the code to train your own model on different datasets.

What you need to do to run the code smoothly:

 * Install the required packages
 * Use a pre-existing dataset or bring your own dataset following the required formats (see `Datasets` section)
 * Train a model with [example_train.py](example_train.py)
 * Generate images with [example_sample.py](example_sample.py)

For superresolution models, you need to first train a low-resolution model with [example_train_lowres.py](example_train_lowres.py) and then a superresolution model with [example_train_superres.py](example_train_superres.py). See also paragraph `Available trained models` to download pretrained low-resolution and superresolution models for SAR. 
Then use [example_sample.py](example_sample.py) to sample from the low-resolution model and save samples as `.npz` files, and then [example_sample_superres.py](example_sample_superres.py) to sample from the superresolution model, conditioned on the low-resolution samples.

## Installation

You can install the required environment for this codebase via conda:
```
conda env create -f environment.yml
conda activate PDM_SAR_InSAR_generation
```

The code was independently tested on a single NVIDIA GeForce RTX 3090 and a single NVIDIA GeForce RTX 4090.

## Datasets

See the [dataset description](./datasets/README.md) for more information and instructions on how to download some of the datasets. We recommend to first download all the datasets described in this README.

The code can currently handle two types of data: image files (readable by `PIL.Image`, like *.png, *.jpg, etc.) and HDF5 files (readable by `h5py`).

If you are not familiar with HDF5 files and how to process them in Python, you can take a look at [this website](https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/) (last accessed August 7, 2023) and the [h5py python package documentation](https://docs.h5py.org/en/stable/).


## Available trained models

We provide 6 trained models at the following [Zenodo link](https://doi.org/10.5281/zenodo.8222778). Each `.zip` file contains the model weights (in `*.pt` format) and the model metadata file (in `*.json` format).

 * `mnist_32_cond_sigma_100.zip`: a class-conditional PDM model trained (the variance is learnt) with 100 diffusion time steps on 32x32 MNIST images;
 ```
 wget https://zenodo.org/record/8222779/files/mnist_32_cond_sigma_100.zip?download=1 -O models_data/mnist_32_cond_sigma_100.zip
 unzip models_data/mnist_32_cond_sigma_100.zip -d models_data/
 ```
 
 * `mnist_32_no_cond_sigma_100.zip`: an unconditional PDM model trained (the variance is learnt) with 100 diffusion time steps on 32x32 MNIST images;
 ```
 wget https://zenodo.org/record/8222779/files/mnist_32_no_cond_sigma_100.zip?download=1 -O models_data/mnist_32_no_cond_sigma_100.zip
 unzip models_data/mnist_32_no_cond_sigma_100.zip -d models_data/
 ```

 * `SAR_lowres_128_cond_sigma_2000.zip`: a low-resolution (256 to 128) PDM model trained with 2000 diffusion time steps on 128x128 TenGeoP-SARwv images; Download zenodo link to the models_data/ folder and unzip there.

```
wget https://zenodo.org/record/8222779/files/SAR_lowres_128_cond_sigma_2000.zip?download=1 -O models_data/SAR_lowres_128_cond_sigma_2000.zip
unzip models_data/SAR_lowres_128_cond_sigma_2000.zip -d models_data/
python example_sample.py
```

 * `SAR_superres_128_to_256_cond_sigma_2000.zip`: a super-resolution (128 to 256) PDM model trained with 2000 diffusion time steps on TenGeoP-SARwv images; Download zenodo link to the models_data/ folder and unzip there. Download zenodo link to the models_data/ folder and unzip there.

 ```
 wget https://zenodo.org/record/8222779/files/SAR_superres_128_to_256_cond_sigma_2000.zip?download=1 -O models_data/SAR_superres_128_to_256_cond_sigma_2000.zip
 unzip models_data/SAR_superres_128_to_256_cond_sigma_2000.zip -d models_data/
 ```

 * `insar_phase_128_sigma_2000.zip`: an unconditional PDM model trained with 2000 diffusion time steps on 128x128 Sentinel-1 InSAR interferometers over New Mexico (USA); Download zenodo link to the models_data/ folder and unzip there.
 ```
 wget https://zenodo.org/record/8222779/files/insar_phase_128_sigma_2000.zip?download=1 -O models_data/insar_phase_128_sigma_2000.zip
 unzip models_data/insar_phase_128_sigma_2000.zip -d models_data/
 ```

 * `insar_noise_32_sigma_1000.zip`: an unconditional PDM model trained with 1000 diffusion time steps on 32x32 Sentinel-1 InSAR ground deformation scenes over New Mexico (USA). Download zenodo link to the models_data/ folder and unzip there.
```
wget https://zenodo.org/record/8222779/files/insar_noise_32_sigma_1000.zip?download=1 -O models_data/insar_noise_32_sigma_1000.zip
unzip models_data/insar_noise_32_sigma_1000.zip
```


## Training

The script to use to train the model is [example_train.py](example_train.py). The `main()` functions takes all required parameters as input, including model hyperparameters. The full list of parameters is described in [train.py](train.py).
Hyperparameters can be divided into three groups: UNet model architecture, diffusion process, and training parameters.

Important Unet model architecture parameters include:
 * the number of channels of input images (`num_input_channels`)
 * the base channel count for the model (`num_model_channels`)
 * the number of residual blocks per downsample (`num_res_blocks`)
 * the number of attention heads in each attention layer (`num_heads`)

Important diffusion process parameters include:
 * the number of diffusion steps (`diffusion_steps`)
 * the noise schedule (`noise_schedule`)
 * a flag to learn the noise variance (`learn_sigma`)
 * a schedule sampler to sample diffusion time steps when calculating the loss (`schedule_sampler`)

Important training parameters include:
 * paths to the training and validation dataset directories (`data_train_dir` and `data_val_dir`)
 * batch (and microbatch) size (`batch_size` and `microbatch`)
 * the number of training epochs (`num_epochs`)
 * the loss function (`loss_name`)
 * a flag to train the model in class-conditional mode (`class_cond`)
 * the initial learning rate (`lr`)
 * a flag to compute a validation error at the end of each model epoch (`compute_val`) (can be very time-consuming!)

If `compute_val=True`, because the VLB/MSE loss can take a long time to compute if the input data or the number of diffusion timesteps are large,
you may prefer to restrict loss calculation to a pre-defined number of initial timesteps with the `subset_timesteps` option
(the VLB loss is indeed dominated by the terms corresponding to the first few timesteps, see [Nichol and Dhariwal (2021)](https://doi.org/10.48550/arXiv.2102.09672)).

Models are periodically saved to `.pt` files in a user-specified model folder (`url_folder_experiment`):
 * the most recent model as `model***.pt`
 * the Exponential Moving Average (EMA) model as `ema***.pt`.

Optimiser state is also saved as `opt***.pt` to resume training from model checkpoint. Models can be saved at the end of each epoch or within each epoch at a pre-defined interval of time steps (`save_interval`).

Model training can be monitored with the `tensorboard` package by calling `tensorboard --logdir url_model_folder` from a terminal and opening the resulting link in a browser.


## Sampling

The script to use to sample from a model is [example_sample.py](example_sample.py). All you need to do is to give a valid model path (`model_path`), the number of sampling diffusion steps (`diffusion_steps`) and the number of samples (`num_samples`). For class-conditional models, you can specify which class you would like to sample from (with the `sample_class` option).

It is also possible to use [DDIM](https://arxiv.org/abs/2010.02502) for faster sampling by setting `use_ddim=True`. The script [sample_DPM.py](sample_DPM.py) also allows you to apply the DPM solver of [Lu et al. (2022)](https://arxiv.org/abs/2206.00927).

Use the script [example_time_comparison_DPM_MNIST.py](example_time_comparison_DPM_MNIST.py) to compare the various parameters of the DPM sampling approach in terms of FID or Improved Precision and Recall. Note that it is very sensible to the feature extractor used to compute the various metrics.

For superresolution models, use [example_sample_superres.py](example_sample_superres.py)

## Evaluation of model performance

We also provide scripts to evaluate the performance of a model against a given dataset.

[utils_FID.py](utils_FID.py) allows to compute the classic FID, as well as the improved Precision and Recall metrics introduced by [Kynkäänniemi et al. (2019)](https://arxiv.org/pdf/1904.06991.pdf). FID is here calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to the distributions of feature representations of a pre-trained InceptionV3 classifier (or VGG) network, fine-tuned on SAR scenes from the [TenGeoP-SARwv](https://doi.org/10.17882/56796) dataset, see the script [example_fine_tuning.py](example_fine_tuning.py).

[model_likelihood.py](model_likelihood.py) calculates the Variational Lower Bound likelihood of a given PDM on a dataset.

[utils/utils_semivariogram.py](utils/utils_semivariogram.py) allows to calculate and plot semivariograms on real and generated datasets of images, saved as HDF5 files.

[example_time_comparison_DPM_MNIST.py](example_time_comparison_DPM_MNIST.py) allows comparing various metrics on an MNIST generation task when varying the parameters of the DPM sampling approach. One can adapt the scripts [sample_time_comparison.py](sample_time_comparison.py) and [sample_DPM.py](sample_DPM.py) to perform various kinds of time comparisons of sampling approaches.

## Licenses

All material except is made available under the [MIT](https://opensource.org/license/mit/) license.

The present code was adapted from the [improved-diffusion](https://github.com/openai/improved-diffusion) codebase by Nichol and Dhariwal, licensed under the MIT License.

The code to calculate FID was adapted from the [Pytorch implementation](https://github.com/mseitzer/pytorch-fid) by Seitzer et al. (2020), licensed under the Apache License 2.0.

The code to calculate the improved Precision and Recall scores was adapted from the [Pytorch implementation](https://github.com/youngjung/improved-precision-and-recall-metric-pytorch) by [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor) of the algorithm introduced by [Kynkäänniemi et al. (2019)](https://arxiv.org/pdf/1904.06991.pdf), licensed under the MIT license.

The code to perform the fast ODE sampling (i.e., the DPM solver) was adapted from the [Pytorch impplementation](https://github.com/LuChengTHU/dpm-solver/blob/main/LICENSE) of [Lu et al. (2022)](https://arxiv.org/abs/2206.00927), licensed under the MIT License.


## Acknowledgments

The funding for this work was provided through a NASA NSPIRES grant (number 80NSSC22K1282).


## Citation

When using our code, please cite our corresponding paper as follows:

```
@article{TuelKerdreux2023,
  author    = {Alexandre Tuel and Thomas Kerdreux and Claudia Hulbert and Bertrand Rouet-Leduc},
  title     = {{Diffusion Models for Interferometric Satellite Aperture Radar}},
  eprint={2308.16847},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  year      = {2023},
  doi       = {10.48550/arXiv.2308.16847},
  url       = {https://doi.org/10.48550/arXiv.2308.16847}
}
```