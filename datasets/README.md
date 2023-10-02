# Downloading datasets

This directory includes instructions and scripts for downloading and processing the datasets for use in this code base.

## TenGeoP-SARwv

The TenGeoP-SARwv dataset ([Chen et al. 2018](https://doi.org/10.1002/gdj3.73)) consists of more than 37,000 Sentinel-1A WV SAR vignettes divided into [ten defined geophysical categories](https://www.seanoe.org/data/00456/56796/data/58682.txt), including both oceanic and meteorologic features. To download it, go to [this page](https://doi.org/10.17882/56796), download the ["High quality images for machine learning approach"](https://www.seanoe.org/data/00456/56796/data/58684.tar.gz) file and unzip it. Use [datasets/TenGeoP_SAR.py](datasets/TenGeoP_SAR.py) to convert the GeoTiffs to image (.png) format, and to split between training, test and validation data. In `.png` format, the data will take about 7 GB of space.

To download the TenGeoP dataset, type the following in your terminal shell:
```
wget https://www.seanoe.org/data/00456/56796/data/58684.tar.gz -P data/
```

Then unzip the folder `data/58684.tar.gz` in the `data/` folder
```
pv data/58684.tar.gz | tar -xvzf - -C data/
```

Then run the script `TenGeoP_SAR.py` to split the dataset in train/val/test and format them appropriately
```
python datasets/TenGeoP_SAR.py
```

## InSAR datasets

We also provide at this [Zenodo link](https://doi.org/10.5281/zenodo.8222778) two InSAR-based datasets:

 * `InSAR_noise_32x32.zip`: a dataset of 32x32 ground deformation scenes obtained. Images were normalised to [0, 1]. To download the dataset of insar noisy deformations:
```
wget https://zenodo.org/record/8222779/files/InSAR_noise_32x32.zip?download=1 -O data/InSAR_noise_32x32.zip
unzip data/InSAR_noise_32x32.zip -d data/
mv data/InSAR_noise_32x32 data/insar_noise/
```

 * `insar_unwrapped_phase_normalised.zip`: a dataset of 128x128 InSAR interferometers obtained from Sentinel-1 acquisitions over Nex Mexico. Images were normalised to [0, 1]. To download the dataset of insar interferos:
```
wget https://zenodo.org/record/8222779/files/insar_unwrapped_phase_normalised.zip?download=1 -O data/insar_unwrapped_phase_normalised.zip
unzip data/insar_unwrapped_phase_normalised.zip -d data/
mv data/medium_0_1/ data/insar_unwrap/
```



## MNIST and CIFAR-10

The MNIST and CIFAR-10 datasets can additionally be downloaded by using the scripts [mnist.py](mnist.py) and [cifar10.py](cifar10.py) inherited from the [improved-diffusion](https://github.com/openai/improved-diffusion) repository. Both create training, validation and test directories containing `.png` files e.g., `5_00182.png` for MNIST and `truck_49997.png` for CIFAR-10, so that the class name is discernable to the data loader.

To download the MNIST or CIFAR10 dataset to the `data/` folder, simply run
```
python datasets/cifar10.py
python datasets/mnist.py
```
