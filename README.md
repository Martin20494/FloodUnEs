# FloodUnEs
A package for estimating uncertainty in flood model outputs caused by square grid orientation and bathymetry estimation

# Installation
To install, you can create a conda environment using the provided ```environment.yaml``` file.

For Windows:

```
conda env create -f environment.yaml
```

For Linux:

```
module purge
module load Miniconda3/23.10.0-1
conda env create -f environment.yaml
```

Then, depends on the types of your machine you can download cudatoolkit as instruction [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and then PyTorch package from [here](https://pytorch.org/get-started/locally/). The author used these two lines for downloading cudatoolkit version 11.8.0 and PyTorch accordingly for both Windows and Linux.

```
conda install cuda -c nvidia/label/cuda-11.8.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
